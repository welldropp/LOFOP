"""SlateHead: NMS-free set prediction over a fixed slate of query slots.

Dense heads predict at every pyramid location, so many locations fire on one
object and a suppression pass is needed to clean up. A set-prediction head
instead keeps a fixed slate of Q learned query slots. Training assigns each
object to exactly one slot (see :class:`~lofop.models.matcher.PivotMatcher`),
which teaches the slate to divide the image between its slots -- so at
inference the head emits at most one box per object and **nothing is
suppressed**.

Structure, deliberately small enough for the edge tier:

* Queries read from the fused pyramid through cross-attention. LOFOP's
  :class:`~lofop.models.neck.DeltaFusion` neck already applies self-attention
  to the stride-32 level, so the expensive global mixing has happened before
  the head runs. SlateHead does not repeat it -- it adds no self-attention
  over feature maps at all, only cheap attention *from* Q queries (Q is 100,
  not thousands) *to* the flattened pyramid.
* Each decoder layer refines the boxes it already predicted, so later layers
  correct earlier ones rather than starting over.
* Boxes are emitted as normalised ``cxcywh`` in ``[0, 1]`` and scaled to
  pixels by the detector, keeping the head resolution-agnostic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.registries import HEADS


def _inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


class _DecoderLayer(nn.Module):
    """Query self-attention, then cross-attention into the pyramid."""

    def __init__(self, width: int, num_heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, width),
        )
        self.norm_self = nn.LayerNorm(width)
        self.norm_cross = nn.LayerNorm(width)
        self.norm_ff = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor, memory_pos: Tensor) -> Tensor:
        # Slots talk to each other first: this is what lets the slate divide
        # the image and stop two slots claiming one object.
        normed = self.norm_self(queries)
        attended, _ = self.self_attention(normed, normed, normed, need_weights=False)
        queries = queries + self.dropout(attended)

        normed = self.norm_cross(queries)
        attended, _ = self.cross_attention(
            normed, memory + memory_pos, memory, need_weights=False
        )
        queries = queries + self.dropout(attended)

        return queries + self.dropout(self.feedforward(self.norm_ff(queries)))


@HEADS.register()
class SlateHead(nn.Module):
    """Fixed-slate query head producing NMS-free detections.

    Args:
        num_classes: Number of object categories.
        width: Channel width of the incoming pyramid maps and of the queries.
        num_queries: Size of the slate. Must exceed the most objects any
            image contains; 100 comfortably covers ordinary scenes.
        num_layers: Decoder layers. Each refines the previous layer's boxes.
        num_heads: Attention heads; must divide ``width``.
        feedforward: Hidden width of the per-layer feed-forward block.
        dropout: Dropout inside the decoder.

    Forward takes the fused pyramid ``[P3, P4, P5]`` and returns
    ``(class_logits, boxes)`` where class logits are ``(B, L, Q, C)`` and
    boxes are ``(B, L, Q, 4)`` normalised ``cxcywh`` -- one entry per decoder
    layer ``L``, so training can supervise every layer and inference reads
    the last.
    """

    def __init__(
        self,
        num_classes: int,
        width: int = 96,
        num_queries: int = 100,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ModelError("num_classes must be >= 1", context={"num_classes": num_classes})
        if num_queries < 1:
            raise ModelError("num_queries must be >= 1", context={"num_queries": num_queries})
        if width % num_heads != 0:
            raise ModelError(
                "width must be divisible by num_heads",
                context={"width": width, "num_heads": num_heads},
            )
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.width = width

        self.query_embed = nn.Embedding(num_queries, width)
        # Each slot owns a learned reference point, so slots start out
        # looking at different parts of the image instead of collapsing.
        self.reference_points = nn.Embedding(num_queries, 4)
        self.level_embed = nn.Parameter(torch.zeros(3, width))
        self.input_norm = nn.LayerNorm(width)

        self.layers = nn.ModuleList(
            _DecoderLayer(width, num_heads, feedforward, dropout) for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.class_head = nn.Linear(width, num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(inplace=True), nn.Linear(width, 4)
        )

        nn.init.normal_(self.query_embed.weight, std=0.02)
        nn.init.uniform_(self.reference_points.weight, -2.0, 2.0)
        nn.init.normal_(self.level_embed, std=0.02)
        # Focal-loss convention: start every class near 1% probability so the
        # first steps are not dominated by the empty slate.
        nn.init.constant_(self.class_head.bias, -4.595)
        nn.init.zeros_(self.box_head[-1].weight)
        nn.init.zeros_(self.box_head[-1].bias)

    def _flatten_pyramid(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Concatenate levels into (B, N, width) memory plus a level encoding."""
        if len(features) != 3:
            raise ModelError(
                "SlateHead expects exactly 3 pyramid levels",
                context={"features": len(features)},
            )
        tokens, positions = [], []
        for level, feature in enumerate(features):
            if feature.shape[1] != self.width:
                raise ModelError(
                    "Pyramid width does not match head width",
                    context={"expected": self.width, "got": int(feature.shape[1])},
                )
            batch, channels, height, wide = feature.shape
            flat = feature.flatten(2).transpose(1, 2)
            tokens.append(flat)
            positions.append(
                self.level_embed[level].view(1, 1, channels).expand(batch, height * wide, channels)
            )
        return self.input_norm(torch.cat(tokens, dim=1)), torch.cat(positions, dim=1)

    def forward(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        memory, memory_pos = self._flatten_pyramid(features)
        batch = memory.shape[0]

        queries = self.query_embed.weight.unsqueeze(0).expand(batch, -1, -1)
        reference = self.reference_points.weight.unsqueeze(0).expand(batch, -1, -1)

        class_outputs, box_outputs = [], []
        for layer in self.layers:
            queries = layer(queries, memory, memory_pos)
            normed = self.output_norm(queries)
            # Residual refinement: each layer nudges the running reference
            # rather than regressing a box from nothing.
            reference = reference + self.box_head(normed)
            class_outputs.append(self.class_head(normed))
            box_outputs.append(reference.sigmoid())
        return torch.stack(class_outputs, dim=1), torch.stack(box_outputs, dim=1)

    @staticmethod
    def to_corners(boxes: Tensor, image_size: float) -> Tensor:
        """Normalised ``cxcywh`` in ``[0, 1]`` to pixel ``xyxy``."""
        centre_x, centre_y, width, height = boxes.unbind(-1)
        half_w, half_h = width / 2.0, height / 2.0
        return torch.stack(
            [centre_x - half_w, centre_y - half_h, centre_x + half_w, centre_y + half_h],
            dim=-1,
        ) * float(image_size)

    @staticmethod
    def to_normalised(boxes: Tensor, image_size: float) -> Tensor:
        """Pixel ``xyxy`` to normalised ``cxcywh`` in ``[0, 1]``."""
        scaled = boxes / float(image_size)
        x1, y1, x2, y2 = scaled.unbind(-1)
        return torch.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=-1)

    def initialise_reference_from(self, boxes: Tensor) -> None:
        """Seed the slate's reference points from a box prior.

        Args:
            boxes: (num_queries, 4) normalised ``cxcywh`` priors, for
                instance a k-means summary of a dataset's box distribution.
                Converted to logit space so the forward pass's sigmoid
                reproduces them.
        """
        if boxes.shape != (self.num_queries, 4):
            raise ModelError(
                "Reference prior must be (num_queries, 4)",
                context={"expected": [self.num_queries, 4], "got": list(boxes.shape)},
            )
        with torch.no_grad():
            self.reference_points.weight.copy_(_inverse_sigmoid(boxes))


def slate_losses(
    class_logits: Tensor,
    boxes_normalised: Tensor,
    targets: list[dict[str, Tensor]],
    matcher: object,
    image_size: float,
    *,
    num_classes: int,
    class_weight: float = 2.0,
    box_weight: float = 5.0,
    giou_weight: float = 2.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> dict[str, Tensor]:
    """Set-prediction losses, summed over every decoder layer.

    Args:
        class_logits: (B, L, Q, C) per-layer class logits.
        boxes_normalised: (B, L, Q, 4) per-layer normalised ``cxcywh`` boxes.
        targets: Per image, ``{"boxes": (M, 4) xyxy pixels, "labels": (M,)}``.
        matcher: A :class:`~lofop.models.matcher.PivotMatcher`.
        image_size: Side length the pixel boxes are expressed in.
        num_classes: Category count, for the one-hot classification target.
        class_weight: Weight of the classification term.
        box_weight: Weight of the L1 box term.
        giou_weight: Weight of the GIoU term.
        focal_alpha: Focal balancing factor.
        focal_gamma: Focal focusing exponent.

    Returns:
        ``{"cls", "box", "giou", "total"}`` scalar tensors. Every decoder
        layer is supervised, which is what makes the intermediate layers
        useful refiners rather than dead weight.
    """
    from lofop.models.matcher import _generalized_iou

    device = class_logits.device
    batch, layers = class_logits.shape[0], class_logits.shape[1]
    total_cls = class_logits.new_zeros(())
    total_box = class_logits.new_zeros(())
    total_giou = class_logits.new_zeros(())
    matched_objects = 0

    for image_index in range(batch):
        target = targets[image_index]
        target_boxes = target["boxes"].to(device)
        target_labels = target["labels"].to(device)
        for layer_index in range(layers):
            logits = class_logits[image_index, layer_index]
            normalised = boxes_normalised[image_index, layer_index]
            pixel_boxes = SlateHead.to_corners(normalised, image_size)

            slot_index, target_index = matcher.match(
                logits, pixel_boxes, target_boxes, target_labels, image_size
            )

            class_target = torch.zeros_like(logits)
            if slot_index.numel() > 0:
                class_target[slot_index, target_labels[target_index]] = 1.0
                matched_normalised = normalised[slot_index]
                gt_normalised = SlateHead.to_normalised(target_boxes[target_index], image_size)
                total_box = total_box + F.l1_loss(
                    matched_normalised, gt_normalised, reduction="sum"
                )
                giou = _generalized_iou(pixel_boxes[slot_index], target_boxes[target_index])
                total_giou = total_giou + (1.0 - giou.diagonal()).sum()
                if layer_index == layers - 1:
                    matched_objects += int(slot_index.numel())

            probability = logits.sigmoid()
            focal_weight = torch.where(
                class_target > 0.5,
                focal_alpha * (1.0 - probability) ** focal_gamma,
                (1.0 - focal_alpha) * probability**focal_gamma,
            )
            total_cls = total_cls + (
                focal_weight
                * F.binary_cross_entropy_with_logits(logits, class_target, reduction="none")
            ).sum()

    # Normalise by matched objects so loss scale is independent of how many
    # objects a batch happens to contain.
    norm = max(matched_objects, 1)
    losses = {
        "cls": class_weight * total_cls / norm,
        "box": box_weight * total_box / norm,
        "giou": giou_weight * total_giou / norm,
    }
    losses["total"] = losses["cls"] + losses["box"] + losses["giou"]
    return losses
