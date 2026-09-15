"""LofopQuery: the NMS-free member of the LOFOP-Detect family.

Same backbone and neck as :class:`~lofop.models.detector.LofopDetect`, but
the dense :class:`~lofop.models.head.ApexHead` is replaced by
:class:`~lofop.models.query_head.SlateHead`, a fixed slate of query slots
trained with one-to-one matching.

The practical difference is at inference: because training assigned exactly
one slot per object, the head does not produce duplicates, so ``predict``
does no suppression at all -- no IoU pass, no score decay, no peak test. It
thresholds and sorts. That makes post-processing a fixed cost independent of
scene density, and it exports to ONNX as plain tensor ops.

``LofopQuery`` deliberately reuses :class:`~lofop.models.neck.DeltaFusion`,
which already applies attention on the stride-32 level only. The head adds
cross-attention from 100 queries into the pyramid but no second self-attention
stack over the feature maps.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.matcher import PivotMatcher
from lofop.models.query_head import SlateHead, slate_losses
from lofop.registries import MODELS


@MODELS.register()
class LofopQuery(nn.Module):
    """Anchor-free, NMS-free set-prediction detector.

    Args:
        backbone: Feature extractor returning ``[C3, C4, C5]``.
        neck: Fusion module mapping those to ``[P3, P4, P5]``.
        head: A :class:`SlateHead`.
        image_size: Training/inference side length the head's normalised
            boxes are scaled by. ``predict`` overrides it from the actual
            input, so this is only the default.
        score_threshold: Minimum score for a detection at inference.
        max_detections: Cap on returned detections. Never exceeds the slate
            size, since the head cannot emit more boxes than it has slots.
        class_weight: Weight of the classification loss term.
        box_weight: Weight of the L1 box loss term.
        giou_weight: Weight of the GIoU loss term.
        matcher: Optional preconfigured :class:`PivotMatcher`; by default one
            is built with weights matching the loss weights, so matching and
            optimisation agree on what a good pairing is.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: SlateHead,
        image_size: int = 640,
        score_threshold: float = 0.25,
        max_detections: int = 100,
        class_weight: float = 2.0,
        box_weight: float = 5.0,
        giou_weight: float = 2.0,
        matcher: PivotMatcher | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(head, SlateHead):
            raise ModelError(
                "LofopQuery requires a SlateHead", context={"got": type(head).__name__}
            )
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.image_size = int(image_size)
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self.class_weight = float(class_weight)
        self.box_weight = float(box_weight)
        self.giou_weight = float(giou_weight)
        self.matcher = matcher or PivotMatcher(
            class_weight=class_weight, box_weight=box_weight, giou_weight=giou_weight
        )
        self._channels_last = False

    @property
    def num_classes(self) -> int:
        return self.head.num_classes

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Per-layer ``(class_logits, normalised_boxes)`` for a batch."""
        return self.head(self.neck(self.backbone(images)))

    def compute_losses(self, images: Tensor, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Set-prediction losses for a batch.

        Args:
            images: (B, 3, H, W) input batch.
            targets: Per image, ``{"boxes": (M, 4) xyxy pixels,
                "labels": (M,)}``.

        Returns:
            ``{"cls", "box", "giou", "total"}`` scalar tensors.
        """
        class_logits, boxes = self.forward(images)
        return slate_losses(
            class_logits,
            boxes,
            targets,
            self.matcher,
            float(images.shape[-1]),
            num_classes=self.head.num_classes,
            class_weight=self.class_weight,
            box_weight=self.box_weight,
            giou_weight=self.giou_weight,
        )

    def optimize_for_inference(self) -> LofopQuery:
        """Switch to eval mode and channels_last memory format, in place."""
        self.eval()
        self.to(memory_format=torch.channels_last)
        self._channels_last = True
        return self

    @torch.no_grad()
    def predict(self, images: Tensor) -> list[dict[str, Any]]:
        """Detect objects in a batch, with no suppression pass.

        Returns, per image: ``{"boxes": (K, 4) xyxy tensor, "scores": (K,),
        "labels": (K,)}`` sorted by score.
        """
        if self._channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        class_logits, boxes = self.forward(images)
        # Only the final decoder layer is read; earlier layers exist to be
        # supervised during training.
        scores_all = class_logits[:, -1].sigmoid()
        pixel_boxes = self.head.to_corners(boxes[:, -1], float(images.shape[-1]))

        results = []
        for image_index in range(images.shape[0]):
            scores, labels = scores_all[image_index].max(dim=1)
            keep = scores > self.score_threshold
            if not keep.any():
                results.append(self._empty_result(images))
                continue
            kept_scores = scores[keep]
            order = kept_scores.argsort(descending=True)
            if self.max_detections > 0:
                order = order[: self.max_detections]
            results.append(
                {
                    "boxes": pixel_boxes[image_index][keep][order],
                    "scores": kept_scores[order],
                    "labels": labels[keep][order],
                }
            )
        return results

    @staticmethod
    def _empty_result(images: Tensor) -> dict[str, Any]:
        device = images.device
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "scores": torch.zeros((0,), device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
        }
