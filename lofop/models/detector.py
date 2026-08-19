"""LofopDetect: the LOFOP flagship detector.

Assembles a registered backbone, neck, and head; computes training losses via
dynamic label assignment; decodes predictions through the native C++
class-aware NMS. Everything is config-driven -- see lofop/configs/lofop-detect/.
Design: docs/lofop-detect.md.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.assigner import DynamicTopKAssigner
from lofop.models.head import ApexHead
from lofop.models.losses import giou_loss, sigmoid_focal_loss
from lofop.ops import nms, soft_nms
from lofop.ops.boxes import CLASS_OFFSET
from lofop.registries import MODELS


def flatten_levels(maps: list[Tensor], channels: int) -> Tensor:
    """Concatenate per-level (B, C, H, W) maps into (B, N, C) location rows.

    Same location ordering as :meth:`ApexHead.level_points`, so row ``i``
    of any flattened map corresponds to point ``i``. Shared by the task
    variants (segmentation coefficients, keypoint offsets).
    """
    batch = maps[0].shape[0]
    return torch.cat(
        [level.permute(0, 2, 3, 1).reshape(batch, -1, channels) for level in maps], dim=1
    )


@MODELS.register()
class LofopDetect(nn.Module):
    """Anchor-free multi-scale detector.

    Args:
        backbone: Feature extractor returning ``[C3, C4, C5]`` (typically a
            built ``backbone/RidgeNet`` spec).
        neck: Fusion module mapping those to ``[P3, P4, P5]``.
        head: Dense prediction head (an :class:`ApexHead`).
        box_weight: Weight of the GIoU loss term.
        score_threshold: Minimum score for a detection at inference.
        nms_iou: IoU threshold for class-aware NMS.
        max_detections: Detections kept per image after NMS.
        nms_mode: Duplicate-removal strategy at inference. ``"greedy"``
            (default) is classic class-aware NMS; ``"soft"`` decays
            overlapping scores instead of dropping (better in crowds);
            ``"free"`` is NMS-free -- keep only locations that are the local
            score peak of their 3x3 neighborhood on their pyramid level, so
            inference is pure tensor math with no suppression loop at all.
        soft_nms_sigma: Gaussian decay width for ``nms_mode="soft"``.
    """

    _NMS_MODES = ("greedy", "soft", "free")

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: ApexHead,
        box_weight: float = 2.0,
        score_threshold: float = 0.05,
        nms_iou: float = 0.6,
        max_detections: int = 300,
        nms_mode: str = "greedy",
        soft_nms_sigma: float = 0.5,
    ) -> None:
        super().__init__()
        if not isinstance(head, ApexHead):
            got = type(head).__name__
            raise ModelError("LofopDetect requires an ApexHead", context={"got": got})
        if nms_mode not in self._NMS_MODES:
            raise ModelError(
                "Unknown nms_mode", context={"got": nms_mode, "known": list(self._NMS_MODES)}
            )
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.assigner = DynamicTopKAssigner()
        self.box_weight = box_weight
        self.score_threshold = score_threshold
        self.nms_iou = nms_iou
        self.max_detections = max_detections
        self.nms_mode = nms_mode
        self.soft_nms_sigma = soft_nms_sigma
        self._channels_last = False

    def _features(self, images: Tensor) -> list[Tensor]:
        """Fused pyramid features ``[P3, P4, P5]`` for a batch of images."""
        return self.neck(self.backbone(images))

    def forward(self, images: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Raw per-level head outputs for a batch of images."""
        return self.head(self._features(images))

    def _flatten(
        self, cls_out: list[Tensor], box_out: list[Tensor], quality_out: list[Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Concatenate per-level maps into per-image rows: (B, N, ...)."""
        batch = cls_out[0].shape[0]
        cls = torch.cat(
            [level.permute(0, 2, 3, 1).reshape(batch, -1, self.head.num_classes)
             for level in cls_out], dim=1)
        box = torch.cat(
            [level.permute(0, 2, 3, 1).reshape(batch, -1, 4) for level in box_out], dim=1)
        quality = torch.cat(
            [level.permute(0, 2, 3, 1).reshape(batch, -1) for level in quality_out], dim=1)
        return cls, box, quality

    def compute_losses(self, images: Tensor, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Training losses for a batch.

        Args:
            images: (B, 3, H, W) input batch.
            targets: Per image: ``{"boxes": (M, 4) xyxy pixels, "labels": (M,)}``.

        Returns:
            ``{"cls": ..., "box": ..., "quality": ..., "total": ...}`` scalar
            tensors (box/quality are zero when the batch has no objects).
        """
        outputs = self.forward(images)
        losses, _ = self._detection_losses(outputs, images, targets)
        return losses

    def _detection_losses(
        self,
        outputs: tuple[list[Tensor], list[Tensor], list[Tensor]],
        images: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
        """Detection losses plus per-image assignment aux for task variants.

        The aux list carries, per image, the flattened-location ``positive``
        mask and ``assigned`` ground-truth indices so subclasses (segmentation,
        pose) can supervise their extra branches on the same assignment
        without re-running the assigner.
        """
        cls_out, box_out, quality_out = outputs
        points, strides = self.head.level_points(cls_out)
        cls, box, quality = self._flatten(cls_out, box_out, quality_out)

        total_cls = images.new_zeros(())
        total_box = images.new_zeros(())
        total_quality = images.new_zeros(())
        total_pos = 0
        aux: list[dict[str, Tensor]] = []
        for image_index, target in enumerate(targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            decoded = self.head.decode_boxes(points, box[image_index])
            assigned, iou_targets = self.assigner.assign(
                points, strides, cls[image_index].detach().sigmoid(), decoded.detach(),
                gt_boxes, gt_labels,
            )
            positive = assigned >= 0
            aux.append({"positive": positive, "assigned": assigned})
            cls_targets = torch.zeros_like(cls[image_index])
            if positive.any():
                cls_targets[positive, gt_labels[assigned[positive]]] = 1.0
                total_box = total_box + giou_loss(
                    decoded[positive], gt_boxes[assigned[positive]]
                ).sum()
                total_pos += int(positive.sum())
            # Quality on positives: plain BCE against assigned IoU (full
            # weight -- this drives the localization-quality estimate).
            # Quality on background: a separate, gently weighted calibration
            # term pushing it toward 0. Without it background quality is
            # unsupervised/arbitrary and inflates the fused score
            # sqrt(cls * quality), producing low-confidence false positives.
            # The sigmoid^2 modulation and the 0.25 factor keep the thousands
            # of easy negatives from competing with the few positives for the
            # quality branch's gradient budget (measured: full-weight
            # background supervision costs ~4 mAP on the fixed benchmark).
            q_logits = quality[image_index]
            if positive.any():
                total_quality = total_quality + F.binary_cross_entropy_with_logits(
                    q_logits[positive], iou_targets[positive], reduction="sum"
                )
            negative = ~positive
            neg_logits = q_logits[negative]
            neg_bce = F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none"
            )
            neg_modulation = neg_logits.detach().sigmoid() ** 2
            total_quality = total_quality + 0.25 * (neg_modulation * neg_bce).sum()
            total_cls = total_cls + sigmoid_focal_loss(cls[image_index], cls_targets).sum()

        norm = max(total_pos, 1)
        losses = {
            "cls": total_cls / norm,
            "box": self.box_weight * total_box / norm,
            "quality": total_quality / norm,
        }
        losses["total"] = losses["cls"] + losses["box"] + losses["quality"]
        return losses, aux

    def optimize_for_inference(self) -> LofopDetect:
        """Switch to eval mode and channels_last memory format, in place.

        channels_last routes the depthwise convolutions onto the fast oneDNN
        CPU path (measured 1.6x forward speedup at 640px, neutral at 128px;
        outputs identical to float tolerance). ``predict`` converts inputs to
        match automatically afterwards. Opt-in because training and export
        paths expect the default layout.
        """
        self.eval()
        self.to(memory_format=torch.channels_last)
        self._channels_last = True
        return self

    @torch.no_grad()
    def predict(self, images: Tensor) -> list[dict[str, Any]]:
        """Detect objects in a batch.

        Returns, per image: ``{"boxes": (K, 4) xyxy tensor, "scores": (K,),
        "labels": (K,)}`` sorted by score, at most ``max_detections`` each.
        """
        if self._channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        results = self._decode_batch(self.forward(images), images)
        for result in results:
            result.pop("locations")
        return results

    def _decode_batch(
        self,
        outputs: tuple[list[Tensor], list[Tensor], list[Tensor]],
        images: Tensor,
    ) -> list[dict[str, Any]]:
        """Threshold + class-aware NMS for a batch of raw head outputs.

        Each result additionally carries ``locations`` -- the flattened
        pyramid-location index of every kept detection -- which task variants
        use to gather their per-location extras (mask coefficients, keypoint
        offsets); :meth:`predict` strips it from the public output.
        """
        cls_out, box_out, quality_out = outputs
        points, _ = self.head.level_points(cls_out)
        cls, box, quality = self._flatten(cls_out, box_out, quality_out)
        scores_all = (cls.sigmoid() * quality.sigmoid().unsqueeze(-1)).sqrt()
        peaks = self._peak_mask(cls_out, quality_out) if self.nms_mode == "free" else None

        results = []
        for image_index in range(images.shape[0]):
            scores, labels = scores_all[image_index].max(dim=1)
            keep_mask = scores > self.score_threshold
            if peaks is not None:
                keep_mask &= peaks[image_index]
            if not keep_mask.any():
                results.append(self._empty_result(images))
                continue
            location_index = keep_mask.nonzero(as_tuple=False).squeeze(1)
            boxes = self.head.decode_boxes(points[keep_mask], box[image_index][keep_mask])
            scores, labels = scores[keep_mask], labels[keep_mask]
            if self.nms_mode == "free":
                # Peak selection already removed duplicates; just rank.
                index = scores.argsort(descending=True)
                if self.max_detections > 0:
                    index = index[: self.max_detections]
            elif self.nms_mode == "soft":
                shifted = boxes + (labels.to(boxes.dtype) * CLASS_OFFSET).unsqueeze(1)
                keep, kept_scores = soft_nms(
                    shifted.contiguous(), scores.contiguous(),
                    sigma=self.soft_nms_sigma, score_threshold=self.score_threshold,
                    max_keep=self.max_detections,
                )
                index = torch.as_tensor(keep, dtype=torch.long, device=images.device)
                scores = torch.as_tensor(
                    kept_scores, dtype=boxes.dtype, device=images.device
                )
                results.append({
                    "boxes": boxes[index], "scores": scores, "labels": labels[index],
                    "locations": location_index[index],
                })
                continue
            else:
                # Class-aware NMS via the coordinate-offset shift done in
                # tensor math; contiguous float32 tensors hit the zero-copy
                # native path in lofop.ops instead of a per-element Python
                # list conversion.
                shifted = boxes + (labels.to(boxes.dtype) * CLASS_OFFSET).unsqueeze(1)
                keep = nms(
                    shifted.contiguous(), scores.contiguous(),
                    iou_threshold=self.nms_iou, max_keep=self.max_detections,
                )
                index = torch.as_tensor(keep, dtype=torch.long, device=images.device)
            results.append({
                "boxes": boxes[index], "scores": scores[index], "labels": labels[index],
                "locations": location_index[index],
            })
        return results

    def _peak_mask(self, cls_out: list[Tensor], quality_out: list[Tensor]) -> Tensor:
        """(B, N) mask of locations that are their 3x3 neighborhood's score
        peak on their own pyramid level -- the NMS-free selection rule."""
        masks = []
        for cls_map, quality_map in zip(cls_out, quality_out):
            fused = (cls_map.sigmoid().amax(dim=1, keepdim=True)
                     * quality_map.sigmoid()).sqrt()
            pooled = F.max_pool2d(fused, kernel_size=3, stride=1, padding=1)
            masks.append((fused >= pooled).squeeze(1).flatten(1))
        return torch.cat(masks, dim=1)

    @staticmethod
    def _empty_result(images: Tensor) -> dict[str, Any]:
        device = images.device
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "scores": torch.zeros((0,), device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
            "locations": torch.zeros((0,), dtype=torch.long, device=device),
        }
