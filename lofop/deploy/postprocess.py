"""Post-processing for dense exported-model outputs.

Torch-free companion to the ONNX graph: takes the dense ``(boxes, scores)``
arrays an exported LOFOP model produces, applies score thresholding and
LOFOP's native class-aware NMS, and returns final detections. Depends only
on the core framework (the C++ ops fast path applies automatically), so it
runs next to onnxruntime, TensorRT, or OpenVINO without a PyTorch install.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lofop.ops import batched_nms, decode_dense, soft_nms
from lofop.ops.boxes import CLASS_OFFSET


@dataclass(frozen=True)
class Detections:
    """Final detections for one image.

    Attributes:
        boxes: (K, 4) xyxy boxes in input pixels.
        scores: K confidence scores, descending.
        labels: K class indices.
        masks: Optional per-detection masks -- a (K, H, W) bool tensor when
            produced by a segmentation model, else ``None``.
        keypoints: Optional per-detection keypoints -- (K, num_keypoints, 3)
            ``(x, y, visibility)`` rows when produced by a pose model, else
            ``None``.
        tracker_ids: Optional K persistent integer tracking IDs when produced
            by a tracker or video inference, else ``None``.
    """

    boxes: list[list[float]]
    scores: list[float]
    labels: list[int]
    masks: Any = None
    keypoints: Any = None
    tracker_ids: list[int] | None = None

    def __len__(self) -> int:
        return len(self.scores)

    def to_supervision(self, class_names: Sequence[str] | None = None) -> Any:
        """Convert to a supervision.Detections object."""
        from lofop.tracking.adapters import to_supervision

        return to_supervision(self, class_names=class_names)


def postprocess_dense(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[Sequence[float]],
    *,
    score_threshold: float = 0.25,
    nms_iou: float = 0.6,
    max_detections: int = 300,
    soft: bool = False,
    soft_sigma: float = 0.5,
) -> Detections:
    """Turn one image's dense model outputs into final detections.

    Args:
        boxes: (N, 4) decoded xyxy boxes (the ONNX ``boxes`` output, one
            image; numpy arrays and nested lists both work).
        scores: (N, C) per-class scores (the ONNX ``scores`` output).
        score_threshold: Minimum class score to consider a candidate.
        nms_iou: IoU threshold for class-aware NMS.
        max_detections: Cap on returned detections.
        soft: Use class-aware Soft-NMS (gaussian decay) instead of greedy
            NMS; better in crowded scenes. Returned scores are the decayed
            ones.
        soft_sigma: Gaussian decay width for Soft-NMS.
    """
    indices, candidate_labels, _ = decode_dense(scores, score_threshold=score_threshold)
    candidate_boxes = [[float(v) for v in boxes[i]] for i in indices]
    # Re-read the winning scores from the input rows: the native kernel works
    # in float32, and callers expect their exact values back.
    candidate_scores = [float(scores[i][lab]) for i, lab in zip(indices, candidate_labels)]
    if soft:
        shifted = [
            [v + CLASS_OFFSET * label for v in box]
            for box, label in zip(candidate_boxes, candidate_labels)
        ]
        keep, kept_scores = soft_nms(
            shifted, candidate_scores, sigma=soft_sigma,
            score_threshold=score_threshold, max_keep=max_detections,
        )
        return Detections(
            boxes=[candidate_boxes[i] for i in keep],
            scores=kept_scores,
            labels=[candidate_labels[i] for i in keep],
        )
    keep = batched_nms(
        candidate_boxes, candidate_scores, candidate_labels,
        iou_threshold=nms_iou, max_keep=max_detections,
    )
    return Detections(
        boxes=[candidate_boxes[i] for i in keep],
        scores=[candidate_scores[i] for i in keep],
        labels=[candidate_labels[i] for i in keep],
    )
