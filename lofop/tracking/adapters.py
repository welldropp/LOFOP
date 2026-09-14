"""Adapters for third-party vision ecosystems, notably Roboflow supervision.

Provides bidirectional conversion between LOFOP Detections and supervision.Detections.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lofop.deploy.postprocess import Detections


def to_supervision(detections: Detections, class_names: Sequence[str] | None = None) -> Any:
    """Convert LOFOP Detections to a supervision.Detections instance.

    Args:
        detections: A LOFOP :class:`~lofop.deploy.postprocess.Detections` object.
        class_names: Optional sequence of human-readable class names mapping labels to strings.

    Returns:
        A ``supervision.Detections`` object ready for use with ByteTrack, PolygonZone,
        BoxAnnotator, TraceAnnotator, etc.

    Raises:
        ImportError: If the ``supervision`` package is not installed.
    """
    try:
        import supervision as sv
    except ImportError as err:
        raise ImportError(
            "The 'supervision' library is required to convert detections. "
            "Please install it via: pip install supervision"
        ) from err

    n = len(detections.boxes)
    if n > 0:
        xyxy = np.asarray(detections.boxes, dtype=np.float32)
        confidence = np.asarray(detections.scores, dtype=np.float32)
        class_id = np.asarray(detections.labels, dtype=np.int32)
    else:
        xyxy = np.empty((0, 4), dtype=np.float32)
        confidence = np.empty((0,), dtype=np.float32)
        class_id = np.empty((0,), dtype=np.int32)

    tracker_id = None
    if getattr(detections, "tracker_ids", None) is not None:
        tracker_id = np.asarray(detections.tracker_ids, dtype=np.int32)

    mask = None
    if getattr(detections, "masks", None) is not None:
        masks = detections.masks
        if hasattr(masks, "cpu"):
            masks = masks.cpu().numpy()
        mask = np.asarray(masks, dtype=bool)

    data: dict[str, Any] = {}
    if class_names is not None and len(class_id) > 0:
        data["class_name"] = np.array([
            class_names[idx] if 0 <= idx < len(class_names) else str(idx)
            for idx in class_id
        ])

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
        mask=mask,
        data=data,
    )


def from_supervision(sv_detections: Any) -> Detections:
    """Convert a supervision.Detections instance to a LOFOP Detections object.

    Args:
        sv_detections: A ``supervision.Detections`` object.

    Returns:
        A LOFOP :class:`~lofop.deploy.postprocess.Detections` object.
    """
    from lofop.deploy.postprocess import Detections

    boxes = sv_detections.xyxy.tolist() if len(sv_detections.xyxy) else []
    scores = (
        sv_detections.confidence.tolist()
        if sv_detections.confidence is not None
        else [1.0] * len(boxes)
    )
    labels = (
        sv_detections.class_id.tolist()
        if sv_detections.class_id is not None
        else [0] * len(boxes)
    )
    tracker_ids = (
        sv_detections.tracker_id.tolist()
        if sv_detections.tracker_id is not None
        else None
    )
    masks = sv_detections.mask

    return Detections(
        boxes=boxes,
        scores=scores,
        labels=labels,
        masks=masks,
        tracker_ids=tracker_ids,
    )
