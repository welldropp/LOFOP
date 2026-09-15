"""Bridge between LOFOP detections and the supervision ecosystem.

`supervision <https://pypi.org/project/supervision/>`_ is a widely used MIT
licensed toolkit for annotating, filtering and zoning detections. LOFOP does
not depend on it and does not vendor any of it -- this module only translates
between LOFOP's :class:`~lofop.deploy.postprocess.Detections` and the
structure supervision expects, so users who already have it keep their
annotators and zone logic.

Both supervision and numpy are imported inside the functions that need them.
A host without either keeps working; only these two calls raise, and they
raise with the exact install command.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from lofop.core.exceptions import LofopError

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from lofop.deploy.postprocess import Detections

_INSTALL_HINT = 'pip install "lofop[supervision]"'


def _require(module_name: str, extra_hint: str) -> Any:
    import importlib

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise LofopError(
            f"{module_name} is required for the supervision bridge. {extra_hint}"
        ) from exc


def to_supervision(detections: Detections) -> Any:
    """Convert LOFOP detections into a ``supervision.Detections``.

    Args:
        detections: A LOFOP :class:`~lofop.deploy.postprocess.Detections`.
            ``masks`` and ``tracker_ids`` are carried across when present.

    Returns:
        A ``supervision.Detections`` ready for supervision's annotators,
        zones and filters.

    Raises:
        LofopError: If supervision or numpy is not installed, naming the
            extra to install.
    """
    supervision = _require("supervision", _INSTALL_HINT)
    numpy = _require("numpy", _INSTALL_HINT)

    count = len(detections.boxes)
    if count == 0:
        return supervision.Detections.empty()

    payload: dict[str, Any] = {
        "xyxy": numpy.asarray(detections.boxes, dtype=numpy.float32).reshape(count, 4),
        "confidence": numpy.asarray(detections.scores, dtype=numpy.float32),
        "class_id": numpy.asarray(detections.labels, dtype=int),
    }

    tracker_ids = getattr(detections, "tracker_ids", None)
    if tracker_ids is not None:
        payload["tracker_id"] = numpy.asarray(tracker_ids, dtype=int)

    masks = getattr(detections, "masks", None)
    if masks is not None and len(masks) == count:
        # Torch tensors and nested sequences both land as a bool array.
        array = masks.cpu().numpy() if hasattr(masks, "cpu") else numpy.asarray(masks)
        payload["mask"] = array.astype(bool)

    return supervision.Detections(**payload)


def from_supervision(detections: Any) -> Detections:
    """Convert a ``supervision.Detections`` into LOFOP's structure.

    Args:
        detections: Any object exposing supervision's attribute names
            (``xyxy``, ``confidence``, ``class_id``, and optionally
            ``tracker_id`` / ``mask``).

    Returns:
        A LOFOP :class:`~lofop.deploy.postprocess.Detections`.
    """
    from lofop.deploy.postprocess import Detections as LofopDetections

    boxes_attr = getattr(detections, "xyxy", None)
    if boxes_attr is None:
        raise LofopError(
            "Object does not look like supervision.Detections (no 'xyxy')",
            context={"type": type(detections).__name__},
        )

    boxes = [[float(v) for v in box] for box in boxes_attr]
    count = len(boxes)

    confidence = getattr(detections, "confidence", None)
    scores = [1.0] * count if confidence is None else [float(v) for v in confidence]

    class_id = getattr(detections, "class_id", None)
    labels = [0] * count if class_id is None else [int(v) for v in class_id]

    tracker_id = getattr(detections, "tracker_id", None)
    tracker_ids = None if tracker_id is None else [int(v) for v in tracker_id]

    return LofopDetections(
        boxes=boxes,
        scores=scores,
        labels=labels,
        masks=getattr(detections, "mask", None),
        tracker_ids=tracker_ids,
    )


def tracks_to_detections(
    tracks: Sequence[Any],
) -> Detections:
    """Package tracker output as a LOFOP :class:`Detections` with ids.

    Args:
        tracks: :class:`~lofop.tracking.tracker.Tracklet` objects, as
            returned by :meth:`~lofop.tracking.tracker.LofopTracker.update`.
    """
    from lofop.deploy.postprocess import Detections as LofopDetections

    return LofopDetections(
        boxes=[track.box for track in tracks],
        scores=[float(track.score) for track in tracks],
        labels=[int(track.label) for track in tracks],
        tracker_ids=[int(track.track_id) for track in tracks],
    )
