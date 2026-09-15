"""LOFOP tracking: turn per-frame detections into persistent identities.

Three layers, each usable on its own:

* :mod:`lofop.tracking.motion` -- a constant-velocity Kalman filter over
  ``[cx, cy, w, h]`` and their velocities.
* :mod:`lofop.tracking.association` -- IoU cost matrices built on LOFOP's
  native C++/CUDA kernel, matched with LOFOP's own assignment solver.
* :mod:`lofop.tracking.tracker` -- :class:`LofopTracker`, which ties them
  together with confidence-tiered matching and track lifecycle management.

The association layer needs nothing beyond the LOFOP core. The motion model
uses numpy, so tracking ships as an opt-in extra::

    pip install "lofop[tracking]"

Importing this package never imports numpy: names are resolved on first
attribute access, so ``import lofop.tracking`` succeeds on a stock install
and only the pieces that genuinely need numpy raise, with the install command
in the message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "MotionEstimator": "lofop.tracking.motion",
    "to_centre_size": "lofop.tracking.motion",
    "to_corners": "lofop.tracking.motion",
    "associate": "lofop.tracking.association",
    "iou_cost_matrix": "lofop.tracking.association",
    "match": "lofop.tracking.association",
    "LofopTracker": "lofop.tracking.tracker",
    "Tracklet": "lofop.tracking.tracker",
    "TrackPhase": "lofop.tracking.tracker",
    "track_detections": "lofop.tracking.tracker",
    "to_supervision": "lofop.tracking.supervision_bridge",
    "from_supervision": "lofop.tracking.supervision_bridge",
    "tracks_to_detections": "lofop.tracking.supervision_bridge",
}

if TYPE_CHECKING:  # pragma: no cover - static analysers want the real names
    # Re-exported for type checkers only; __all__ is built from _EXPORTS above,
    # which ruff cannot follow, hence the explicit noqa markers.
    from lofop.tracking.association import (  # noqa: F401
        associate,
        iou_cost_matrix,
        match,
    )
    from lofop.tracking.motion import (  # noqa: F401
        MotionEstimator,
        to_centre_size,
        to_corners,
    )
    from lofop.tracking.supervision_bridge import (  # noqa: F401
        from_supervision,
        to_supervision,
        tracks_to_detections,
    )
    from lofop.tracking.tracker import (  # noqa: F401
        LofopTracker,
        Tracklet,
        TrackPhase,
        track_detections,
    )

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve public names on first use, keeping numpy off the import path."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return __all__
