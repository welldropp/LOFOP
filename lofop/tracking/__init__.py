"""LOFOP Tracking: temporal state estimation and multi-object tracking.

Provides:
  - :class:`KalmanFilter`: 8D state-space motion model.
  - :class:`ByteTracker`: Two-stage multi-object association tracker.
  - :class:`STrack`: Single-target tracklet representation.
  - :class:`TrackState`: Lifecycle state enum (New, Tracked, Lost, Removed).
  - :func:`to_supervision`: Universal adapter to Roboflow supervision.
  - :func:`from_supervision`: Reverse adapter from supervision to LOFOP.
"""

from lofop.tracking.adapters import from_supervision, to_supervision
from lofop.tracking.byte_tracker import ByteTracker, STrack, TrackState
from lofop.tracking.kalman import KalmanFilter

__all__ = [
    "KalmanFilter",
    "ByteTracker",
    "STrack",
    "TrackState",
    "to_supervision",
    "from_supervision",
]
