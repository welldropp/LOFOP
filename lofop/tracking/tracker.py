"""LofopTracker: confidence-tiered multi-object tracking.

Detectors emit a long tail of low-confidence boxes. Most are noise, but some
are real objects that briefly became hard to see -- motion-blurred, partly
occluded, turned away. Discarding everything below the display threshold
throws those away and fragments tracks; keeping everything invites false
positives.

LOFOP resolves this by matching in two tiers per frame:

1. **Strong tier.** Confident detections are matched against every live
   track. These are trusted enough to both continue a track and open one.
2. **Weak tier.** Detections below the confidence bar are matched only
   against tracks left unmatched by the strong tier. A weak detection can
   therefore *sustain* an existing track through a difficult moment, but it
   can never *start* one -- so the noise tail never manufactures identities.

Tracks age through :class:`TrackPhase`. A new track is ``Tentative`` until it
has been confirmed on enough consecutive frames, which suppresses one-frame
detector flickers. A track that stops matching becomes ``Dormant`` and keeps
being predicted forward for a grace period, so a re-appearing object recovers
its original id instead of being issued a new one.

Motion comes from :class:`~lofop.tracking.motion.MotionEstimator` and
matching from :mod:`lofop.tracking.association`, which runs on LOFOP's native
IoU kernel.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

from lofop.core.exceptions import LofopError
from lofop.tracking.association import associate
from lofop.tracking.motion import MotionEstimator, to_centre_size, to_corners

BoxList = Sequence[Sequence[float]]


class TrackPhase(Enum):
    """Lifecycle phase of a tracklet."""

    Tentative = "tentative"
    """Seen, but not yet confirmed on enough frames to be reported."""

    Active = "active"
    """Confirmed and matched recently; reported to the caller."""

    Dormant = "dormant"
    """Unmatched for at least one frame; predicted forward awaiting recovery."""

    Ended = "ended"
    """Dormant past the grace period; removed and never revived."""


class Tracklet:
    """One tracked object: an identity plus its motion state.

    Attributes:
        track_id: Stable integer identity, unique within a tracker.
        label: Class index carried from the detection that opened the track.
        score: Confidence of the most recent matching detection.
        phase: Current :class:`TrackPhase`.
        age: Frames since the track was created.
        hits: Total number of frames on which it matched a detection.
        time_since_match: Frames since the last match (0 on a matched frame).
    """

    __slots__ = (
        "track_id", "label", "score", "phase", "age", "hits",
        "time_since_match", "_mean", "_covariance", "_estimator",
    )

    def __init__(
        self,
        track_id: int,
        box: Sequence[float],
        score: float,
        label: int,
        estimator: MotionEstimator,
    ) -> None:
        self.track_id = int(track_id)
        self.label = int(label)
        self.score = float(score)
        self.phase = TrackPhase.Tentative
        self.age = 0
        self.hits = 1
        self.time_since_match = 0
        self._estimator = estimator
        self._mean, self._covariance = estimator.initiate(to_centre_size(box))

    @property
    def box(self) -> list[float]:
        """Current estimated box in ``xyxy``."""
        return to_corners([float(v) for v in self._mean[:4]])

    def predict(self) -> None:
        """Advance the motion state by one frame."""
        self._mean, self._covariance = self._estimator.predict(self._mean, self._covariance)
        self.age += 1
        self.time_since_match += 1

    def correct(self, box: Sequence[float], score: float, confirm_after: int) -> None:
        """Fold in a matched detection and refresh the phase."""
        self._mean, self._covariance = self._estimator.correct(
            self._mean, self._covariance, to_centre_size(box)
        )
        self.score = float(score)
        self.hits += 1
        self.time_since_match = 0
        if self.phase is TrackPhase.Tentative and self.hits >= confirm_after:
            self.phase = TrackPhase.Active
        elif self.phase is TrackPhase.Dormant:
            self.phase = TrackPhase.Active

    def mark_missed(self, grace_frames: int, confirm_after: int) -> None:
        """Handle a frame on which this track found no detection."""
        if self.phase is TrackPhase.Tentative and self.hits < confirm_after:
            # An unconfirmed track that misses even once was probably a
            # detector flicker; do not spend a grace period on it.
            self.phase = TrackPhase.Ended
        elif self.time_since_match > grace_frames:
            self.phase = TrackPhase.Ended
        else:
            self.phase = TrackPhase.Dormant

    def __repr__(self) -> str:
        box = ", ".join(f"{v:.1f}" for v in self.box)
        return (
            f"Tracklet(id={self.track_id}, label={self.label}, "
            f"phase={self.phase.value}, box=[{box}])"
        )


class LofopTracker:
    """Confidence-tiered tracker over per-frame detections.

    Args:
        strong_threshold: Detections at or above this score form the strong
            tier: they can continue and open tracks.
        weak_threshold: Detections between this and ``strong_threshold``
            form the weak tier: they can only continue an existing track.
            Anything below is discarded.
        max_cost: Association ceiling as ``1 - IoU``; ``0.8`` accepts
            pairings down to 20% overlap.
        weak_max_cost: Separate, stricter ceiling for the weak tier -- a
            low-confidence box must align more convincingly before it is
            allowed to sustain a track.
        grace_frames: Frames a confirmed track survives without a match
            before ending. Scale with frame rate: the default suits ~30 fps.
        confirm_after: Matching frames a new track needs before it is
            reported.
        class_aware: Forbid associations between different class ids.
        optimal: Use the exact assignment solver (default) rather than the
            cheaper greedy pass.
        estimator: Optional preconfigured :class:`MotionEstimator`.

    Typical use is one :meth:`update` per frame::

        tracker = LofopTracker()
        for detections in stream:
            tracks = tracker.update(
                detections.boxes, detections.scores, detections.labels
            )
    """

    def __init__(
        self,
        *,
        strong_threshold: float = 0.5,
        weak_threshold: float = 0.1,
        max_cost: float = 0.8,
        weak_max_cost: float = 0.5,
        grace_frames: int = 30,
        confirm_after: int = 3,
        class_aware: bool = True,
        optimal: bool = True,
        estimator: MotionEstimator | None = None,
    ) -> None:
        if weak_threshold > strong_threshold:
            raise LofopError(
                "weak_threshold must not exceed strong_threshold",
                context={"weak": weak_threshold, "strong": strong_threshold},
            )
        self.strong_threshold = float(strong_threshold)
        self.weak_threshold = float(weak_threshold)
        self.max_cost = float(max_cost)
        self.weak_max_cost = float(weak_max_cost)
        self.grace_frames = int(grace_frames)
        self.confirm_after = int(confirm_after)
        self.class_aware = bool(class_aware)
        self.optimal = bool(optimal)
        # Constructing the estimator imports numpy, so a tracker built on a
        # host without the tracking extra fails here with a clear message
        # rather than midway through the first frame.
        self.estimator = estimator or MotionEstimator()
        self.tracks: list[Tracklet] = []
        self.frame_index = 0
        self._next_id = 1

    def reset(self) -> None:
        """Drop all tracks and restart identity numbering."""
        self.tracks = []
        self.frame_index = 0
        self._next_id = 1

    def update(
        self,
        boxes: BoxList,
        scores: Sequence[float],
        labels: Sequence[int] | None = None,
    ) -> list[Tracklet]:
        """Consume one frame of detections and return the reportable tracks.

        Args:
            boxes: Detection boxes, xyxy, in image pixels.
            scores: Confidence per detection.
            labels: Class index per detection; defaults to class 0.

        Returns:
            The tracks in :attr:`TrackPhase.Active` after this frame, ordered
            by descending score. Tentative and dormant tracks are kept
            internally but not returned.
        """
        boxes = [list(map(float, box)) for box in boxes]
        scores = [float(value) for value in scores]
        if len(boxes) != len(scores):
            raise LofopError(
                "boxes and scores must have equal length",
                context={"boxes": len(boxes), "scores": len(scores)},
            )
        labels = [0] * len(boxes) if labels is None else [int(value) for value in labels]
        if len(labels) != len(boxes):
            raise LofopError(
                "labels must match the number of boxes",
                context={"boxes": len(boxes), "labels": len(labels)},
            )

        self.frame_index += 1
        for track in self.tracks:
            track.predict()

        strong = [i for i, score in enumerate(scores) if score >= self.strong_threshold]
        weak = [
            i
            for i, score in enumerate(scores)
            if self.weak_threshold <= score < self.strong_threshold
        ]

        live = [t for t in self.tracks if t.phase is not TrackPhase.Ended]
        matched_pairs, unmatched_track_slots, unmatched_strong = self._match_tier(
            live, strong, boxes, labels, self.max_cost
        )
        for track_slot, detection_index in matched_pairs:
            live[track_slot].correct(
                boxes[detection_index], scores[detection_index], self.confirm_after
            )

        # Second tier: only tracks that went unmatched above are eligible, so
        # a weak detection can never outbid a confident one.
        remaining = [live[slot] for slot in unmatched_track_slots]
        weak_pairs, still_unmatched_slots, _ = self._match_tier(
            remaining, weak, boxes, labels, self.weak_max_cost
        )
        for track_slot, detection_index in weak_pairs:
            remaining[track_slot].correct(
                boxes[detection_index], scores[detection_index], self.confirm_after
            )

        for slot in still_unmatched_slots:
            remaining[slot].mark_missed(self.grace_frames, self.confirm_after)

        for detection_index in unmatched_strong:
            self.tracks.append(
                Tracklet(
                    self._next_id,
                    boxes[detection_index],
                    scores[detection_index],
                    labels[detection_index],
                    self.estimator,
                )
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.phase is not TrackPhase.Ended]
        active = [t for t in self.tracks if t.phase is TrackPhase.Active]
        active.sort(key=lambda track: track.score, reverse=True)
        return active

    def _match_tier(
        self,
        tracks: list[Tracklet],
        detection_indices: list[int],
        boxes: list[list[float]],
        labels: list[int],
        max_cost: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Associate one confidence tier.

        Returns ``(pairs, unmatched_track_slots, unmatched_detection_ids)``
        where pair entries are ``(slot in tracks, index into boxes)``.
        """
        if not tracks or not detection_indices:
            return [], list(range(len(tracks))), list(detection_indices)

        tier_boxes = [boxes[i] for i in detection_indices]
        tier_labels = [labels[i] for i in detection_indices]
        pairs, unmatched_tracks, unmatched_tier = associate(
            [track.box for track in tracks],
            tier_boxes,
            track_labels=[track.label for track in tracks] if self.class_aware else None,
            detection_labels=tier_labels if self.class_aware else None,
            max_cost=max_cost,
            optimal=self.optimal,
        )
        resolved = [(slot, detection_indices[tier_index]) for slot, tier_index in pairs]
        unmatched_detection_ids = [detection_indices[i] for i in unmatched_tier]
        return resolved, unmatched_tracks, unmatched_detection_ids

    def __repr__(self) -> str:
        active = sum(1 for t in self.tracks if t.phase is TrackPhase.Active)
        return (
            f"LofopTracker(frame={self.frame_index}, tracks={len(self.tracks)}, "
            f"active={active})"
        )


def track_detections(
    frames: Sequence[Any],
    **tracker_kwargs: Any,
) -> list[list[Tracklet]]:
    """Run a tracker over a sequence of per-frame detection results.

    Args:
        frames: Per frame, either a
            :class:`~lofop.deploy.postprocess.Detections` or a
            ``(boxes, scores, labels)`` tuple.
        **tracker_kwargs: Forwarded to :class:`LofopTracker`.

    Returns:
        The active tracks for each frame, in input order.
    """
    tracker = LofopTracker(**tracker_kwargs)
    results = []
    for frame in frames:
        if isinstance(frame, tuple):
            boxes, scores, labels = frame
        else:
            boxes, scores, labels = frame.boxes, frame.scores, frame.labels
        results.append(tracker.update(boxes, scores, labels))
    return results
