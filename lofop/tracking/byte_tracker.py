"""ByteTrack multi-object tracking algorithm with Kalman filter motion model.

Implements two-stage association matching:
  1. High-confidence detections matched against confirmed active tracks.
  2. Low-confidence detections matched against remaining/lost tracks to recover
     temporarily occluded or motion-blurred objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from lofop.tracking.kalman import KalmanFilter

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    # Minimal greedy fallback if scipy is not present
    def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cost = cost_matrix.copy()
        rows, cols = [], []
        while cost.size > 0 and not np.isneginf(cost).all() and not np.isposinf(cost).all():
            idx = np.unravel_index(np.argmin(cost), cost.shape)
            r, c = idx
            if np.isinf(cost[r, c]):
                break
            rows.append(r)
            cols.append(c)
            cost[r, :] = np.inf
            cost[:, c] = np.inf
        return np.array(rows, dtype=int), np.array(cols, dtype=int)


class TrackState(IntEnum):
    """Lifecycle state of a tracked object."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class STrack:
    """Single-object tracklet maintaining an 8D Kalman filter state."""

    _count = 0

    def __init__(self, tlbr: Sequence[float], score: float, class_id: int) -> None:
        self.tlbr = np.asarray(tlbr, dtype=np.float32)
        self.score = float(score)
        self.class_id = int(class_id)

        self.kalman_filter = KalmanFilter()
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None

        self.is_activated = False
        self.track_id = 0
        self.state = TrackState.New

        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    @classmethod
    def next_id(cls) -> int:
        cls._count += 1
        return cls._count

    @classmethod
    def reset_id(cls) -> None:
        cls._count = 0

    @property
    def tlwh(self) -> np.ndarray:
        """Get bounding box in [top-left-x, top-left-y, width, height] format."""
        ret = self.tlbr.copy()
        ret[2] -= ret[0]
        ret[3] -= ret[1]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Convert current box to [center_x, center_y, aspect_ratio, height]."""
        ret = self.tlwh
        ret[0] += ret[2] / 2.0
        ret[1] += ret[3] / 2.0
        ret[2] /= max(ret[3], 1e-6)
        return ret

    def activate(self, frame_id: int) -> None:
        """Initialize the track state with its first detection."""
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.to_xyah())
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        """Re-activate a lost track with a newly matched detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self._update_tlbr_from_mean()
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        self.class_id = new_track.class_id
        if new_id:
            self.track_id = self.next_id()

    def predict(self) -> None:
        """Advance Kalman filter state forward by one frame."""
        if self.mean is None or self.covariance is None:
            return
        if self.state != TrackState.Tracked:
            self.mean[7] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self._update_tlbr_from_mean()

    def update(self, new_track: STrack, frame_id: int) -> None:
        """Update Kalman filter state with a matched detection in the current frame."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self._update_tlbr_from_mean()
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.class_id = new_track.class_id

    def mark_lost(self) -> None:
        """Mark track as lost (temporarily unobserved)."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark track as removed (permanently terminated)."""
        self.state = TrackState.Removed

    def _update_tlbr_from_mean(self) -> None:
        if self.mean is None:
            return
        w = self.mean[2] * self.mean[3]
        h = self.mean[3]
        x1 = self.mean[0] - w / 2.0
        y1 = self.mean[1] - h / 2.0
        self.tlbr = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)


def box_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of xyxy boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a_x1, a_y1, a_x2, a_y2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    b_x1, b_y1, b_x2, b_y2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
    area_b = (b_x2 - b_x1) * (b_y2 - b_y1)
    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return np.where(union_area > 0, inter_area / union_area, 0.0)


class ByteTracker:
    """Multi-object tracker utilizing ByteTrack two-stage data association."""

    def __init__(
        self,
        *,
        track_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        track_buffer: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)

        self.frame_id = 0
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

    def reset(self) -> None:
        """Reset internal state and ID counter."""
        self.frame_id = 0
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        STrack.reset_id()

    def update(
        self,
        boxes: Sequence[Sequence[float]] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        labels: Sequence[int] | np.ndarray,
    ) -> list[STrack]:
        """Update tracker with detections for the current frame.

        Args:
            boxes: (N, 4) bounding boxes in [x1, y1, x2, y2] format.
            scores: (N,) confidence scores.
            labels: (N,) class indices.

        Returns:
            List of active confirmed STrack instances in the current frame.
        """
        self.frame_id += 1
        activated_stracks: list[STrack] = []
        refind_stracks: list[STrack] = []
        lost_stracks: list[STrack] = []
        removed_stracks: list[STrack] = []

        boxes_arr = np.asarray(boxes, dtype=np.float32)
        scores_arr = np.asarray(scores, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int32)

        if len(boxes_arr) > 0:
            remain_inds = scores_arr >= self.track_thresh
            low_inds = (scores_arr >= self.low_thresh) & (scores_arr < self.track_thresh)

            detections_high = [
                STrack(b, s, label)
                for b, s, label in zip(
                    boxes_arr[remain_inds],
                    scores_arr[remain_inds],
                    labels_arr[remain_inds],
                )
            ]
            detections_low = [
                STrack(b, s, label)
                for b, s, label in zip(
                    boxes_arr[low_inds],
                    scores_arr[low_inds],
                    labels_arr[low_inds],
                )
            ]
        else:
            detections_high, detections_low = [], []

        unconfirmed: list[STrack] = []
        tracked_pool: list[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_pool.append(track)

        # Step 1: Predict new locations for confirmed and lost tracks
        strack_pool = tracked_pool + self.lost_stracks
        for track in strack_pool:
            track.predict()

        # Step 2: First association (high-confidence detections with active/lost tracks)
        dists = self._iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection = self._linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association (low-confidence detections with remaining active tracks)
        r_tracked_stracks = [
            strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked
        ]
        dists = self._iou_distance(r_tracked_stracks, detections_low)
        matches, u_remain, _ = self._linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_remain:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 4: Associate unconfirmed tracks with remaining high-confidence detections
        detections_rem = [detections_high[i] for i in u_detection]
        dists = self._iou_distance(unconfirmed, detections_rem)
        matches, u_unconfirmed, u_detection_final = self._linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_rem[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # Step 5: Initialize new tracks from unmatched high-confidence detections
        for inew in u_detection_final:
            track = detections_rem[inew]
            if track.score >= self.track_thresh:
                track.activate(self.frame_id)
                activated_stracks.append(track)

        # Step 6: Update lost and removed track pools
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks += [t for t in activated_stracks if t not in self.tracked_stracks]
        self.tracked_stracks += [t for t in refind_stracks if t not in self.tracked_stracks]

        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks += [t for t in lost_stracks if t not in self.lost_stracks]
        self.lost_stracks = [t for t in self.lost_stracks if t not in self.tracked_stracks]
        self.lost_stracks = [t for t in self.lost_stracks if t not in removed_stracks]

        self.removed_stracks += removed_stracks

        # Output currently tracked items
        return [t for t in self.tracked_stracks if t.is_activated]

    def _iou_distance(self, atracks: Sequence[STrack], btracks: Sequence[STrack]) -> np.ndarray:
        if not atracks or not btracks:
            return np.empty((len(atracks), len(btracks)), dtype=np.float32)
        boxes_a = np.array([t.tlbr for t in atracks], dtype=np.float32)
        boxes_b = np.array([t.tlbr for t in btracks], dtype=np.float32)
        ious = box_iou_batch(boxes_a, boxes_b)
        return 1.0 - ious

    def _linear_assignment(
        self, cost_matrix: np.ndarray, thresh: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        unmatched_a = list(range(cost_matrix.shape[0]))
        unmatched_b = list(range(cost_matrix.shape[1]))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append((r, c))
                if r in unmatched_a:
                    unmatched_a.remove(r)
                if c in unmatched_b:
                    unmatched_b.remove(c)

        return matches, unmatched_a, unmatched_b
