"""Unit tests for ByteTracker multi-object tracking."""

import numpy as np

from lofop.tracking.byte_tracker import ByteTracker, TrackState


def test_single_object_tracking():
    tracker = ByteTracker(track_thresh=0.5, track_buffer=30)
    tracker.reset()

    # Frame 1: object at [10, 10, 50, 50]
    tracks_f1 = tracker.update([[10, 10, 50, 50]], [0.9], [0])
    assert len(tracks_f1) == 1
    t1_id = tracks_f1[0].track_id
    assert tracks_f1[0].state == TrackState.Tracked

    # Frame 2: object moves slightly to [12, 12, 52, 52]
    tracks_f2 = tracker.update([[12, 12, 52, 52]], [0.85], [0])
    assert len(tracks_f2) == 1
    # Track ID must persist
    assert tracks_f2[0].track_id == t1_id
    np.testing.assert_allclose(tracks_f2[0].tlbr, [12, 12, 52, 52], atol=2.0)


def test_low_confidence_recovery():
    """Test ByteTrack recovering a track via second association stage."""
    tracker = ByteTracker(track_thresh=0.6, low_thresh=0.2, track_buffer=30)
    tracker.reset()

    # Frame 1: high confidence detection
    tracker.update([[20, 20, 60, 60]], [0.9], [0])

    # Frame 2: low confidence detection (e.g. motion blur, score=0.35)
    tracks_f2 = tracker.update([[22, 22, 62, 62]], [0.35], [0])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == 1


def test_occlusion_and_track_removal():
    """Test track coasting and eventual removal when unobserved."""
    tracker = ByteTracker(track_thresh=0.5, track_buffer=2, frame_rate=30)
    tracker.reset()

    # Frame 1: object appears
    tracker.update([[10, 10, 50, 50]], [0.9], [0])

    # Frame 2: occluded (no detections)
    tracks_f2 = tracker.update([], [], [])
    # Track is not in active confirmed output of empty frame
    assert len(tracks_f2) == 0
    assert len(tracker.lost_stracks) == 1

    # Frame 3: still occluded
    tracker.update([], [], [])

    # Frame 4: exceeds max_time_lost (track_buffer=2), should be removed
    tracker.update([], [], [])
    assert len(tracker.lost_stracks) == 0
    assert len(tracker.removed_stracks) == 1
