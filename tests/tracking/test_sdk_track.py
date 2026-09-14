"""Integration tests for Detector.track() API."""

from PIL import Image

from lofop import Detector


def test_detector_track_sequence():
    det = Detector("n", num_classes=2, class_names=["circle", "triangle"])

    # Generate 3 dummy images representing consecutive frames
    frames = [
        Image.new("RGB", (128, 128), color=(i * 20, 100, 150))
        for i in range(3)
    ]

    results = list(det.track(frames, score_threshold=0.01, track_thresh=0.01))
    assert len(results) == 3
    for frame_det in results:
        assert frame_det.tracker_ids is not None
        assert len(frame_det.boxes) == len(frame_det.tracker_ids)
