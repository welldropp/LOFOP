"""Tests for the tracking subsystem: motion, association, and the tracker."""

import pytest

from lofop.core.exceptions import LofopError
from lofop.ops.assignment import greedy_assignment, optimal_assignment
from lofop.tracking.association import FORBIDDEN_COST, associate, iou_cost_matrix, match

numpy = pytest.importorskip("numpy")

from lofop.tracking import LofopTracker, MotionEstimator, TrackPhase  # noqa: E402
from lofop.tracking.motion import to_centre_size, to_corners  # noqa: E402


class TestAssignment:
    def test_beats_greedy_when_they_differ(self):
        cost = [[4, 1, 3], [2, 0, 5], [3, 2, 2]]
        best = optimal_assignment(cost)
        assert sum(cost[i][best[i]] for i in range(3)) == 5
        naive = greedy_assignment(cost)
        assert sum(cost[i][naive[i]] for i in range(3)) == 6

    @pytest.mark.parametrize("rows,columns", [(1, 1), (2, 4), (4, 2), (5, 5)])
    def test_one_to_one_and_complete(self, rows, columns):
        rng = numpy.random.RandomState(rows * 10 + columns)
        cost = rng.rand(rows, columns).tolist()
        assignment = optimal_assignment(cost)
        used = [c for c in assignment if c >= 0]
        assert len(assignment) == rows
        assert len(used) == min(rows, columns)
        assert len(set(used)) == len(used)

    def test_empty_and_ragged(self):
        assert optimal_assignment([]) == []
        with pytest.raises(LofopError):
            optimal_assignment([[1.0, 2.0], [3.0]])

    def test_greedy_respects_max_cost(self):
        assert greedy_assignment([[0.1, 9.0], [9.0, 9.0]], max_cost=1.0) == [0, -1]


class TestAssociation:
    def test_cost_is_one_minus_iou(self):
        [[cost]] = iou_cost_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10]])
        assert cost == pytest.approx(0.0)

    def test_matches_across_shuffled_order(self):
        tracks = [[0, 0, 10, 10], [50, 50, 60, 60]]
        detections = [[51, 51, 61, 61], [1, 1, 11, 11]]
        pairs, unmatched_tracks, unmatched_detections = associate(tracks, detections)
        assert sorted(pairs) == [(0, 1), (1, 0)]
        assert unmatched_tracks == [] and unmatched_detections == []

    def test_class_mismatch_is_forbidden(self):
        costs = iou_cost_matrix(
            [[0, 0, 10, 10]], [[0, 0, 10, 10]],
            track_labels=[0], detection_labels=[1],
        )
        assert costs[0][0] == FORBIDDEN_COST
        pairs, _, _ = associate(
            [[0, 0, 10, 10]], [[0, 0, 10, 10]],
            track_labels=[0], detection_labels=[1],
        )
        assert pairs == []

    def test_far_apart_boxes_are_not_matched(self):
        pairs, tracks, detections = associate([[0, 0, 10, 10]], [[500, 500, 510, 510]])
        assert pairs == [] and tracks == [0] and detections == [0]

    def test_empty_sides(self):
        assert associate([], [[0, 0, 1, 1]]) == ([], [], [0])
        assert associate([[0, 0, 1, 1]], []) == ([], [0], [])

    def test_match_honours_ceiling(self):
        # 1 - IoU of these is well above 0.1, so the ceiling rejects the pair.
        cost = iou_cost_matrix([[0, 0, 10, 10]], [[7, 7, 17, 17]])
        assert match(cost, max_cost=0.1)[0] == []


class TestMotion:
    def test_coordinate_roundtrip(self):
        box = [10.0, 20.0, 50.0, 80.0]
        assert to_corners(to_centre_size(box)) == pytest.approx(box)

    def test_constant_velocity_prediction(self):
        estimator = MotionEstimator()
        mean, covariance = estimator.initiate([10.0, 10.0, 20.0, 20.0])
        for step in range(6):
            mean, covariance = estimator.predict(mean, covariance)
            mean, covariance = estimator.correct(
                mean, covariance, [10.0 + 5.0 * (step + 1), 10.0, 20.0, 20.0]
            )
        # After a steady rightward run the filter should have learned the
        # velocity and its next prediction should lead the last observation.
        predicted, _ = estimator.predict(mean, covariance)
        assert predicted[0] > mean[0]
        assert float(mean[1]) == pytest.approx(10.0, abs=1.0)

    def test_uncertainty_grows_without_observations(self):
        estimator = MotionEstimator()
        mean, covariance = estimator.initiate([10.0, 10.0, 20.0, 20.0])
        before = float(numpy.trace(covariance))
        for _ in range(5):
            mean, covariance = estimator.predict(mean, covariance)
        assert float(numpy.trace(covariance)) > before

    def test_width_height_stay_positive(self):
        estimator = MotionEstimator()
        mean, covariance = estimator.initiate([10.0, 10.0, 20.0, 20.0])
        mean, covariance = estimator.correct(mean, covariance, [10.0, 10.0, 2.0, 2.0])
        for _ in range(40):
            mean, covariance = estimator.predict(mean, covariance)
        assert mean[2] > 0.0 and mean[3] > 0.0

    def test_covariance_stays_symmetric(self):
        estimator = MotionEstimator()
        mean, covariance = estimator.initiate([5.0, 5.0, 10.0, 10.0])
        for _ in range(10):
            mean, covariance = estimator.predict(mean, covariance)
            mean, covariance = estimator.correct(mean, covariance, [5.0, 5.0, 10.0, 10.0])
        assert numpy.allclose(covariance, covariance.T)

    def test_residual_distance_ranks_candidates(self):
        estimator = MotionEstimator()
        mean, covariance = estimator.initiate([10.0, 10.0, 20.0, 20.0])
        near, far = estimator.residual_distance(
            mean, covariance, [[10.0, 10.0, 20.0, 20.0], [900.0, 900.0, 20.0, 20.0]]
        )
        assert near < far

    def test_rejects_bad_inputs(self):
        with pytest.raises(LofopError):
            MotionEstimator(position_noise=0.0)
        estimator = MotionEstimator()
        with pytest.raises(LofopError):
            estimator.initiate([1.0, 2.0])


class TestTracker:
    def test_identity_is_stable_for_moving_object(self):
        tracker = LofopTracker(confirm_after=2, grace_frames=5)
        seen = set()
        for frame in range(12):
            x = 10.0 + 5.0 * frame
            tracks = tracker.update([[x, 20.0, x + 30.0, 60.0]], [0.9], [0])
            seen.update(track.track_id for track in tracks)
        assert len(seen) == 1

    def test_weak_detections_sustain_but_never_create(self):
        tracker = LofopTracker(confirm_after=2, strong_threshold=0.5, weak_threshold=0.1)
        # Weak-only input never opens a track.
        for _ in range(5):
            assert tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.2], [0]) == []
        assert tracker.tracks == []

        # With a confident start, weak detections keep it alive.
        for _ in range(3):
            tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.9], [0])
        established = tracker.tracks[0].track_id
        for _ in range(4):
            tracks = tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.2], [0])
        assert tracks and tracks[0].track_id == established

    def test_dormant_then_recovered_keeps_id(self):
        tracker = LofopTracker(confirm_after=1, grace_frames=4)
        tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.9], [0])
        original = tracker.tracks[0].track_id
        for _ in range(3):
            tracker.update([], [], [])
        assert tracker.tracks[0].phase is TrackPhase.Dormant
        recovered = tracker.update([[6.0, 0.0, 26.0, 20.0]], [0.9], [0])
        assert recovered and recovered[0].track_id == original

    def test_track_ends_after_grace_period(self):
        tracker = LofopTracker(confirm_after=1, grace_frames=2)
        tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.9], [0])
        for _ in range(6):
            tracker.update([], [], [])
        assert tracker.tracks == []

    def test_unconfirmed_flicker_is_discarded(self):
        tracker = LofopTracker(confirm_after=3, grace_frames=10)
        tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.9], [0])
        tracker.update([], [], [])
        assert tracker.tracks == []

    def test_two_objects_keep_separate_ids(self):
        tracker = LofopTracker(confirm_after=1)
        for frame in range(6):
            offset = float(frame)
            tracks = tracker.update(
                [[offset, 0.0, offset + 20.0, 20.0], [200.0 - offset, 0.0, 220.0 - offset, 20.0]],
                [0.9, 0.9], [0, 0],
            )
        assert len({track.track_id for track in tracks}) == 2

    def test_class_aware_matching_does_not_swap_labels(self):
        tracker = LofopTracker(confirm_after=1, class_aware=True)
        tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.9], [0])
        first = tracker.tracks[0].track_id
        # Same place, different class: must open a new identity.
        tracks = tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.9], [1])
        assert all(track.track_id != first or track.label == 0 for track in tracks)

    def test_reset_clears_state(self):
        tracker = LofopTracker(confirm_after=1)
        tracker.update([[0.0, 0.0, 20.0, 20.0]], [0.9], [0])
        tracker.reset()
        assert tracker.tracks == [] and tracker.frame_index == 0

    def test_input_validation(self):
        tracker = LofopTracker()
        with pytest.raises(LofopError):
            tracker.update([[0.0, 0.0, 1.0, 1.0]], [])
        with pytest.raises(LofopError):
            LofopTracker(strong_threshold=0.2, weak_threshold=0.8)

    def test_track_detections_helper(self):
        from lofop.tracking import track_detections

        frames = [([[0.0, 0.0, 20.0, 20.0]], [0.9], [0]) for _ in range(4)]
        per_frame = track_detections(frames, confirm_after=1)
        assert len(per_frame) == 4
        assert per_frame[-1][0].track_id == per_frame[1][0].track_id
