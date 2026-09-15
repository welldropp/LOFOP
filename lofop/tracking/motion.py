"""Constant-velocity motion estimation for tracked boxes.

A linear Kalman filter written directly from the standard recursive
estimator equations:

    predict   x' = F x                  P' = F P F^T + Q
    innovate  y  = z - H x'             S  = H P' H^T + R
    gain      K  = P' H^T S^-1
    correct   x  = x' + K y             P  = (I - K H) P'

LOFOP tracks boxes in **direct centre-size** coordinates,
``[cx, cy, w, h, vcx, vcy, vw, vh]``: centre, width, height, and the velocity
of each. Width and height are estimated independently rather than through an
aspect-ratio term, so a box that changes shape (a person turning, an object
becoming partly occluded) is modelled as two ordinary velocities instead of
one ratio whose derivative is ill-conditioned when the height is small.

Process and measurement noise scale with the box's **geometric mean side**,
``sqrt(w * h)``: uncertainty in pixels grows with apparent object size, and
the geometric mean responds to both dimensions rather than privileging
height. The three noise coefficients on :class:`MotionEstimator` are tuning
knobs with documented defaults, not constants of nature -- raise
``measurement_noise`` for a jittery detector, raise ``velocity_noise`` for
erratic motion.

numpy provides the linear algebra and is imported lazily, so the rest of
LOFOP keeps installing and running without it. Install with
``pip install "lofop[tracking]"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lofop.core.exceptions import LofopError

# State layout: index of each quantity in the 8-vector.
CENTRE_X, CENTRE_Y, WIDTH, HEIGHT = 0, 1, 2, 3
STATE_DIM = 8
MEASURE_DIM = 4

_numpy: Any = None


def _require_numpy() -> Any:
    """Import numpy on first use, with an actionable message when absent."""
    global _numpy
    if _numpy is None:
        try:
            import numpy
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise LofopError(
                "LOFOP tracking needs numpy for its motion model. "
                'Install the tracking extra:  pip install "lofop[tracking]"'
            ) from exc
        _numpy = numpy
    return _numpy


def to_centre_size(box: Sequence[float]) -> list[float]:
    """Convert an ``xyxy`` box to ``[cx, cy, w, h]``."""
    x1, y1, x2, y2 = (float(v) for v in box)
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1]


def to_corners(centre_size: Sequence[float]) -> list[float]:
    """Convert ``[cx, cy, w, h]`` back to an ``xyxy`` box."""
    cx, cy, w, h = (float(v) for v in centre_size[:4])
    half_w, half_h = w / 2.0, h / 2.0
    return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]


class MotionEstimator:
    """Kalman filter over ``[cx, cy, w, h]`` and their velocities.

    Args:
        position_noise: Process-noise coefficient for the four positional
            terms, as a fraction of the box's geometric mean side. The
            default admits roughly a twentieth of the box size of drift per
            frame before the filter starts distrusting its own prediction.
        velocity_noise: Process-noise coefficient for the four velocity
            terms. Smaller than ``position_noise`` because velocity is
            expected to change more slowly than position.
        measurement_noise: Observation-noise coefficient describing how much
            the detector's boxes jitter, again relative to box size.
        time_step: Interval between updates in frames. Rarely changed; a
            tracker running on every frame leaves it at 1.

    A filter instance holds no state: :meth:`initiate`, :meth:`predict` and
    :meth:`correct` take and return ``(mean, covariance)`` pairs, so one
    estimator can serve every tracklet and the state lives with the track.
    """

    def __init__(
        self,
        *,
        position_noise: float = 0.055,
        velocity_noise: float = 0.0095,
        measurement_noise: float = 0.045,
        time_step: float = 1.0,
    ) -> None:
        if min(position_noise, velocity_noise, measurement_noise) <= 0.0:
            raise LofopError(
                "Noise coefficients must be positive",
                context={
                    "position": position_noise,
                    "velocity": velocity_noise,
                    "measurement": measurement_noise,
                },
            )
        numpy = _require_numpy()
        self.position_noise = float(position_noise)
        self.velocity_noise = float(velocity_noise)
        self.measurement_noise = float(measurement_noise)
        self.time_step = float(time_step)

        # F: constant velocity -- each position term gains its velocity once
        # per step, each velocity term persists.
        self.transition = numpy.eye(STATE_DIM, dtype=numpy.float64)
        for axis in range(MEASURE_DIM):
            self.transition[axis, MEASURE_DIM + axis] = self.time_step
        # H: the detector observes position only, never velocity.
        self.observation = numpy.eye(MEASURE_DIM, STATE_DIM, dtype=numpy.float64)

    def _scale(self, mean_or_measurement: Any) -> float:
        """Geometric mean side of the box, floored to stay non-degenerate."""
        numpy = _require_numpy()
        width = abs(float(mean_or_measurement[WIDTH]))
        height = abs(float(mean_or_measurement[HEIGHT]))
        return float(max(numpy.sqrt(max(width, 1e-3) * max(height, 1e-3)), 1.0))

    def _process_noise(self, mean: Any) -> Any:
        numpy = _require_numpy()
        scale = self._scale(mean)
        deviations = [self.position_noise * scale] * MEASURE_DIM
        deviations += [self.velocity_noise * scale] * MEASURE_DIM
        return numpy.diag(numpy.square(numpy.asarray(deviations, dtype=numpy.float64)))

    def _measurement_noise(self, reference: Any) -> Any:
        numpy = _require_numpy()
        scale = self._scale(reference)
        deviations = numpy.full(MEASURE_DIM, self.measurement_noise * scale, dtype=numpy.float64)
        return numpy.diag(numpy.square(deviations))

    def initiate(self, measurement: Sequence[float]) -> tuple[Any, Any]:
        """Seed a new track from its first observation.

        Args:
            measurement: ``[cx, cy, w, h]`` of the detection that opened the
                track.

        Returns:
            ``(mean, covariance)``: the 8-vector state and its 8x8
            covariance. Velocity starts at zero and is given a deliberately
            wide prior, since a single observation carries no motion
            information at all.
        """
        numpy = _require_numpy()
        observed = numpy.asarray(measurement, dtype=numpy.float64).reshape(-1)
        if observed.size != MEASURE_DIM:
            raise LofopError(
                "Measurement must be [cx, cy, w, h]", context={"size": int(observed.size)}
            )
        mean = numpy.zeros(STATE_DIM, dtype=numpy.float64)
        mean[:MEASURE_DIM] = observed

        scale = self._scale(observed)
        # Position starts near the measurement, so its prior tracks the
        # measurement noise; velocity is unknown, so its prior is an order of
        # magnitude looser and the first few corrections dominate it.
        deviations = [self.measurement_noise * 2.0 * scale] * MEASURE_DIM
        deviations += [self.velocity_noise * 12.0 * scale] * MEASURE_DIM
        covariance = numpy.diag(numpy.square(numpy.asarray(deviations, dtype=numpy.float64)))
        return mean, covariance

    def predict(self, mean: Any, covariance: Any) -> tuple[Any, Any]:
        """Advance the state one time step under constant velocity."""
        numpy = _require_numpy()
        process_noise = self._process_noise(mean)
        predicted_mean = self.transition @ numpy.asarray(mean, dtype=numpy.float64)
        predicted_covariance = (
            self.transition @ numpy.asarray(covariance, dtype=numpy.float64) @ self.transition.T
            + process_noise
        )
        # Width and height are physical extents; a prediction must not drive
        # them through zero however long the track has gone unobserved.
        predicted_mean[WIDTH] = max(predicted_mean[WIDTH], 1e-3)
        predicted_mean[HEIGHT] = max(predicted_mean[HEIGHT], 1e-3)
        return predicted_mean, self._symmetrise(predicted_covariance)

    def project(self, mean: Any, covariance: Any) -> tuple[Any, Any]:
        """Map the state into measurement space, adding observation noise."""
        numpy = _require_numpy()
        mean = numpy.asarray(mean, dtype=numpy.float64)
        covariance = numpy.asarray(covariance, dtype=numpy.float64)
        projected_mean = self.observation @ mean
        projected_covariance = (
            self.observation @ covariance @ self.observation.T + self._measurement_noise(mean)
        )
        return projected_mean, self._symmetrise(projected_covariance)

    def correct(
        self, mean: Any, covariance: Any, measurement: Sequence[float]
    ) -> tuple[Any, Any]:
        """Fold a new observation into the state.

        Args:
            mean: Predicted 8-vector state.
            covariance: Predicted 8x8 covariance.
            measurement: Observed ``[cx, cy, w, h]``.

        Returns:
            The corrected ``(mean, covariance)``.
        """
        numpy = _require_numpy()
        observed = numpy.asarray(measurement, dtype=numpy.float64).reshape(-1)
        if observed.size != MEASURE_DIM:
            raise LofopError(
                "Measurement must be [cx, cy, w, h]", context={"size": int(observed.size)}
            )
        mean = numpy.asarray(mean, dtype=numpy.float64)
        covariance = numpy.asarray(covariance, dtype=numpy.float64)
        projected_mean, innovation_covariance = self.project(mean, covariance)

        # Solve K S = P H^T for the gain instead of inverting S: the solve is
        # better conditioned when a long-lived track makes S nearly singular.
        cross_covariance = covariance @ self.observation.T
        gain = numpy.linalg.solve(innovation_covariance.T, cross_covariance.T).T
        innovation = observed - projected_mean

        corrected_mean = mean + gain @ innovation
        identity = numpy.eye(STATE_DIM, dtype=numpy.float64)
        factor = identity - gain @ self.observation
        corrected_covariance = factor @ covariance
        corrected_mean[WIDTH] = max(corrected_mean[WIDTH], 1e-3)
        corrected_mean[HEIGHT] = max(corrected_mean[HEIGHT], 1e-3)
        return corrected_mean, self._symmetrise(corrected_covariance)

    def residual_distance(self, mean: Any, covariance: Any, measurements: Any) -> list[float]:
        """Squared Mahalanobis distance from the state to each measurement.

        Useful as a motion gate: a candidate whose distance exceeds a
        chi-square threshold for four degrees of freedom is implausible
        regardless of how well its box overlaps.
        """
        numpy = _require_numpy()
        candidates = numpy.asarray(measurements, dtype=numpy.float64)
        if candidates.ndim == 1:
            candidates = candidates.reshape(1, -1)
        if candidates.size == 0:
            return []
        projected_mean, innovation_covariance = self.project(mean, covariance)
        deltas = candidates - projected_mean
        solved = numpy.linalg.solve(innovation_covariance, deltas.T)
        return [float(value) for value in numpy.sum(deltas.T * solved, axis=0)]

    @staticmethod
    def _symmetrise(matrix: Any) -> Any:
        """Average a covariance with its transpose.

        Repeated predict/correct cycles accumulate floating-point asymmetry;
        left alone it eventually breaks the linear solves above.
        """
        return (matrix + matrix.T) / 2.0
