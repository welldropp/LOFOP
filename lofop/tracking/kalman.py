"""Kalman filter for bounding box tracking in image space.

Implements an 8-dimensional state space:
    [x, y, a, h, vx, vy, va, vh]
where (x, y) is the bounding box center, a is aspect ratio (width / height),
h is height, and (vx, vy, va, vh) are their respective velocities.

Motion follows a constant velocity model with discrete Gaussian process noise.
Measurements are direct 4D observations of [x, y, a, h].
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """A linear Kalman filter for bounding box tracking in image space."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # State transition matrix F (constant velocity motion model)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement matrix H (observes [x, y, a, h])
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty weights
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create initial state mean and covariance from a measurement.

        Args:
            measurement: Bounding box coordinates [x, y, a, h].

        Returns:
            mean: (8,) initial state vector.
            covariance: (8, 8) initial covariance matrix.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project the state distribution forward one time step.

        Args:
            mean: Current (8,) state mean vector.
            covariance: Current (8, 8) state covariance matrix.

        Returns:
            mean: Projected (8,) state mean vector.
            covariance: Projected (8, 8) state covariance matrix.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        transformed_cov = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)
        )
        covariance = transformed_cov + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space.

        Args:
            mean: Current (8,) state mean vector.
            covariance: Current (8, 8) state covariance matrix.

        Returns:
            projected_mean: (4,) measurement mean vector.
            projected_cov: (4, 4) measurement covariance matrix.
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        transformed_cov = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        covariance = transformed_cov + innovation_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Update state distribution with an incoming measurement.

        Args:
            mean: Predicted (8,) state mean.
            covariance: Predicted (8, 8) state covariance.
            measurement: Incoming (4,) measurement [x, y, a, h].

        Returns:
            new_mean: Corrected (8,) state mean vector.
            new_covariance: Corrected (8, 8) state covariance matrix.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol_factor, np.dot(covariance, self._update_mat.T).T
        ).T
        kalman_gain = np.linalg.solve(chol_factor.T, kalman_gain.T).T

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        reduction = np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        new_covariance = covariance - reduction
        return new_mean, new_covariance
