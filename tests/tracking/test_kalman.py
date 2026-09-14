"""Unit tests for the 8D linear Kalman filter."""

import numpy as np
import pytest

from lofop.tracking.kalman import KalmanFilter


def test_initiate():
    kf = KalmanFilter()
    # measurement: center_x=100, center_y=150, aspect_ratio=0.5, height=200
    measurement = np.array([100.0, 150.0, 0.5, 200.0])
    mean, cov = kf.initiate(measurement)

    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    np.testing.assert_allclose(mean[:4], measurement)
    np.testing.assert_allclose(mean[4:], np.zeros(4))
    # Covariance should be symmetric positive definite
    assert np.all(np.linalg.eigvals(cov) > 0)


def test_predict_constant_velocity():
    kf = KalmanFilter()
    # Object at x=50 moving at vx=5 per frame
    mean = np.array([50.0, 50.0, 1.0, 40.0, 5.0, 0.0, 0.0, 0.0])
    cov = np.eye(8)

    pred_mean, pred_cov = kf.predict(mean, cov)
    # x should have advanced to 55.0
    assert pytest.approx(pred_mean[0], 0.1) == 55.0
    assert pytest.approx(pred_mean[1], 0.1) == 50.0
    assert np.all(np.linalg.eigvals(pred_cov) > 0)


def test_update_reduces_uncertainty():
    kf = KalmanFilter()
    measurement = np.array([100.0, 100.0, 1.0, 50.0])
    mean, cov = kf.initiate(measurement)

    pred_mean, pred_cov = kf.predict(mean, cov)
    # New observation slightly shifted
    new_obs = np.array([102.0, 101.0, 1.0, 50.0])
    updated_mean, updated_cov = kf.update(pred_mean, pred_cov, new_obs)

    assert updated_mean.shape == (8,)
    # Trace of covariance should decrease after observation update
    assert np.trace(updated_cov) < np.trace(pred_cov)
    # Corrected position should be between prediction and measurement
    assert min(pred_mean[0], new_obs[0]) <= updated_mean[0] <= max(pred_mean[0], new_obs[0])
