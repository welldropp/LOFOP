"""Unit tests for third-party adapters (Roboflow supervision)."""

from unittest.mock import MagicMock, patch

import pytest

from lofop.deploy.postprocess import Detections
from lofop.tracking.adapters import from_supervision, to_supervision


def test_to_supervision_missing_package_raises_import_error():
    det = Detections(
        boxes=[[10.0, 20.0, 50.0, 60.0]],
        scores=[0.95],
        labels=[0],
        tracker_ids=[1],
    )
    with patch.dict("sys.modules", {"supervision": None}):
        with pytest.raises(ImportError) as exc_info:
            to_supervision(det)
        assert "pip install supervision" in str(exc_info.value)


def test_to_supervision_with_mocked_library():
    det = Detections(
        boxes=[[10.0, 20.0, 50.0, 60.0]],
        scores=[0.95],
        labels=[0],
        tracker_ids=[1],
    )

    mock_sv = MagicMock()
    with patch.dict("sys.modules", {"supervision": mock_sv}):
        det.to_supervision(class_names=["person"])
        mock_sv.Detections.assert_called_once()
        call_kwargs = mock_sv.Detections.call_args[1]
        assert call_kwargs["xyxy"].shape == (1, 4)
        assert call_kwargs["confidence"].shape == (1,)
        assert call_kwargs["tracker_id"].shape == (1,)
        assert "class_name" in call_kwargs["data"]


def test_from_supervision():
    import numpy as np

    mock_sv_det = MagicMock()
    mock_sv_det.xyxy = np.array([[5.0, 5.0, 25.0, 25.0]])
    mock_sv_det.confidence = np.array([0.88])
    mock_sv_det.class_id = np.array([2])
    mock_sv_det.tracker_id = np.array([42])
    mock_sv_det.mask = None

    lofop_det = from_supervision(mock_sv_det)
    assert lofop_det.boxes == [[5.0, 5.0, 25.0, 25.0]]
    assert lofop_det.scores == [0.88]
    assert lofop_det.labels == [2]
    assert lofop_det.tracker_ids == [42]
