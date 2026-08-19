"""Tests for the torch-free ONNX runtime (``lofop.runtime``).

The expected detections here are the same ones ``cpp/tests/test_lofop.cpp``
asserts, driven by the same stub outputs. Pinning both suites to identical
numbers is what makes the dual-language claim testable: if the Python runtime
and the C++ SDK ever disagree on a shared model, one of the two suites breaks.
"""

from __future__ import annotations

import dataclasses

import pytest

from lofop.core.exceptions import DataError, LofopError
from lofop.runtime import Detection, Detector, Image

onnx = pytest.importorskip("onnx", reason="onnx is needed to build the stub model")
pytest.importorskip("onnxruntime", reason="onnxruntime is needed to run the stub model")
np = pytest.importorskip("numpy", reason="numpy is needed by the runtime")

SIZE = 64
NUM_CANDIDATES = 4
NUM_CLASSES = 3

# Two overlapping class-0 boxes (the weaker must be suppressed), one class-1
# box, and one candidate below the default 0.25 threshold.
STUB_BOXES = [[10, 10, 30, 30], [11, 11, 31, 31], [40, 40, 60, 60], [0, 0, 5, 5]]
STUB_SCORES = [(0, 0.9), (0, 0.8), (1, 0.7), (2, 0.1)]


@pytest.fixture(scope="module")
def stub_model(tmp_path_factory) -> str:
    """An ONNX graph with LOFOP's dense (boxes, scores) output contract."""
    from onnx import TensorProto, helper

    boxes = np.array([STUB_BOXES], dtype=np.float32)
    scores = np.zeros((1, NUM_CANDIDATES, NUM_CLASSES), dtype=np.float32)
    for row, (label, score) in enumerate(STUB_SCORES):
        scores[0, row, label] = score

    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant", [], ["boxes"],
                value=helper.make_tensor(
                    "b", TensorProto.FLOAT, [1, NUM_CANDIDATES, 4], boxes.flatten()
                ),
            ),
            helper.make_node(
                "Constant", [], ["scores"],
                value=helper.make_tensor(
                    "s", TensorProto.FLOAT, [1, NUM_CANDIDATES, NUM_CLASSES], scores.flatten()
                ),
            ),
        ],
        "lofop_stub",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, NUM_CANDIDATES, 4]),
            helper.make_tensor_value_info(
                "scores", TensorProto.FLOAT, [1, NUM_CANDIDATES, NUM_CLASSES]
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path = tmp_path_factory.mktemp("runtime") / "stub.onnx"
    onnx.save(model, str(path))
    return str(path)


@pytest.fixture
def frame() -> Image:
    """A 128x96 (4:3) image: letterboxed to 64 it scales by 0.5 and pads 8px."""
    return Image.from_pixels(bytes(128 * 96 * 3), 128, 96, 3)


class TestImage:
    def test_from_pixels_reports_its_shape(self, frame):
        assert frame.size == (128, 96)
        assert frame.channels == 3

    def test_rejects_a_mismatched_buffer(self):
        with pytest.raises(DataError):
            Image.from_pixels(bytes(10), 8, 8, 3)

    def test_rejects_unsupported_channels(self):
        with pytest.raises(DataError):
            Image.from_pixels(bytes(4 * 4 * 4), 4, 4, 4)

    def test_loads_a_file(self, tmp_path):
        pil = pytest.importorskip("PIL.Image")
        path = tmp_path / "frame.png"
        pil.new("RGB", (32, 24), (10, 20, 30)).save(path)
        image = Image.load(path)
        assert image.size == (32, 24)
        assert image.channels == 3

    def test_reports_a_missing_file_clearly(self, tmp_path):
        with pytest.raises(DataError):
            Image.load(tmp_path / "nope.png")


class TestDetector:
    def test_reads_input_size_from_the_graph(self, stub_model):
        assert Detector(stub_model).image_size == SIZE

    def test_rejects_a_missing_model(self, tmp_path):
        with pytest.raises(LofopError):
            Detector(tmp_path / "missing.onnx")

    def test_suppresses_duplicates_and_weak_candidates(self, stub_model, frame):
        results = Detector(stub_model).predict(frame)
        # 4 candidates in: one duplicate suppressed by NMS, one below threshold.
        assert len(results) == 2

    def test_maps_boxes_back_to_original_coordinates(self, stub_model, frame):
        results = Detector(stub_model).predict(frame)
        # scale 0.5, pad_top 8: x_src = x / 0.5, y_src = (y - 8) / 0.5.
        assert results[0].box == pytest.approx((20.0, 4.0, 60.0, 44.0), abs=1e-3)
        # The second box runs past the bottom edge and must be clamped.
        assert results[1].box == pytest.approx((80.0, 64.0, 120.0, 96.0), abs=1e-3)

    def test_orders_detections_by_confidence(self, stub_model, frame):
        results = Detector(stub_model).predict(frame)
        assert results[0].score == pytest.approx(0.9, abs=1e-5)
        assert results[1].score == pytest.approx(0.7, abs=1e-5)
        assert results[0].score >= results[1].score

    def test_confidence_aliases_score(self, stub_model, frame):
        hit = Detector(stub_model).predict(frame)[0]
        assert hit.confidence == hit.score

    def test_applies_class_names(self, stub_model, frame):
        detector = Detector(stub_model, class_names=["cat", "dog", "bird"])
        assert [hit.name for hit in detector.predict(frame)] == ["cat", "dog"]

    def test_falls_back_to_indexed_names(self, stub_model, frame):
        assert Detector(stub_model).predict(frame)[0].name == "class_0"

    def test_threshold_override_applies_per_call(self, stub_model, frame):
        detector = Detector(stub_model)
        assert len(detector.predict(frame, score_threshold=0.75)) == 1
        # The override must not persist into the next call.
        assert len(detector.predict(frame)) == 2

    def test_constructor_threshold_is_respected(self, stub_model, frame):
        assert len(Detector(stub_model, score_threshold=0.95).predict(frame)) == 0

    def test_accepts_an_image_path(self, stub_model, tmp_path):
        pil = pytest.importorskip("PIL.Image")
        path = tmp_path / "frame.png"
        pil.new("RGB", (128, 96), (40, 50, 60)).save(path)
        assert len(Detector(stub_model).predict(path)) == 2

    def test_predict_batch_returns_one_list_per_image(self, stub_model, frame):
        results = Detector(stub_model).predict_batch([frame, frame])
        assert len(results) == 2
        assert all(len(item) == 2 for item in results)

    def test_soft_nms_retains_decayed_duplicates(self, stub_model, frame):
        results = Detector(stub_model, soft_nms=True).predict(frame)
        assert len(results) >= 2

    def test_reports_its_backends(self, stub_model):
        detector = Detector(stub_model)
        assert any("CPU" in provider for provider in detector.providers)
        assert detector.ops_backend in {"cuda", "native", "python"}

    def test_repr_is_informative(self, stub_model):
        assert "stub.onnx" in repr(Detector(stub_model))


class TestDetection:
    def test_is_immutable(self):
        hit = Detection(box=(0.0, 0.0, 1.0, 1.0), score=0.5, label=0, name="cat")
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.score = 0.9

    def test_repr_shows_name_and_score(self):
        hit = Detection(box=(0.0, 0.0, 1.0, 1.0), score=0.5, label=0, name="cat")
        assert "cat" in repr(hit) and "0.500" in repr(hit)


class TestIsolationFromTheTrainingApi:
    """The runtime must neither depend on, nor disturb, the PyTorch SDK."""

    def test_importing_the_runtime_does_not_import_torch(self):
        """Deployment images omit PyTorch entirely, so this must hold.

        Checked in a clean interpreter: asserting on ``sys.modules`` in-process
        would pass merely because no earlier test imported torch.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys; import lofop.runtime;"
                " assert 'torch' not in sys.modules, sorted(sys.modules);"
                " print('clean')",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "clean" in result.stdout

    def test_the_two_detectors_are_distinct_apis(self):
        """``lofop.Detector`` (training) and ``lofop.runtime.Detector`` (serving).

        Neither shadows the other: they live in different modules and are
        reached by different imports, so adding the runtime cannot change what
        existing ``from lofop import Detector`` code resolves to.
        """
        import importlib.util

        import lofop

        assert Detector.__module__ == "lofop.runtime.detector"
        # The training SDK is still the name exported from the package root...
        assert "Detector" in lofop.__all__
        # ...and still lives in its own module, which find_spec locates without
        # executing it (importing lofop.sdk would require torch).
        assert importlib.util.find_spec("lofop.sdk") is not None
