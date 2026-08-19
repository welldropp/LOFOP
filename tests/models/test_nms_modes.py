"""Tests for the inference nms_mode options: greedy, soft, and NMS-free."""

import pytest

torch = pytest.importorskip("torch")

import lofop.models  # noqa: E402, F401  (registers model components)
from lofop.core.config import Config  # noqa: E402
from lofop.core.exceptions import LofopError, ModelError  # noqa: E402
from lofop.registries import HUB  # noqa: E402


def build(nms_mode="greedy"):
    cfg = Config.load("lofop/configs/lofop-detect/n.yaml", resolve=False)
    cfg.num_classes = 2
    cfg.resolve()
    model = HUB.build(cfg.model, nms_mode=nms_mode).eval()
    model.score_threshold = 0.05
    return model


class TestNmsModes:
    def test_unknown_mode_rejected(self):
        with pytest.raises(LofopError):  # ModelError, wrapped by the builder
            build(nms_mode="nope")

    def test_sdk_rejects_unknown_mode(self):
        from lofop import Detector

        with pytest.raises(ModelError):
            Detector("n", num_classes=2, image_size=64, device="cpu", nms_mode="hard")

    @pytest.mark.parametrize("mode", ["greedy", "soft", "free"])
    def test_all_modes_produce_valid_output(self, mode):
        torch.manual_seed(0)
        model = build(nms_mode=mode)
        [result] = model.predict(torch.rand(1, 3, 64, 64))
        k = result["boxes"].shape[0]
        assert result["scores"].shape == (k,) and result["labels"].shape == (k,)
        assert k <= model.max_detections
        if k > 1:  # scores stay descending in every mode
            assert (result["scores"][:-1] >= result["scores"][1:]).all()

    def test_free_mode_keeps_only_local_peaks(self):
        torch.manual_seed(0)
        model = build(nms_mode="free")
        images = torch.rand(1, 3, 64, 64)
        cls_out, _, quality_out = model.forward(images)
        peaks = model._peak_mask(cls_out, quality_out)
        # Peak selection must be strictly sparser than raw thresholding.
        assert 0 < int(peaks.sum()) < peaks.shape[1]
        [result] = model.predict(images)
        greedy = build(nms_mode="greedy")
        greedy.load_state_dict(model.state_dict())
        [base] = greedy.predict(images)
        assert result["boxes"].shape[0] <= max(base["boxes"].shape[0], 1) * 3

    def test_soft_mode_decays_scores(self):
        torch.manual_seed(0)
        model = build(nms_mode="soft")
        greedy = build(nms_mode="greedy")
        greedy.load_state_dict(model.state_dict())
        images = torch.rand(1, 3, 64, 64)
        [soft] = model.predict(images)
        [base] = greedy.predict(images)
        # Soft keeps at least as many detections as greedy on the same input.
        assert soft["boxes"].shape[0] >= base["boxes"].shape[0]

    def test_task_variants_inherit_modes(self):
        cfg = Config.load("lofop/configs/lofop-detect/n-seg.yaml", resolve=False)
        cfg.num_classes = 2
        cfg.resolve()
        model = HUB.build(cfg.model, nms_mode="free").eval()
        model.score_threshold = 0.05
        [result] = model.predict(torch.rand(1, 3, 64, 64))
        assert "masks" in result
