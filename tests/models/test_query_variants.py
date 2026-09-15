"""Tests for the NMS-free query family and the SwiftNet backbone."""

import pytest

torch = pytest.importorskip("torch")

import lofop.models  # noqa: E402, F401  (registers components)
from lofop.core.config import Config  # noqa: E402
from lofop.core.exceptions import LofopError, ModelError  # noqa: E402
from lofop.models import LofopQuery, PivotMatcher, SlateHead, SwiftNet  # noqa: E402
from lofop.registries import HUB  # noqa: E402

CONFIG_DIR = "lofop/configs/lofop-detect"


def build(name, num_classes=3, **overrides):
    cfg = Config.load(f"{CONFIG_DIR}/{name}.yaml", resolve=False)
    cfg.num_classes = num_classes
    for key, value in overrides.items():
        cfg[key] = value
    cfg.resolve()
    return HUB.build(cfg.model)


def sample_targets():
    return [
        {
            "boxes": torch.tensor([[8.0, 8.0, 60.0, 60.0], [70.0, 70.0, 110.0, 120.0]]),
            "labels": torch.tensor([0, 2]),
        },
        {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)},
    ]


class TestSwiftNet:
    def test_emits_three_levels_at_expected_strides(self):
        backbone = SwiftNet()
        features = backbone(torch.rand(1, 3, 128, 128))
        assert len(features) == 3
        assert [f.shape[-1] for f in features] == [16, 8, 4]
        assert tuple(f.shape[1] for f in features) == backbone.out_channels

    def test_residual_only_when_shapes_allow(self):
        from lofop.models import InvertedResidual

        assert InvertedResidual(32, 32, stride=1).use_residual
        assert not InvertedResidual(32, 64, stride=1).use_residual
        assert not InvertedResidual(32, 32, stride=2).use_residual

    def test_rejects_malformed_configuration(self):
        with pytest.raises(ModelError):
            SwiftNet(widths=(8, 16), depths=(1, 1))
        with pytest.raises(ModelError):
            SwiftNet(depths=(0, 1, 1, 1))

    def test_drops_into_the_dense_family(self):
        model = build("mb-n")
        losses = model.compute_losses(torch.rand(2, 3, 128, 128), sample_targets())
        assert "quality" in losses
        losses["total"].backward()


class TestSlateHead:
    def test_output_shapes_cover_every_layer(self):
        head = SlateHead(num_classes=4, width=32, num_queries=12, num_layers=2, num_heads=4)
        features = [torch.rand(2, 32, s, s) for s in (16, 8, 4)]
        class_logits, boxes = head(features)
        assert class_logits.shape == (2, 2, 12, 4)
        assert boxes.shape == (2, 2, 12, 4)

    def test_boxes_are_normalised(self):
        head = SlateHead(num_classes=2, width=32, num_queries=8, num_heads=4)
        _, boxes = head([torch.rand(1, 32, s, s) for s in (8, 4, 2)])
        assert bool((boxes >= 0).all()) and bool((boxes <= 1).all())

    def test_coordinate_roundtrip(self):
        boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.5]]])
        corners = SlateHead.to_corners(boxes, 100.0)
        assert corners.squeeze().tolist() == pytest.approx([37.5, 25.0, 62.5, 75.0])
        assert SlateHead.to_normalised(corners, 100.0).squeeze().tolist() == pytest.approx(
            boxes.squeeze().tolist()
        )

    def test_rejects_bad_configuration(self):
        with pytest.raises(ModelError):
            SlateHead(num_classes=0)
        with pytest.raises(ModelError):
            SlateHead(num_classes=2, width=30, num_heads=4)
        with pytest.raises(ModelError):
            SlateHead(num_classes=2, num_queries=0)

    def test_rejects_wrong_level_count(self):
        head = SlateHead(num_classes=2, width=32, num_heads=4)
        with pytest.raises(ModelError):
            head([torch.rand(1, 32, 8, 8)])

    def test_reference_prior_can_be_seeded(self):
        head = SlateHead(num_classes=2, width=32, num_queries=4, num_heads=4)
        prior = torch.full((4, 4), 0.5)
        head.initialise_reference_from(prior)
        assert torch.allclose(head.reference_points.weight.sigmoid(), prior, atol=1e-4)
        with pytest.raises(ModelError):
            head.initialise_reference_from(torch.zeros(3, 4))


class TestPivotMatcher:
    def test_assignment_is_one_to_one(self):
        matcher = PivotMatcher()
        logits = torch.randn(10, 3)
        boxes = torch.rand(10, 4) * 100
        targets = torch.tensor([[0.0, 0.0, 50.0, 50.0], [50.0, 50.0, 90.0, 90.0]])
        slots, objects = matcher.match(logits, boxes, targets, torch.tensor([0, 1]), 128.0)
        assert slots.numel() == 2
        assert len(set(slots.tolist())) == 2
        assert sorted(objects.tolist()) == [0, 1]

    def test_prefers_the_closest_slot(self):
        matcher = PivotMatcher(class_weight=0.0, giou_weight=0.0)
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [80.0, 80.0, 100.0, 100.0]])
        target = torch.tensor([[79.0, 79.0, 99.0, 99.0]])
        slots, _ = matcher.match(
            torch.zeros(2, 2), boxes, target, torch.tensor([0]), 128.0
        )
        assert slots.tolist() == [1]

    def test_no_targets_returns_empty(self):
        matcher = PivotMatcher()
        slots, objects = matcher.match(
            torch.randn(5, 2), torch.rand(5, 4), torch.zeros(0, 4),
            torch.zeros(0, dtype=torch.long), 128.0,
        )
        assert slots.numel() == 0 and objects.numel() == 0

    def test_requires_a_nonzero_weight(self):
        with pytest.raises(ModelError):
            PivotMatcher(class_weight=0.0, box_weight=0.0, giou_weight=0.0)


class TestLofopQuery:
    @pytest.mark.parametrize("name", ["q-n", "q-s", "mbq-n"])
    def test_variants_build_train_and_predict(self, name):
        model = build(name)
        assert isinstance(model, LofopQuery)
        losses = model.compute_losses(torch.rand(2, 3, 128, 128), sample_targets())
        assert set(losses) == {"cls", "box", "giou", "total"}
        losses["total"].backward()
        model.eval()
        with torch.no_grad():
            result = model.predict(torch.rand(1, 3, 128, 128))[0]
        assert sorted(result) == ["boxes", "labels", "scores"]

    def test_never_exceeds_the_slate_size(self):
        model = build("q-n").eval()
        model.score_threshold = -1.0  # keep every slot
        with torch.no_grad():
            result = model.predict(torch.rand(1, 3, 128, 128))[0]
        assert result["boxes"].shape[0] <= model.head.num_queries

    def test_scores_are_descending(self):
        model = build("q-n").eval()
        model.score_threshold = -1.0
        with torch.no_grad():
            scores = model.predict(torch.rand(1, 3, 128, 128))[0]["scores"]
        assert bool((scores[:-1] >= scores[1:]).all())

    def test_requires_a_slate_head(self):
        dense = build("n")
        with pytest.raises(ModelError):
            LofopQuery(dense.backbone, dense.neck, dense.head)

    def test_learns_without_producing_duplicates(self):
        # The core NMS-free claim: one-to-one matching teaches the slate to
        # emit one box per object, so no suppression pass is needed.
        from lofop.ops import iou_matrix

        torch.manual_seed(0)
        model = build("q-n", num_classes=2)
        image = torch.rand(1, 3, 128, 128)
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [70.0, 75.0, 115.0, 120.0]])
        targets = [{"boxes": boxes, "labels": torch.tensor([0, 1])}]

        optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3)
        for _ in range(220):
            loss = model.compute_losses(image, targets)["total"]
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        model.eval()
        with torch.no_grad():
            result = model.predict(image)[0]
        assert result["boxes"].shape[0] == 2
        assert sorted(result["labels"].tolist()) == [0, 1]
        # No NMS ran, so any duplicate slot would survive into the output.
        overlaps = iou_matrix(result["boxes"].tolist(), result["boxes"].tolist())
        assert overlaps[0][1] < 0.3
        # And the boxes actually landed on the objects.
        truth = iou_matrix(result["boxes"].tolist(), boxes.tolist())
        assert max(truth[0]) > 0.8 and max(truth[1]) > 0.8


class TestSdkIntegration:
    def test_query_variant_through_the_sdk(self):
        from lofop import Detector

        detector = Detector("q-n", num_classes=2, image_size=128, device="cpu")
        assert isinstance(detector.model, LofopQuery)
        [result] = detector.predict(torch.rand(3, 128, 128), score_threshold=0.0)
        assert result.tracker_ids is None

    def test_nms_mode_is_rejected_for_query_variants(self):
        from lofop import Detector

        with pytest.raises(ModelError, match="NMS-free"):
            Detector("q-n", num_classes=2, image_size=128, device="cpu", nms_mode="soft")

    def test_export_is_guarded(self, tmp_path):
        from lofop import Detector

        detector = Detector("q-n", num_classes=2, image_size=128, device="cpu")
        with pytest.raises(LofopError, match="does not yet cover"):
            detector.export(tmp_path / "model.onnx")

    def test_track_yields_identities(self, tmp_path):
        pytest.importorskip("numpy")
        from PIL import Image

        from lofop import Detector

        frames = []
        for index in range(4):
            path = tmp_path / f"frame{index}.png"
            Image.new("RGB", (128, 128), (20 * index, 90, 140)).save(path)
            frames.append(path)

        detector = Detector("n", num_classes=2, image_size=128, device="cpu")
        outputs = list(detector.track(frames, score_threshold=0.0, confirm_after=1))
        assert len(outputs) == 4
        for detections in outputs:
            assert detections.tracker_ids is not None
            assert len(detections.tracker_ids) == len(detections.boxes)
