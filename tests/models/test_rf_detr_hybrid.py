"""Unit tests for RF-DETR ReceptiveFieldBlock and RT-DETR HybridEncoder components."""

from pathlib import Path

import torch

from lofop.core.config import Config
from lofop.models.hybrid_encoder import AIFI, HybridEncoder
from lofop.models.rf_blocks import ReceptiveFieldBlock, RFRidgeNet
from lofop.registries import HUB


def test_receptive_field_block():
    block = ReceptiveFieldBlock(in_channels=64, out_channels=64)
    x = torch.randn(2, 64, 20, 20)
    out = block(x)

    assert out.shape == (2, 64, 20, 20)
    # Check gradient backpropagation
    loss = out.sum()
    loss.backward()
    for param in block.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_rf_ridge_net():
    backbone = RFRidgeNet(
        widths=(32, 64, 128, 256),
        depths=(1, 1, 2, 1),
    )
    x = torch.randn(2, 3, 256, 256)
    c3, c4, c5 = backbone(x)

    # Strides: 8, 16, 32
    assert c3.shape == (2, 64, 32, 32)
    assert c4.shape == (2, 128, 16, 16)
    assert c5.shape == (2, 256, 8, 8)
    assert backbone.out_channels == (64, 128, 256)


def test_aifi_intra_scale_attention():
    aifi = AIFI(embed_dim=64, num_heads=4)
    x = torch.randn(2, 64, 10, 10)
    out = aifi(x)

    assert out.shape == (2, 64, 10, 10)
    # Verify permutation awareness through position embeddings
    x_reversed = x.flip(dims=[-1])
    out_rev = aifi(x_reversed)
    assert not torch.allclose(out.flip(dims=[-1]), out_rev, atol=1e-3)


def test_hybrid_encoder():
    encoder = HybridEncoder(
        in_channels=(64, 128, 256),
        width=64,
        num_heads=4,
    )
    c3 = torch.randn(2, 64, 32, 32)
    c4 = torch.randn(2, 128, 16, 16)
    c5 = torch.randn(2, 256, 8, 8)

    p3, p4, p5 = encoder([c3, c4, c5])
    assert p3.shape == (2, 64, 32, 32)
    assert p4.shape == (2, 64, 16, 16)
    assert p5.shape == (2, 64, 8, 8)


def test_rt_n_model_assembly_and_forward():
    config_path = (
        Path(__file__).resolve().parents[2]
        / "lofop"
        / "configs"
        / "lofop-detect"
        / "rt-n.yaml"
    )
    cfg = Config.load(config_path)
    model = HUB.build(cfg.model)
    model.eval()

    # Forward pass at 256x256
    x = torch.randn(1, 3, 256, 256)
    with torch.inference_mode():
        detections = model.predict(x)
        assert len(detections) == 1
        assert "boxes" in detections[0]
        assert "scores" in detections[0]
        assert "labels" in detections[0]
