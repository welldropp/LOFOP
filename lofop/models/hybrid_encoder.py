"""RT-DETR Efficient Hybrid Encoder (AIFI + CCFM) for LOFOP.

Combines:
  - AIFI: Attention-based Intra-scale Feature Interaction operating strictly on
    stride-32 (S5) to model global context with multi-head self-attention.
  - CCFM: Cross-scale Feature-fusion Module with multi-layer RepBlocks linking
    multi-scale feature maps (P3, P4, P5) in bi-directional top-down and bottom-up passes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.blocks import ConvNormAct
from lofop.registries import NECKS


def build_2d_sincos_position_embedding(
    width: int, height: int, embed_dim: int, device: torch.device
) -> Tensor:
    """Generate 2D sinusoidal position embeddings."""
    dim_per_axis = embed_dim // 2
    omega = torch.arange(dim_per_axis // 2, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / (dim_per_axis // 2)))

    y_grid = torch.arange(height, dtype=torch.float32, device=device)
    x_grid = torch.arange(width, dtype=torch.float32, device=device)
    y_grid, x_grid = torch.meshgrid(y_grid, x_grid, indexing="ij")

    out_x = torch.einsum("hw,d->hwd", x_grid, omega)
    out_y = torch.einsum("hw,d->hwd", y_grid, omega)

    pos_x = torch.cat([torch.sin(out_x), torch.cos(out_x)], dim=-1)
    pos_y = torch.cat([torch.sin(out_y), torch.cos(out_y)], dim=-1)
    pos = torch.cat([pos_x, pos_y], dim=-1)  # (H, W, embed_dim)
    return pos.flatten(0, 1).unsqueeze(0)  # (1, H*W, embed_dim)


class AIFI(nn.Module):
    """Attention-based Intra-scale Feature Interaction for high-level S5 feature maps."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        hidden_dim = ffn_dim or (embed_dim * 2)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        src = x.flatten(2).transpose(1, 2)
        pos = build_2d_sincos_position_embedding(width, height, channels, x.device)

        # Pre-LN Self-Attention with sinusoidal position embeddings
        src_norm = self.norm1(src)
        q = k = src_norm + pos
        v = src_norm
        attn_out, _ = self.attn(q, k, v)
        src = src + attn_out

        # Pre-LN FFN
        src = src + self.ffn(self.norm2(src))

        # Reshape back to (B, C, H, W)
        out = src.transpose(1, 2).reshape(batch, channels, height, width)
        return out


class RepBlock(nn.Module):
    """Convolutional fusion block for CCFM cross-scale feature mixing."""

    def __init__(self, in_channels: int, out_channels: int, depth: int = 2) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=1)
        layers = []
        for _ in range(depth):
            layers.append(
                nn.Sequential(
                    ConvNormAct(out_channels, out_channels, kernel_size=3),
                    ConvNormAct(out_channels, out_channels, kernel_size=3),
                )
            )
        self.blocks = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        for block in self.blocks:
            x = x + block(x)
        return x


@NECKS.register()
class HybridEncoder(nn.Module):
    """RT-DETR style Efficient Hybrid Encoder (AIFI + CCFM).

    Args:
        in_channels: Channels of incoming feature maps [C3, C4, C5].
        width: Uniform feature pyramid channel width.
        num_heads: Number of attention heads in AIFI.
        depth: Convolution depth inside each RepBlock.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (96, 192, 384),
        width: int = 96,
        num_heads: int = 4,
        depth: int = 1,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ModelError(
                "HybridEncoder expects exactly 3 input feature levels [C3, C4, C5]",
                context={"in_channels": list(in_channels)},
            )
        self.width = width

        # Lateral 1x1 convolutions for C3, C4, C5
        self.laterals = nn.ModuleList(
            ConvNormAct(c, width, kernel_size=1) for c in in_channels
        )

        # Intra-scale feature interaction on C5 (stride 32)
        self.aifi = AIFI(embed_dim=width, num_heads=num_heads)

        # Top-down pathway
        self.reduce_s5 = ConvNormAct(width, width, kernel_size=1)
        self.fuse_p4_td = RepBlock(width * 2, width, depth=depth)
        self.reduce_p4 = ConvNormAct(width, width, kernel_size=1)
        self.fuse_p3_td = RepBlock(width * 2, width, depth=depth)

        # Bottom-up pathway
        self.downsample_p3 = ConvNormAct(width, width, kernel_size=3, stride=2)
        self.fuse_p4_bu = RepBlock(width * 2, width, depth=depth)
        self.downsample_p4 = ConvNormAct(width, width, kernel_size=3, stride=2)
        self.fuse_p5_bu = RepBlock(width * 2, width, depth=depth)

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        if len(features) != 3:
            raise ModelError(
                "HybridEncoder received the wrong number of feature maps",
                context={"got": len(features)},
            )

        c3, c4, c5 = [lateral(feat) for lateral, feat in zip(self.laterals, features)]

        # 1. AIFI attention on C5
        s5_attn = self.aifi(c5)

        # 2. Top-Down Fusion (CCFM)
        s5_up = F.interpolate(self.reduce_s5(s5_attn), size=c4.shape[-2:], mode="nearest")
        p4_td = self.fuse_p4_td(torch.cat([s5_up, c4], dim=1))

        p4_up = F.interpolate(self.reduce_p4(p4_td), size=c3.shape[-2:], mode="nearest")
        p3_out = self.fuse_p3_td(torch.cat([p4_up, c3], dim=1))

        # 3. Bottom-Up Fusion (CCFM)
        p3_down = self.downsample_p3(p3_out)
        p4_out = self.fuse_p4_bu(torch.cat([p3_down, p4_td], dim=1))

        p4_down = self.downsample_p4(p4_out)
        p5_out = self.fuse_p5_bu(torch.cat([p4_down, s5_attn], dim=1))

        return [p3_out, p4_out, p5_out]
