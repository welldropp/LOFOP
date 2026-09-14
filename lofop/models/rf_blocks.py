"""Receptive Field Blocks (RFB) and RF-enhanced backbone for LOFOP.

Integrates multi-branch dilated convolutions inspired by human visual cortex
receptive fields to expand the effective receptive field for multi-scale
and small-object detection without the quadratic cost of full self-attention.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.models.blocks import ConvNormAct, RidgeBlock
from lofop.registries import BACKBONES


class ReceptiveFieldBlock(nn.Module):
    """Multi-branch Receptive Field Block (RFB).

    Splits input channels across four parallel branches with varying receptive fields:
      - Branch 0: 1x1 conv (global/context feature).
      - Branch 1: 1x1 conv -> 3x3 conv (standard local receptive field).
      - Branch 2: 1x1 conv -> 3x3 conv -> 3x3 dilated conv (dilation=2, intermediate scale).
      - Branch 3: 1x1 conv -> 3x3 conv -> 3x3 dilated conv (dilation=3, panoramic scale).

    Concatenates branch outputs, projects back to out_channels, and adds a residual shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int, scale: float = 0.1) -> None:
        super().__init__()
        branch_channels = max(1, out_channels // 4)
        total_branch_channels = branch_channels * 4

        self.branch0 = ConvNormAct(in_channels, branch_channels, kernel_size=1)

        self.branch1 = nn.Sequential(
            ConvNormAct(in_channels, branch_channels, kernel_size=1),
            ConvNormAct(branch_channels, branch_channels, kernel_size=3),
        )

        self.branch2 = nn.Sequential(
            ConvNormAct(in_channels, branch_channels, kernel_size=1),
            ConvNormAct(branch_channels, branch_channels, kernel_size=3),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            ConvNormAct(in_channels, branch_channels, kernel_size=1),
            ConvNormAct(branch_channels, branch_channels, kernel_size=3),
            nn.Conv2d(
                branch_channels,
                branch_channels,
                kernel_size=3,
                padding=3,
                dilation=3,
                bias=False,
            ),
            nn.BatchNorm2d(branch_channels),
            nn.SiLU(inplace=True),
        )

        self.project = ConvNormAct(total_branch_channels, out_channels, kernel_size=1, act=False)

        if in_channels != out_channels:
            self.shortcut = ConvNormAct(in_channels, out_channels, kernel_size=1, act=False)
        else:
            self.shortcut = nn.Identity()

        self.scale = nn.Parameter(torch.tensor(scale))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b0, b1, b2, b3], dim=1)
        out = self.project(concat)
        return self.act(self.shortcut(x) + self.scale * out)


@BACKBONES.register()
class RFRidgeNet(nn.Module):
    """Receptive-Field enhanced RidgeNet backbone.

    Combines RidgeNet's inverted residual bottleneck stages with ReceptiveFieldBlocks
    in stages C4 and C5 for multi-scale feature representation.
    """

    def __init__(
        self,
        widths: tuple[int, int, int, int] = (48, 96, 192, 384),
        depths: tuple[int, int, int, int] = (1, 2, 4, 2),
        in_channels: int = 3,
        expansion: float = 2.0,
    ) -> None:
        super().__init__()
        if len(widths) != 4 or len(depths) != 4:
            raise ModelError(
                "RFRidgeNet needs exactly 4 stage widths and depths",
                context={"widths": list(widths), "depths": list(depths)},
            )
        self.widths = tuple(widths)
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, widths[0] // 2, stride=2),
            ConvNormAct(widths[0] // 2, widths[0], stride=2),
        )

        stages = []
        prev = widths[0]
        for stage_index, (width, depth) in enumerate(zip(widths, depths)):
            layers: list[nn.Module] = []
            if stage_index > 0:
                layers.append(ConvNormAct(prev, width, stride=2))
            elif prev != width:
                layers.append(ConvNormAct(prev, width, kernel_size=1))

            # Stages 0 and 1: Standard RidgeBlocks (fast, high-resolution)
            # Stages 2 and 3: Enhanced with ReceptiveFieldBlock
            for i in range(depth):
                layers.append(RidgeBlock(width, expansion))
                if stage_index >= 2 and i == depth - 1:
                    layers.append(ReceptiveFieldBlock(width, width))

            stages.append(nn.Sequential(*layers))
            prev = width
        self.stages = nn.ModuleList(stages)

    @property
    def out_channels(self) -> tuple[int, int, int]:
        """Channels of the returned C3/C4/C5 maps."""
        return self.widths[1:]

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features[1:]
