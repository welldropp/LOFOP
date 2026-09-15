"""SwiftNet: an inverted-residual backbone for the edge tier.

LOFOP's default :class:`~lofop.models.backbone.RidgeNet` is a large-kernel
depthwise design tuned for accuracy per parameter. SwiftNet trades some of
that for latency on weak hardware, using the inverted-residual pattern that
mobile CPUs execute well: expand pointwise, filter depthwise at low spatial
cost, project back down, and add a residual when the shapes allow it.

Two deliberate choices over a textbook inverted residual:

* **Squeeze-excite on the deepest two stages only.** Channel gating pays for
  itself where channels are many and spatial size is small; on early
  high-resolution stages it costs latency for little accuracy.
* **ReLU6 rather than SiLU.** It quantises cleanly to INT8, which matters for
  the TensorRT and edge-runtime paths LOFOP targets.

Emits ``[C3, C4, C5]`` at strides 8/16/32, the same contract as RidgeNet, so
it drops into any LOFOP neck without further changes.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from lofop.core.exceptions import ModelError
from lofop.registries import BACKBONES


def _round_channels(channels: float, divisor: int = 8) -> int:
    """Round to a multiple of ``divisor``, never dropping more than 10%."""
    rounded = max(divisor, int(channels + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * channels:
        rounded += divisor
    return int(rounded)


class SqueezeExcite(nn.Module):
    """Global-context channel gate."""

    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(8, _round_channels(channels * ratio))
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        weights = x.mean(dim=(2, 3), keepdim=True)
        weights = torch.relu(self.reduce(weights))
        return x * torch.sigmoid(self.expand(weights))


class InvertedResidual(nn.Module):
    """Expand -> depthwise -> project, with an optional residual.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        stride: Spatial stride of the depthwise convolution.
        expansion: Channel multiplier for the internal bottleneck. An
            expansion of 1 skips the expand convolution entirely.
        kernel_size: Depthwise kernel size.
        use_se: Insert a :class:`SqueezeExcite` after the depthwise stage.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: float = 4.0,
        kernel_size: int = 3,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        hidden = _round_channels(in_channels * expansion)
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []
        if hidden != in_channels:
            layers += [
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(
                hidden, hidden, kernel_size, stride=stride,
                padding=kernel_size // 2, groups=hidden, bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
        ]
        if use_se:
            layers.append(SqueezeExcite(hidden))
        layers += [
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        return x + out if self.use_residual else out


@BACKBONES.register()
class SwiftNet(nn.Module):
    """Inverted-residual feature extractor emitting ``[C3, C4, C5]``.

    Args:
        widths: Output channels of the four stages (strides 4/8/16/32).
        depths: Number of inverted-residual blocks per stage.
        expansion: Bottleneck expansion inside each block.
        stem_channels: Channels of the stride-2 stem. Defaults to the first
            stage's width.
        kernel_size: Depthwise kernel size used throughout.

    Returns the last three stages, so ``widths[1:]`` are the channel counts a
    neck should be configured with.
    """

    def __init__(
        self,
        widths: tuple[int, ...] = (24, 48, 96, 192),
        depths: tuple[int, ...] = (1, 2, 3, 1),
        expansion: float = 4.0,
        stem_channels: int | None = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if len(widths) != 4 or len(depths) != 4:
            raise ModelError(
                "SwiftNet expects 4 stage widths and 4 stage depths",
                context={"widths": list(widths), "depths": list(depths)},
            )
        if min(depths) < 1:
            raise ModelError("Every stage needs at least one block",
                             context={"depths": list(depths)})

        stem = _round_channels(stem_channels if stem_channels is not None else widths[0])
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem),
            nn.ReLU6(inplace=True),
        )

        stages = []
        in_channels = stem
        for index, (width, depth) in enumerate(zip(widths, depths)):
            out_channels = _round_channels(width)
            blocks = [
                InvertedResidual(
                    in_channels, out_channels, stride=2, expansion=expansion,
                    kernel_size=kernel_size, use_se=index >= 2,
                )
            ]
            blocks += [
                InvertedResidual(
                    out_channels, out_channels, stride=1, expansion=expansion,
                    kernel_size=kernel_size, use_se=index >= 2,
                )
                for _ in range(depth - 1)
            ]
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.out_channels = tuple(_round_channels(w) for w in widths[1:])

    def forward(self, x: Tensor) -> list[Tensor]:
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features[1:]
