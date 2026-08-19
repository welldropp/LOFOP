"""Minimal image container for the LOFOP inference runtime.

Deliberately small: an :class:`Image` is width, height, channel count, and a
row-major 8-bit HWC buffer -- the same shape the C++ SDK's ``lofop::Image``
holds, so the two runtimes accept the same inputs and the shared preprocessing
kernel can consume either without a copy in the C++ case.

Decoding is not this module's job. Pillow handles it here because the LOFOP
core already depends on Pillow; production callers that already hold pixels
(a decoded video frame, a camera buffer, a ``cv::Mat`` on the C++ side) pass
them straight to :meth:`Image.from_pixels` and skip file I/O entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lofop.core.exceptions import DataError

__all__ = ["Image"]


@dataclass(frozen=True)
class Image:
    """An 8-bit image in row-major HWC layout.

    Attributes:
        width: Width in pixels.
        height: Height in pixels.
        channels: 1 (grayscale) or 3 (RGB).
        pixels: ``height * width * channels`` bytes.
    """

    width: int
    height: int
    channels: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise DataError(
                "Image needs positive dimensions",
                context={"width": self.width, "height": self.height},
            )
        if self.channels not in (1, 3):
            raise DataError(
                "Image supports 1 or 3 channels", context={"channels": self.channels}
            )
        expected = self.width * self.height * self.channels
        if len(self.pixels) != expected:
            raise DataError(
                "Pixel buffer does not match the declared image size",
                context={"expected": expected, "got": len(self.pixels)},
            )

    @classmethod
    def from_pixels(
        cls, pixels: bytes | bytearray, width: int, height: int, channels: int = 3
    ) -> Image:
        """Wrap an existing HWC 8-bit buffer without decoding or copying it."""
        return cls(width, height, channels, bytes(pixels))

    @classmethod
    def load(cls, path: str | Path) -> Image:
        """Decode an image file (any format Pillow reads) into RGB.

        Raises:
            DataError: If the file cannot be opened or decoded.
        """
        try:
            from PIL import Image as PILImage
        except ImportError as exc:  # pragma: no cover - Pillow is a core dep
            raise DataError("Pillow is required to load image files") from exc
        try:
            with PILImage.open(path) as handle:
                rgb = handle.convert("RGB")
                return cls(rgb.width, rgb.height, 3, rgb.tobytes())
        except OSError as exc:
            raise DataError(
                f"Cannot load image: {exc}", context={"path": str(path)}
            ) from exc

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.width, self.height

    def __repr__(self) -> str:
        return f"Image({self.width}x{self.height}, channels={self.channels})"
