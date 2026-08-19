"""Image preprocessing shared by every LOFOP inference path.

Aspect-preserving resize with symmetric padding ("letterboxing") is the
transform a detector's accuracy is calibrated against, so training and serving
must apply it identically. This module is the Python half of a single shared
definition: :func:`letterbox` dispatches to the C++ kernel in
``lofop/csrc/preprocess.cpp`` when the native library is built, and otherwise
runs a pure Python reference that computes the same result. The C++ SDK links
that same source file, so all three paths agree by construction rather than by
convention -- and ``tests/ops/test_preprocess.py`` proves it.

Torch-free by design: this runs anywhere the LOFOP core runs, which is what
lets the deployment path drop PyTorch entirely.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass

from lofop.core.exceptions import DataError
from lofop.ops.native import load_native

__all__ = ["LetterboxMeta", "letterbox", "unletterbox_boxes"]


@dataclass(frozen=True)
class LetterboxMeta:
    """Geometry of one letterbox transform, enough to invert it.

    Attributes:
        scale: Factor the original image was multiplied by.
        pad_left: Pixels of padding added on the left.
        pad_top: Pixels of padding added on the top.
        width: Original image width.
        height: Original image height.
    """

    scale: float
    pad_left: float
    pad_top: float
    width: int
    height: int


def _sample_bilinear(
    pixels: Sequence[int], width: int, height: int, channels: int,
    channel: int, x: float, y: float,
) -> float:
    """Bilinear sample of one channel, clamped at the edges.

    Mirrors ``sample_bilinear`` in ``csrc/preprocess.cpp`` exactly; the two are
    the specification of LOFOP's resampling and must not diverge.
    """
    x = min(max(x, 0.0), float(width - 1))
    y = min(max(y, 0.0), float(height - 1))
    x0 = int(x)
    y0 = int(y)
    x1 = x0 + 1 if x0 + 1 < width else x0
    y1 = y0 + 1 if y0 + 1 < height else y0
    fx = x - x0
    fy = y - y0
    stride = width * channels
    p00 = pixels[y0 * stride + x0 * channels + channel]
    p01 = pixels[y0 * stride + x1 * channels + channel]
    p10 = pixels[y1 * stride + x0 * channels + channel]
    p11 = pixels[y1 * stride + x1 * channels + channel]
    top = p00 + (p01 - p00) * fx
    bottom = p10 + (p11 - p10) * fx
    return top + (bottom - top) * fy


def _letterbox_geometry(width: int, height: int, size: int) -> tuple[float, int, int, int, int]:
    """Scale, destination extent, and padding for one letterbox.

    Shared by both implementations so the geometry cannot drift even if the
    resampling loops do.
    """
    scale = min(size / width, size / height)
    # round-half-away-from-zero, matching std::lround in the C++ kernel;
    # Python's round() is banker's rounding and would disagree on .5 cases.
    new_w = int(width * scale + 0.5)
    new_h = int(height * scale + 0.5)
    new_w = max(1, min(new_w, size))
    new_h = max(1, min(new_h, size))
    return scale, new_w, new_h, (size - new_w) // 2, (size - new_h) // 2


def _letterbox_python(
    pixels: Sequence[int], width: int, height: int, channels: int,
    size: int, pad_value: float,
) -> tuple[list[float], LetterboxMeta]:
    """Pure Python reference implementation. Always available."""
    scale, new_w, new_h, pad_left, pad_top = _letterbox_geometry(width, height, size)
    plane = size * size
    out = [pad_value] * (plane * 3)
    inv_scale_x = width / new_w
    inv_scale_y = height / new_h
    for y in range(new_h):
        src_y = (y + 0.5) * inv_scale_y - 0.5
        row = (y + pad_top) * size
        for x in range(new_w):
            src_x = (x + 0.5) * inv_scale_x - 0.5
            offset = row + x + pad_left
            for c in range(3):
                source_channel = 0 if channels == 1 else c
                value = _sample_bilinear(
                    pixels, width, height, channels, source_channel, src_x, src_y
                )
                out[c * plane + offset] = value / 255.0
    meta = LetterboxMeta(float(scale), float(pad_left), float(pad_top), width, height)
    return out, meta


def letterbox(
    pixels: Sequence[int] | bytes | bytearray,
    width: int,
    height: int,
    *,
    channels: int = 3,
    size: int = 640,
    pad_value: float = 0.447,
) -> tuple[list[float], LetterboxMeta]:
    """Resize an image into a square model input, preserving aspect ratio.

    Args:
        pixels: Row-major HWC pixel data, 8-bit per channel (a ``bytes``
            buffer, ``bytearray``, or any integer sequence).
        width: Source image width in pixels.
        height: Source image height in pixels.
        channels: Channels in ``pixels`` -- 1 (broadcast to RGB) or 3.
        size: Side length of the square destination.
        pad_value: Fill value for the padded margin, in ``[0, 1]``. The
            default is the mid-grey conventionally used by detectors, which
            keeps padding from reading as a dark object edge.

    Returns:
        ``(tensor, meta)`` where ``tensor`` is a flat CHW float list of
        length ``3 * size * size`` with values in ``[0, 1]``, and ``meta``
        is the :class:`LetterboxMeta` needed to map boxes back.

    Raises:
        DataError: On invalid dimensions, channel count, or a pixel buffer
            shorter than ``width * height * channels``.
    """
    if width <= 0 or height <= 0 or size <= 0:
        raise DataError(
            "letterbox needs positive dimensions",
            context={"width": width, "height": height, "size": size},
        )
    if channels not in (1, 3):
        raise DataError("letterbox supports 1 or 3 channels", context={"channels": channels})
    expected = width * height * channels
    if len(pixels) < expected:
        raise DataError(
            "pixel buffer is smaller than the declared image",
            context={"expected": expected, "got": len(pixels)},
        )

    lib = load_native()
    if lib is not None and hasattr(lib, "lofop_letterbox"):
        buffer = pixels if isinstance(pixels, (bytes, bytearray)) else bytes(bytearray(pixels))
        src = (ctypes.c_uint8 * expected).from_buffer_copy(buffer[:expected])
        dst = (ctypes.c_float * (3 * size * size))()
        meta_out = (ctypes.c_float * 3)()
        status = lib.lofop_letterbox(
            src, ctypes.c_int32(width), ctypes.c_int32(height), ctypes.c_int32(channels),
            ctypes.c_int32(size), ctypes.c_float(pad_value), dst, meta_out,
        )
        if status == 0:
            meta = LetterboxMeta(
                float(meta_out[0]), float(meta_out[1]), float(meta_out[2]), width, height
            )
            return list(dst), meta
        # A negative status means the kernel rejected the arguments; fall
        # through to the reference so behaviour never depends on the tier.

    return _letterbox_python(pixels, width, height, channels, size, pad_value)


def unletterbox_boxes(
    boxes: Sequence[Sequence[float]], meta: LetterboxMeta
) -> list[list[float]]:
    """Map boxes from letterboxed coordinates back to original image pixels.

    The exact inverse of :func:`letterbox`'s geometry: padding is removed,
    the scale undone, and boxes are clamped to the image bounds.

    Args:
        boxes: ``(N, 4)`` xyxy boxes in letterboxed pixel coordinates.
        meta: Geometry returned by :func:`letterbox`.

    Returns:
        ``(N, 4)`` xyxy boxes in the original image's pixel coordinates.
    """
    if not boxes:
        return []
    if not meta.scale > 0:
        raise DataError("letterbox meta has a non-positive scale",
                        context={"scale": meta.scale})

    lib = load_native()
    if lib is not None and hasattr(lib, "lofop_unletterbox_boxes"):
        count = len(boxes)
        flat = (ctypes.c_float * (count * 4))()
        for i, box in enumerate(boxes):
            flat[i * 4] = float(box[0])
            flat[i * 4 + 1] = float(box[1])
            flat[i * 4 + 2] = float(box[2])
            flat[i * 4 + 3] = float(box[3])
        meta_in = (ctypes.c_float * 3)(meta.scale, meta.pad_left, meta.pad_top)
        out = (ctypes.c_float * (count * 4))()
        status = lib.lofop_unletterbox_boxes(
            flat, ctypes.c_int32(count), meta_in,
            ctypes.c_int32(meta.width), ctypes.c_int32(meta.height), out,
        )
        if status == 0:
            return [list(out[i * 4:i * 4 + 4]) for i in range(count)]

    result = []
    for box in boxes:
        x1 = (float(box[0]) - meta.pad_left) / meta.scale
        y1 = (float(box[1]) - meta.pad_top) / meta.scale
        x2 = (float(box[2]) - meta.pad_left) / meta.scale
        y2 = (float(box[3]) - meta.pad_top) / meta.scale
        result.append([
            min(max(x1, 0.0), float(meta.width)),
            min(max(y1, 0.0), float(meta.height)),
            min(max(x2, 0.0), float(meta.width)),
            min(max(y2, 0.0), float(meta.height)),
        ])
    return result
