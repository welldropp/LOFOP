"""Tests for the shared letterbox preprocessing.

The point of these tests is parity. ``letterbox`` has a pure Python reference
and a C++ kernel, and the C++ SDK compiles that same kernel, so a divergence
between them would silently change detection accuracy between the training and
serving paths with nothing to trace it to. The parity tests below are the
guard that makes "Python trains, C++ serves" safe.
"""

from __future__ import annotations

import random
import struct

import pytest

from lofop.core.exceptions import DataError
from lofop.ops.native import load_native
from lofop.ops.preprocess import (
    LetterboxMeta,
    _letterbox_python,
    letterbox,
    unletterbox_boxes,
)

# The kernels compute in float32 while the Python reference computes in
# float64, so agreement is expected to float32 epsilon (1.19e-7), not bit for
# bit. Sampling positions are deliberately computed in double on both sides to
# keep the gap this small even on heavy downscales.
FLOAT32_EPS = 1.2e-07


def native_letterbox_available() -> bool:
    """True when a compiled library exposing the letterbox kernel is loaded."""
    lib = load_native()
    return lib is not None and hasattr(lib, "lofop_letterbox")


def random_pixels(width: int, height: int, channels: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(width * height * channels))


class TestGeometry:
    def test_preserves_aspect_ratio_with_symmetric_padding(self):
        # 128x96 is 4:3; at 64px the image occupies 64x48 with 8px bands.
        _, meta = letterbox(bytes(128 * 96 * 3), 128, 96, size=64)
        assert meta.scale == pytest.approx(0.5)
        assert (meta.pad_left, meta.pad_top) == (0.0, 8.0)

    def test_pads_the_other_axis_for_tall_images(self):
        _, meta = letterbox(bytes(48 * 96 * 3), 48, 96, size=64)
        assert meta.scale == pytest.approx(64 / 96)
        assert meta.pad_top == 0.0
        assert meta.pad_left > 0.0

    def test_square_image_needs_no_padding(self):
        _, meta = letterbox(bytes(40 * 40 * 3), 40, 40, size=20)
        assert (meta.pad_left, meta.pad_top) == (0.0, 0.0)
        assert meta.scale == pytest.approx(0.5)

    def test_output_is_chw_float_in_unit_range(self):
        tensor, _ = letterbox(random_pixels(20, 12, 3, seed=1), 20, 12, size=16)
        assert len(tensor) == 3 * 16 * 16
        assert all(0.0 <= value <= 1.0 for value in tensor)

    def test_padding_uses_the_requested_fill(self):
        # A 32x8 source at size 16 leaves padded rows top and bottom.
        tensor, meta = letterbox(bytes(32 * 8 * 3), 32, 8, size=16, pad_value=0.25)
        assert meta.pad_top > 0
        assert tensor[0] == pytest.approx(0.25)  # first row is padding

    def test_grayscale_is_broadcast_to_three_channels(self):
        pixels = random_pixels(8, 8, 1, seed=2)
        tensor, _ = letterbox(pixels, 8, 8, channels=1, size=8)
        plane = 8 * 8
        assert tensor[:plane] == tensor[plane:2 * plane] == tensor[2 * plane:]

    def test_uniform_image_survives_the_round_trip(self):
        # Every pixel 128 -> every non-padded output value 128/255.
        tensor, meta = letterbox(bytes([128]) * (16 * 16 * 3), 16, 16, size=8)
        assert all(value == pytest.approx(128 / 255, abs=1e-6) for value in tensor)
        assert (meta.pad_left, meta.pad_top) == (0.0, 0.0)


class TestValidation:
    @pytest.mark.parametrize(
        "width,height,size",
        [(0, 10, 8), (10, 0, 8), (10, 10, 0), (-4, 10, 8)],
    )
    def test_rejects_non_positive_dimensions(self, width, height, size):
        with pytest.raises(DataError):
            letterbox(bytes(max(width, 1) * max(height, 1) * 3), width, height, size=size)

    def test_rejects_unsupported_channel_counts(self):
        with pytest.raises(DataError):
            letterbox(bytes(4 * 4 * 4), 4, 4, channels=4, size=4)

    def test_rejects_a_short_pixel_buffer(self):
        with pytest.raises(DataError):
            letterbox(bytes(10), 8, 8, size=8)

    def test_rejects_a_degenerate_meta_on_inverse(self):
        with pytest.raises(DataError):
            unletterbox_boxes([[0, 0, 1, 1]], LetterboxMeta(0.0, 0.0, 0.0, 10, 10))


class TestInverseMapping:
    def test_maps_the_full_frame_back_to_the_original(self):
        _, meta = letterbox(bytes(128 * 96 * 3), 128, 96, size=64)
        boxed = [[
            meta.pad_left, meta.pad_top,
            meta.pad_left + 128 * meta.scale, meta.pad_top + 96 * meta.scale,
        ]]
        assert unletterbox_boxes(boxed, meta)[0] == pytest.approx([0, 0, 128, 96], abs=1e-3)

    def test_clamps_boxes_to_the_image_bounds(self):
        meta = LetterboxMeta(scale=0.5, pad_left=0.0, pad_top=8.0, width=128, height=96)
        # A box running off every edge must come back inside the image.
        result = unletterbox_boxes([[-50.0, -50.0, 500.0, 500.0]], meta)[0]
        assert result == pytest.approx([0.0, 0.0, 128.0, 96.0])

    def test_empty_input_returns_empty(self):
        meta = LetterboxMeta(0.5, 0.0, 0.0, 10, 10)
        assert unletterbox_boxes([], meta) == []

    def test_round_trips_an_interior_box(self):
        _, meta = letterbox(bytes(200 * 100 * 3), 200, 100, size=64)
        original = [[20.0, 30.0, 90.0, 80.0]]
        boxed = [[
            original[0][0] * meta.scale + meta.pad_left,
            original[0][1] * meta.scale + meta.pad_top,
            original[0][2] * meta.scale + meta.pad_left,
            original[0][3] * meta.scale + meta.pad_top,
        ]]
        assert unletterbox_boxes(boxed, meta)[0] == pytest.approx(original[0], abs=1e-3)


@pytest.mark.skipif(
    not native_letterbox_available(),
    reason="native ops library not built; build_native() enables the parity check",
)
class TestNativeParity:
    """The C++ kernel and the Python reference must agree.

    The C++ SDK compiles the very same kernel, so these assertions cover the
    C++ runtime too: if Python and the kernel agree, all three paths agree.
    """

    CASES = [
        (64, 48, 32, 3),     # ordinary downscale, pads vertically
        (17, 40, 16, 3),     # tall, pads horizontally
        (40, 40, 20, 3),     # square, no padding
        (9, 3, 8, 1),        # grayscale, extreme aspect ratio
        (31, 17, 13, 3),     # odd sizes, non-integer scale
        (5, 5, 5, 3),        # identity scale
        (1, 1, 4, 3),        # upscale from a single pixel
        (100, 37, 24, 3),    # heavy downscale, where float error concentrates
        (200, 13, 16, 3),    # extreme aspect ratio
    ]

    @pytest.mark.parametrize("width,height,size,channels", CASES)
    def test_pixels_match_the_reference(self, width, height, size, channels):
        pixels = random_pixels(width, height, channels, seed=width * 31 + height)
        native, _ = letterbox(pixels, width, height, channels=channels, size=size)
        reference, _ = _letterbox_python(pixels, width, height, channels, size, 0.447)
        assert len(native) == len(reference)
        worst = max(abs(a - b) for a, b in zip(native, reference))
        assert worst <= FLOAT32_EPS, f"C++ and Python diverged by {worst:.3e}"

    @pytest.mark.parametrize("width,height,size,channels", CASES)
    def test_geometry_matches_the_reference(self, width, height, size, channels):
        pixels = random_pixels(width, height, channels, seed=7)
        _, native_meta = letterbox(pixels, width, height, channels=channels, size=size)
        _, reference_meta = _letterbox_python(pixels, width, height, channels, size, 0.447)
        as_float32 = struct.unpack("f", struct.pack("f", reference_meta.scale))[0]
        assert native_meta.scale == as_float32
        assert native_meta.pad_left == reference_meta.pad_left
        assert native_meta.pad_top == reference_meta.pad_top

    def test_inverse_mapping_matches_the_reference(self):
        meta = LetterboxMeta(scale=0.375, pad_left=3.0, pad_top=11.0, width=90, height=70)
        boxes = [[4.0, 12.0, 40.0, 44.0], [-5.0, 0.0, 900.0, 900.0], [0.0, 0.0, 0.0, 0.0]]
        native = unletterbox_boxes(boxes, meta)
        expected = []
        for box in boxes:
            expected.append([
                min(max((box[0] - meta.pad_left) / meta.scale, 0.0), 90.0),
                min(max((box[1] - meta.pad_top) / meta.scale, 0.0), 70.0),
                min(max((box[2] - meta.pad_left) / meta.scale, 0.0), 90.0),
                min(max((box[3] - meta.pad_top) / meta.scale, 0.0), 70.0),
            ])
        for actual, wanted in zip(native, expected):
            assert actual == pytest.approx(wanted, abs=1e-4)
