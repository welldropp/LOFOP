// Native image preprocessing for LOFOP inference.
//
// Detection accuracy depends on the training and serving pipelines preparing
// pixels identically. Defining that transform twice -- once in Python, once in
// C++ -- invites silent drift: a half-pixel disagreement costs accuracy with
// no error to trace. So the transform is defined exactly once, here, and both
// language bindings call this code.
//
// `lofop_letterbox` performs an aspect-preserving resize with symmetric
// padding (the standard detector preprocessing) and writes the CHW float
// tensor a model consumes; `lofop_unletterbox_boxes` maps predicted boxes
// back to original image coordinates. The two are exact inverses of each
// other's geometry.
//
// Exposed through a plain C ABI (loaded via ctypes, linked directly by the
// C++ SDK) so the library builds with any C++17 compiler and needs no Python
// headers, no pybind11, and no framework runtime.

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

// Bilinear sample of one channel at a source position, clamped at the edges.
// LOFOP deliberately defines its own sampler rather than matching a specific
// imaging library: the requirement is that Python and C++ agree bit-for-bit,
// which a small, fully specified kernel guarantees and a third-party
// resampler (whose antialiasing differs between versions) does not.
// Sampling positions are computed in double throughout. On a heavy downscale
// the destination-to-source product grows large enough that float32 rounding
// shifts the sample by microns of a pixel -- harmless on its own, but it makes
// the C++ and Python results diverge by more than the 8-bit input can justify.
// Double positions keep the two implementations agreeing to float32 epsilon at
// no measurable cost: this loop is memory-bound, not arithmetic-bound.
inline float sample_bilinear(const uint8_t* src, int32_t width, int32_t height,
                             int32_t channels, int32_t channel, double x, double y) {
    if (x < 0.0) {
        x = 0.0;
    }
    if (y < 0.0) {
        y = 0.0;
    }
    const double max_x = static_cast<double>(width - 1);
    const double max_y = static_cast<double>(height - 1);
    if (x > max_x) {
        x = max_x;
    }
    if (y > max_y) {
        y = max_y;
    }
    const int32_t x0 = static_cast<int32_t>(x);
    const int32_t y0 = static_cast<int32_t>(y);
    const int32_t x1 = (x0 + 1 < width) ? x0 + 1 : x0;
    const int32_t y1 = (y0 + 1 < height) ? y0 + 1 : y0;
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);

    const int64_t stride = static_cast<int64_t>(width) * channels;
    const double p00 = src[static_cast<int64_t>(y0) * stride + static_cast<int64_t>(x0) * channels + channel];
    const double p01 = src[static_cast<int64_t>(y0) * stride + static_cast<int64_t>(x1) * channels + channel];
    const double p10 = src[static_cast<int64_t>(y1) * stride + static_cast<int64_t>(x0) * channels + channel];
    const double p11 = src[static_cast<int64_t>(y1) * stride + static_cast<int64_t>(x1) * channels + channel];

    const double top = p00 + (p01 - p00) * fx;
    const double bottom = p10 + (p11 - p10) * fx;
    return static_cast<float>(top + (bottom - top) * fy);
}

}  // namespace

extern "C" {

// Aspect-preserving resize of an HWC uint8 image into a CHW float32 tensor.
//
//   src       (sh, sw, channels) uint8, channels 1 or 3; 1 is broadcast to 3
//   size      side length of the square destination
//   pad_value fill for the padded margin, in [0, 1]
//   dst       (3, size, size) float32 in [0, 1], caller-allocated
//   meta      [scale, pad_left, pad_top], caller-allocated (3 floats)
//
// Returns 0 on success, -1 on invalid arguments.
int32_t lofop_letterbox(const uint8_t* src, int32_t sw, int32_t sh, int32_t channels,
                        int32_t size, float pad_value, float* dst, float* meta) {
    if (src == nullptr || dst == nullptr || meta == nullptr) {
        return -1;
    }
    if (sw <= 0 || sh <= 0 || size <= 0 || (channels != 1 && channels != 3)) {
        return -1;
    }

    const float scale = std::min(static_cast<float>(size) / static_cast<float>(sw),
                                 static_cast<float>(size) / static_cast<float>(sh));
    // Round to nearest, then clamp: a scale of exactly 1 must not round the
    // extent past the destination on either axis.
    int32_t new_w = static_cast<int32_t>(std::lround(static_cast<double>(sw) * scale));
    int32_t new_h = static_cast<int32_t>(std::lround(static_cast<double>(sh) * scale));
    new_w = std::max(1, std::min(new_w, size));
    new_h = std::max(1, std::min(new_h, size));
    const int32_t pad_left = (size - new_w) / 2;
    const int32_t pad_top = (size - new_h) / 2;

    const int64_t plane = static_cast<int64_t>(size) * size;
    for (int64_t i = 0; i < plane * 3; ++i) {
        dst[i] = pad_value;
    }

    // Map destination pixel centres back through the scale, the convention
    // that keeps the transform symmetric and its inverse exact.
    const double inv_scale_x = static_cast<double>(sw) / static_cast<double>(new_w);
    const double inv_scale_y = static_cast<double>(sh) / static_cast<double>(new_h);
    for (int32_t y = 0; y < new_h; ++y) {
        const double src_y = (static_cast<double>(y) + 0.5) * inv_scale_y - 0.5;
        for (int32_t x = 0; x < new_w; ++x) {
            const double src_x = (static_cast<double>(x) + 0.5) * inv_scale_x - 0.5;
            const int64_t offset = static_cast<int64_t>(y + pad_top) * size + (x + pad_left);
            for (int32_t c = 0; c < 3; ++c) {
                const int32_t source_channel = (channels == 1) ? 0 : c;
                const float value =
                    sample_bilinear(src, sw, sh, channels, source_channel, src_x, src_y);
                dst[static_cast<int64_t>(c) * plane + offset] = value / 255.0f;
            }
        }
    }

    meta[0] = scale;
    meta[1] = static_cast<float>(pad_left);
    meta[2] = static_cast<float>(pad_top);
    return 0;
}

// Map boxes from letterboxed coordinates back to original image pixels,
// undoing `lofop_letterbox`'s padding and scale and clamping to the image.
//
//   boxes  (n, 4) float32 xyxy in letterboxed pixels; updated in place when
//          `out` aliases it, otherwise written to `out`
//   meta   [scale, pad_left, pad_top] as produced by lofop_letterbox
//
// Returns 0 on success, -1 on invalid arguments.
int32_t lofop_unletterbox_boxes(const float* boxes, int32_t n, const float* meta,
                                int32_t width, int32_t height, float* out) {
    if (boxes == nullptr || meta == nullptr || out == nullptr || n < 0) {
        return -1;
    }
    const float scale = meta[0];
    if (!(scale > 0.0f)) {
        return -1;
    }
    const float pad_left = meta[1];
    const float pad_top = meta[2];
    const float max_x = static_cast<float>(width);
    const float max_y = static_cast<float>(height);
    for (int32_t i = 0; i < n; ++i) {
        const float* box = boxes + static_cast<int64_t>(i) * 4;
        float* dst = out + static_cast<int64_t>(i) * 4;
        const float x1 = (box[0] - pad_left) / scale;
        const float y1 = (box[1] - pad_top) / scale;
        const float x2 = (box[2] - pad_left) / scale;
        const float y2 = (box[3] - pad_top) / scale;
        dst[0] = std::min(std::max(x1, 0.0f), max_x);
        dst[1] = std::min(std::max(y1, 0.0f), max_y);
        dst[2] = std::min(std::max(x2, 0.0f), max_x);
        dst[3] = std::min(std::max(y2, 0.0f), max_y);
    }
    return 0;
}

}  // extern "C"
