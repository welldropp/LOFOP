// Implementation of the LOFOP C ABI (include/lofop/lofop_c.h).
//
// A thin, exception-free shell over the C++ API: every entry point catches
// everything, stores a thread-local message, and returns a status code, so no
// exception ever unwinds across the ABI boundary into a caller that cannot
// handle it (Go, Rust, C#, Java).

#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "lofop/lofop.hpp"
#include "lofop/lofop_c.h"

namespace {

// Thread-local so concurrent callers never overwrite each other's diagnostics.
thread_local std::string g_last_error;

void set_error(const std::string& message) { g_last_error = message; }

}  // namespace

// Holds the detector plus the buffer backing the most recent result, which is
// what lets the C API hand out a pointer the caller does not have to free.
struct lofop_detector {
    std::unique_ptr<lofop::Detector> detector;
    std::vector<lofop_detection> results;
};

namespace {

lofop::Options to_cpp_options(const lofop_options* options) {
    lofop::Options result;
    if (options == nullptr) {
        return result;
    }
    result.score_threshold = options->score_threshold;
    result.nms_iou = options->nms_iou;
    result.max_detections = options->max_detections;
    result.image_size = options->image_size;
    result.pad_value = options->pad_value;
    result.soft_nms = options->soft_nms != 0;
    result.soft_sigma = options->soft_sigma;
    return result;
}

int32_t emit(lofop_detector* handle, const std::vector<lofop::Detection>& detections,
             const lofop_detection** out) {
    handle->results.clear();
    handle->results.reserve(detections.size());
    for (const auto& detection : detections) {
        lofop_detection item;
        item.x1 = detection.x1;
        item.y1 = detection.y1;
        item.x2 = detection.x2;
        item.y2 = detection.y2;
        item.confidence = detection.confidence;
        item.label = static_cast<int32_t>(detection.label);
        handle->results.push_back(item);
    }
    if (out != nullptr) {
        *out = handle->results.empty() ? nullptr : handle->results.data();
    }
    return static_cast<int32_t>(handle->results.size());
}

}  // namespace

extern "C" {

void lofop_options_default(lofop_options* options) {
    if (options == nullptr) {
        return;
    }
    const lofop::Options defaults;
    options->score_threshold = defaults.score_threshold;
    options->nms_iou = defaults.nms_iou;
    options->max_detections = defaults.max_detections;
    options->image_size = defaults.image_size;
    options->pad_value = defaults.pad_value;
    options->soft_nms = defaults.soft_nms ? 1 : 0;
    options->soft_sigma = defaults.soft_sigma;
}

lofop_detector* lofop_detector_create(const char* model_path, const lofop_options* options) {
    if (model_path == nullptr) {
        set_error("model_path is null");
        return nullptr;
    }
    try {
        std::unique_ptr<lofop_detector> handle(new lofop_detector());
        handle->detector.reset(new lofop::Detector(model_path, to_cpp_options(options)));
        g_last_error.clear();
        return handle.release();
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return nullptr;
    } catch (...) {
        set_error("unknown error while creating the detector");
        return nullptr;
    }
}

void lofop_detector_free(lofop_detector* detector) { delete detector; }

int32_t lofop_detector_predict_path(lofop_detector* detector, const char* image_path,
                                    const lofop_detection** out) {
    if (detector == nullptr || image_path == nullptr) {
        set_error("detector or image_path is null");
        return LOFOP_ERROR;
    }
    try {
        const auto detections = detector->detector->predict(std::string(image_path));
        g_last_error.clear();
        return emit(detector, detections, out);
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return LOFOP_ERROR;
    } catch (...) {
        set_error("unknown error during inference");
        return LOFOP_ERROR;
    }
}

int32_t lofop_detector_predict_pixels(lofop_detector* detector, const uint8_t* pixels,
                                      int32_t width, int32_t height, int32_t channels,
                                      const lofop_detection** out) {
    if (detector == nullptr || pixels == nullptr) {
        set_error("detector or pixels is null");
        return LOFOP_ERROR;
    }
    try {
        const lofop::Image image = lofop::Image::from_pixels(pixels, width, height, channels);
        const auto detections = detector->detector->predict(image);
        g_last_error.clear();
        return emit(detector, detections, out);
    } catch (const std::exception& exc) {
        set_error(exc.what());
        return LOFOP_ERROR;
    } catch (...) {
        set_error("unknown error during inference");
        return LOFOP_ERROR;
    }
}

int32_t lofop_detector_input_size(const lofop_detector* detector) {
    if (detector == nullptr || !detector->detector) {
        return LOFOP_ERROR;
    }
    return static_cast<int32_t>(detector->detector->input_size());
}

const char* lofop_last_error(void) { return g_last_error.c_str(); }

const char* lofop_version(void) { return lofop::kVersion; }

int32_t lofop_has_onnxruntime(void) { return lofop::has_onnxruntime() ? 1 : 0; }

}  // extern "C"
