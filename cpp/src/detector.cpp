// The LOFOP C++ inference pipeline: preprocess, run, decode, suppress, map back.
//
// Every numerical step here calls a kernel that already existed and is already
// tested: lofop_letterbox and lofop_unletterbox_boxes from csrc/preprocess.cpp,
// lofop_decode_dense, lofop_nms, and lofop_soft_nms from csrc/box_ops.cpp.
// Python's lofop.runtime.Detector calls the same four functions through
// ctypes, which is why the two runtimes agree on a shared model rather than
// merely intending to.

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "lofop/lofop.hpp"

// The native kernels, declared rather than re-included: box_ops.cpp and
// preprocess.cpp are compiled into this library and expose a plain C ABI.
extern "C" {
int32_t lofop_letterbox(const uint8_t* src, int32_t sw, int32_t sh, int32_t channels,
                        int32_t size, float pad_value, float* dst, float* meta);
int32_t lofop_unletterbox_boxes(const float* boxes, int32_t n, const float* meta,
                                int32_t width, int32_t height, float* out);
int32_t lofop_decode_dense(const float* scores, int32_t n, int32_t num_classes, float threshold,
                           int32_t* index_out, int32_t* label_out, float* score_out);
int32_t lofop_nms(const float* boxes, const float* scores, int32_t n, float iou_threshold,
                  int32_t max_keep, int32_t* keep_out);
int32_t lofop_soft_nms(const float* boxes, const float* scores, int32_t n, float iou_threshold,
                       float sigma, float score_threshold, int32_t method, int32_t max_keep,
                       int32_t* keep_out, float* score_out);
}

namespace lofop {
namespace {

// Class-aware suppression is expressed by shifting each class into its own
// coordinate band, so a single-class kernel never compares across classes.
// This must stay equal to lofop.ops.boxes.CLASS_OFFSET: the two runtimes
// would otherwise suppress differently on multi-class images.
constexpr float kClassOffset = 1e7f;

std::vector<float> shift_by_class(const std::vector<float>& boxes,
                                  const std::vector<int32_t>& labels) {
    std::vector<float> shifted(boxes.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const float offset = kClassOffset * static_cast<float>(labels[i]);
        for (int c = 0; c < 4; ++c) {
            shifted[i * 4 + c] = boxes[i * 4 + c] + offset;
        }
    }
    return shifted;
}

}  // namespace

Detector::Detector(const std::string& model_path, Options options)
    : engine_(make_onnxruntime_engine(model_path, options.image_size)),
      options_(std::move(options)) {}

Detector::Detector(std::unique_ptr<Engine> engine, Options options)
    : engine_(std::move(engine)), options_(std::move(options)) {
    if (!engine_) {
        throw Error("Detector received a null engine");
    }
}

std::string Detector::class_name(int label) const {
    if (label >= 0 && static_cast<std::size_t>(label) < options_.class_names.size()) {
        return options_.class_names[static_cast<std::size_t>(label)];
    }
    return "class_" + std::to_string(label);
}

std::vector<Detection> Detector::predict(const std::string& image_path) const {
    return predict(Image::load(image_path));
}

std::vector<Detection> Detector::predict(const Image& image) const {
    if (image.empty()) {
        throw Error("Detector::predict received an empty image");
    }
    const int size = engine_->input_size();
    if (size <= 0) {
        throw Error("Engine reported a non-positive input size");
    }

    // 1. Preprocess: aspect-preserving resize into a CHW float tensor.
    std::vector<float> tensor(static_cast<std::size_t>(size) * size * 3);
    float meta[3] = {0.0f, 0.0f, 0.0f};
    if (lofop_letterbox(image.pixels().data(), image.width(), image.height(), image.channels(),
                        size, options_.pad_value, tensor.data(), meta) != 0) {
        throw Error("Preprocessing failed for the supplied image");
    }

    // 2. Execute the graph.
    const DenseOutput dense = engine_->run(tensor.data(), tensor.size());
    if (dense.num_candidates <= 0 || dense.num_classes <= 0) {
        return {};
    }

    // 3. Dense decode: best class per candidate, thresholded.
    const int32_t n = dense.num_candidates;
    std::vector<int32_t> indices(static_cast<std::size_t>(n));
    std::vector<int32_t> labels(static_cast<std::size_t>(n));
    std::vector<float> scores(static_cast<std::size_t>(n));
    const int32_t candidates =
        lofop_decode_dense(dense.scores.data(), n, dense.num_classes, options_.score_threshold,
                           indices.data(), labels.data(), scores.data());
    if (candidates <= 0) {
        return {};
    }
    indices.resize(static_cast<std::size_t>(candidates));
    labels.resize(static_cast<std::size_t>(candidates));
    scores.resize(static_cast<std::size_t>(candidates));

    std::vector<float> candidate_boxes(static_cast<std::size_t>(candidates) * 4);
    for (int32_t i = 0; i < candidates; ++i) {
        const std::size_t source = static_cast<std::size_t>(indices[i]) * 4;
        std::copy(dense.boxes.begin() + source, dense.boxes.begin() + source + 4,
                  candidate_boxes.begin() + static_cast<std::size_t>(i) * 4);
    }

    // 4. Class-aware duplicate removal.
    const std::vector<float> shifted = shift_by_class(candidate_boxes, labels);
    std::vector<int32_t> keep(static_cast<std::size_t>(candidates));
    std::vector<float> kept_scores(static_cast<std::size_t>(candidates));
    int32_t kept = 0;
    if (options_.soft_nms) {
        kept = lofop_soft_nms(shifted.data(), scores.data(), candidates, options_.nms_iou,
                              options_.soft_sigma, options_.score_threshold, /*method=*/0,
                              options_.max_detections, keep.data(), kept_scores.data());
    } else {
        kept = lofop_nms(shifted.data(), scores.data(), candidates, options_.nms_iou,
                         options_.max_detections, keep.data());
        for (int32_t i = 0; i < kept; ++i) {
            kept_scores[static_cast<std::size_t>(i)] = scores[static_cast<std::size_t>(keep[i])];
        }
    }
    if (kept <= 0) {
        return {};
    }

    // 5. Map surviving boxes back to the original image's coordinates.
    std::vector<float> final_boxes(static_cast<std::size_t>(kept) * 4);
    for (int32_t i = 0; i < kept; ++i) {
        const std::size_t source = static_cast<std::size_t>(keep[i]) * 4;
        std::copy(candidate_boxes.begin() + source, candidate_boxes.begin() + source + 4,
                  final_boxes.begin() + static_cast<std::size_t>(i) * 4);
    }
    std::vector<float> mapped(final_boxes.size());
    if (lofop_unletterbox_boxes(final_boxes.data(), kept, meta, image.width(), image.height(),
                                mapped.data()) != 0) {
        throw Error("Mapping boxes back to image coordinates failed");
    }

    std::vector<Detection> results;
    results.reserve(static_cast<std::size_t>(kept));
    for (int32_t i = 0; i < kept; ++i) {
        Detection detection;
        detection.x1 = mapped[static_cast<std::size_t>(i) * 4];
        detection.y1 = mapped[static_cast<std::size_t>(i) * 4 + 1];
        detection.x2 = mapped[static_cast<std::size_t>(i) * 4 + 2];
        detection.y2 = mapped[static_cast<std::size_t>(i) * 4 + 3];
        detection.confidence = kept_scores[static_cast<std::size_t>(i)];
        detection.label = static_cast<int>(labels[static_cast<std::size_t>(keep[i])]);
        detection.name = class_name(detection.label);
        results.push_back(std::move(detection));
    }
    return results;
}

}  // namespace lofop
