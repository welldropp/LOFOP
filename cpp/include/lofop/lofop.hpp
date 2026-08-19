// LOFOP C++ inference SDK.
//
// Runs a LOFOP detector exported to ONNX from a C++ process, with no Python
// and no PyTorch anywhere in the deployed artifact:
//
//     #include <lofop/lofop.hpp>
//
//     lofop::Detector model("model.onnx");
//     for (const auto& hit : model.predict("image.ppm")) {
//         std::cout << hit.name << ' ' << hit.confidence << '\n';
//     }
//
// The pipeline is deliberately thin, because most of it already exists:
// preprocessing comes from lofop/csrc/preprocess.cpp, dense decoding and
// class-aware NMS from lofop/csrc/box_ops.cpp -- the same kernels the Python
// runtime calls through ctypes -- and graph execution is delegated to ONNX
// Runtime rather than reimplemented. Given one model.onnx, this SDK and
// lofop.runtime return the same detections.
//
// Execution sits behind the Engine interface, so ONNX Runtime is a
// compile-time option (LOFOP_WITH_ONNXRUNTIME): the library, its tests, and
// everything except the graph call build with nothing but a C++17 compiler.

#ifndef LOFOP_LOFOP_HPP
#define LOFOP_LOFOP_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace lofop {

// Version of the SDK, kept in step with the Python package.
constexpr const char* kVersion = "1.2.1";

// Thrown for every failure this SDK reports: missing files, unreadable
// images, malformed models, inference errors.
class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

// An 8-bit image in row-major HWC layout -- the same shape lofop.runtime.Image
// holds in Python, so both runtimes accept identical pixel buffers.
class Image {
public:
    Image() = default;
    Image(int width, int height, int channels, std::vector<uint8_t> pixels);

    // Wrap pixels you already hold (a decoded frame, a cv::Mat's data, a
    // camera buffer). No decoding, no format assumptions.
    static Image from_pixels(const uint8_t* pixels, int width, int height, int channels = 3);

    // Read an image file. Binary PPM (P6) and PGM (P5) are supported with no
    // dependencies; building with LOFOP_WITH_STB_IMAGE adds every format
    // stb_image handles (JPEG, PNG, BMP, ...). Throws Error otherwise.
    static Image load(const std::string& path);

    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    bool empty() const { return pixels_.empty(); }
    const std::vector<uint8_t>& pixels() const { return pixels_; }

private:
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
    std::vector<uint8_t> pixels_;
};

// One detected object, in the source image's own pixel coordinates.
struct Detection {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float confidence = 0.0f;
    int label = 0;
    std::string name;

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() > 0 && height() > 0 ? width() * height() : 0.0f; }
};

// Inference settings. Defaults match lofop.runtime.Detector so the two
// runtimes agree without either side restating them.
struct Options {
    float score_threshold = 0.25f;
    float nms_iou = 0.6f;
    int max_detections = 300;
    // Model input side length. 0 means "read it from the graph", which works
    // for fixed-shape exports; dynamic-shape models need it set.
    int image_size = 0;
    // Padding fill for letterboxing, in [0, 1].
    float pad_value = 0.447f;
    // Class-aware Soft-NMS score decay instead of greedy suppression.
    bool soft_nms = false;
    float soft_sigma = 0.5f;
    std::vector<std::string> class_names;
};

// The dense output of one forward pass: the exported graph's two tensors.
struct DenseOutput {
    std::vector<float> boxes;   // num_candidates * 4, xyxy in letterboxed pixels
    std::vector<float> scores;  // num_candidates * num_classes, in [0, 1]
    int num_candidates = 0;
    int num_classes = 0;
};

// Graph execution, abstracted so the backend is a choice rather than a
// dependency: ONNX Runtime today, TensorRT or OpenVINO by implementing this
// interface, and a deterministic stub in the test suite.
class Engine {
public:
    virtual ~Engine() = default;

    // Run one CHW float32 image of `input_size()` per side, values in [0, 1].
    virtual DenseOutput run(const float* nchw, std::size_t length) = 0;

    // Side length this engine expects.
    virtual int input_size() const = 0;

    // Human-readable backend name, e.g. "onnxruntime".
    virtual std::string backend() const = 0;
};

// Build an ONNX Runtime engine for a model file.
// Throws Error when the SDK was built without LOFOP_WITH_ONNXRUNTIME.
std::unique_ptr<Engine> make_onnxruntime_engine(const std::string& model_path, int image_size);

// True when this build can load .onnx files directly.
bool has_onnxruntime();

// Detect objects in images using an exported LOFOP model.
class Detector {
public:
    // Load a model by path. Requires an ONNX Runtime-enabled build.
    explicit Detector(const std::string& model_path, Options options = {});

    // Take an engine directly: custom backends, or a stub in tests.
    explicit Detector(std::unique_ptr<Engine> engine, Options options = {});

    // Detect objects; boxes come back in the image's own coordinates.
    std::vector<Detection> predict(const Image& image) const;

    // Convenience overload that loads the file first.
    std::vector<Detection> predict(const std::string& image_path) const;

    // Name for a class index, falling back to "class_<index>".
    std::string class_name(int label) const;

    int input_size() const { return engine_->input_size(); }
    std::string backend() const { return engine_->backend(); }
    const Options& options() const { return options_; }
    Options& options() { return options_; }

private:
    std::unique_ptr<Engine> engine_;
    Options options_;
};

}  // namespace lofop

#endif  // LOFOP_LOFOP_HPP
