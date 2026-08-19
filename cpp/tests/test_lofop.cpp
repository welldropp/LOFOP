// Tests for the LOFOP C++ SDK.
//
// No test framework: LOFOP's C++ builds with any C++17 compiler and no
// dependencies, and its test suite keeps that property so `cmake --build .`
// followed by `ctest` works on a bare machine. The harness below is twenty
// lines and reports the same things a framework would.
//
// Graph execution is supplied by StubEngine, which returns fixed dense
// outputs. That is not a shortcut around testing the real thing: it isolates
// everything LOFOP actually implements -- preprocessing, decoding,
// suppression, coordinate mapping -- from ONNX Runtime, which has its own test
// suite. The numbers asserted here match tests/runtime/test_runtime.py, so
// the two language runtimes are pinned to the same expected output.

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "lofop/lofop.hpp"
#include "lofop/lofop_c.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& what) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::cerr << "  FAIL: " << what << '\n';
    }
}

void check_close(float actual, float expected, const std::string& what, float tolerance = 1e-3f) {
    ++g_checks;
    if (std::fabs(actual - expected) > tolerance) {
        ++g_failures;
        std::cerr << "  FAIL: " << what << " (expected " << expected << ", got " << actual
                  << ")\n";
    }
}

// Returns the same dense outputs the Python parity test feeds its stub model:
// two overlapping class-0 boxes (the second must be suppressed), one class-1
// box, and one candidate below the score threshold.
class StubEngine : public lofop::Engine {
public:
    explicit StubEngine(int size) : size_(size) {}

    lofop::DenseOutput run(const float* nchw, std::size_t length) override {
        last_input_.assign(nchw, nchw + length);
        lofop::DenseOutput dense;
        dense.num_candidates = 4;
        dense.num_classes = 3;
        dense.boxes = {
            10.0f, 10.0f, 30.0f, 30.0f,
            11.0f, 11.0f, 31.0f, 31.0f,
            40.0f, 40.0f, 60.0f, 60.0f,
            0.0f, 0.0f, 5.0f, 5.0f,
        };
        dense.scores = {
            0.9f, 0.0f, 0.0f,
            0.8f, 0.0f, 0.0f,
            0.0f, 0.7f, 0.0f,
            0.0f, 0.0f, 0.1f,
        };
        return dense;
    }

    int input_size() const override { return size_; }
    std::string backend() const override { return "stub"; }
    const std::vector<float>& last_input() const { return last_input_; }

private:
    int size_;
    std::vector<float> last_input_;
};

lofop::Options stub_options() {
    lofop::Options options;
    options.class_names = {"cat", "dog", "bird"};
    return options;
}

// A 128x96 image: aspect 4:3, so letterboxing to 64 gives scale 0.5 and 8
// pixels of padding top and bottom -- the geometry the assertions below use.
lofop::Image make_test_image() {
    std::vector<uint8_t> pixels(static_cast<std::size_t>(128) * 96 * 3);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<uint8_t>(i % 251);
    }
    return lofop::Image(128, 96, 3, std::move(pixels));
}

void test_image_construction() {
    std::cout << "test_image_construction\n";
    const lofop::Image image = make_test_image();
    check(image.width() == 128 && image.height() == 96, "image reports its dimensions");
    check(image.channels() == 3, "image reports its channel count");
    check(!image.empty(), "constructed image is not empty");

    bool threw = false;
    try {
        lofop::Image bad(4, 4, 3, std::vector<uint8_t>(10));
    } catch (const lofop::Error&) {
        threw = true;
    }
    check(threw, "mismatched pixel buffer is rejected");

    threw = false;
    try {
        lofop::Image bad(0, 4, 3, std::vector<uint8_t>());
    } catch (const lofop::Error&) {
        threw = true;
    }
    check(threw, "non-positive dimensions are rejected");
}

void test_predict_pipeline() {
    std::cout << "test_predict_pipeline\n";
    auto engine = std::unique_ptr<StubEngine>(new StubEngine(64));
    const StubEngine* observer = engine.get();
    lofop::Detector detector(std::move(engine), stub_options());

    const auto results = detector.predict(make_test_image());
    check(results.size() == 2, "NMS suppresses the duplicate and the threshold drops the weak one");
    if (results.size() != 2) {
        return;
    }

    check(results[0].name == "cat", "highest-scoring detection keeps its class name");
    check_close(results[0].confidence, 0.9f, "top detection confidence");
    // Letterbox geometry: scale 0.5, pad_top 8. x_src = x/0.5, y_src = (y-8)/0.5.
    check_close(results[0].x1, 20.0f, "box maps back to original x1");
    check_close(results[0].y1, 4.0f, "box maps back to original y1");
    check_close(results[0].x2, 60.0f, "box maps back to original x2");
    check_close(results[0].y2, 44.0f, "box maps back to original y2");

    check(results[1].name == "dog", "second detection keeps its class name");
    check_close(results[1].confidence, 0.7f, "second detection confidence");
    check_close(results[1].x1, 80.0f, "second box x1");
    check_close(results[1].y2, 96.0f, "second box is clamped to the image height");

    // The preprocessor must have produced a full CHW tensor in [0, 1].
    check(observer->last_input().size() == static_cast<std::size_t>(64) * 64 * 3,
          "engine received a complete CHW tensor");
    bool in_range = true;
    for (float value : observer->last_input()) {
        if (value < 0.0f || value > 1.0f) {
            in_range = false;
            break;
        }
    }
    check(in_range, "preprocessed tensor is normalised to [0, 1]");
}

void test_threshold_and_names() {
    std::cout << "test_threshold_and_names\n";
    lofop::Options options = stub_options();
    options.score_threshold = 0.75f;
    lofop::Detector detector(
        std::unique_ptr<lofop::Engine>(new StubEngine(64)), options);
    const auto results = detector.predict(make_test_image());
    check(results.size() == 1, "raising the threshold drops the 0.7 detection");

    lofop::Options unnamed;
    lofop::Detector plain(
        std::unique_ptr<lofop::Engine>(new StubEngine(64)), unnamed);
    const auto fallback = plain.predict(make_test_image());
    check(!fallback.empty() && fallback[0].name == "class_0",
          "missing class names fall back to class_<index>");
}

void test_soft_nms_keeps_both() {
    std::cout << "test_soft_nms_keeps_both\n";
    lofop::Options options = stub_options();
    options.soft_nms = true;
    lofop::Detector detector(
        std::unique_ptr<lofop::Engine>(new StubEngine(64)), options);
    const auto results = detector.predict(make_test_image());
    // Soft-NMS decays the duplicate's score rather than deleting it, so it
    // survives where greedy NMS removed it.
    check(results.size() >= 2, "soft-NMS retains decayed duplicates");
}

void test_error_paths() {
    std::cout << "test_error_paths\n";
    bool threw = false;
    try {
        lofop::Detector detector(std::unique_ptr<lofop::Engine>(), lofop::Options{});
    } catch (const lofop::Error&) {
        threw = true;
    }
    check(threw, "a null engine is rejected");

    threw = false;
    try {
        lofop::Detector detector(
            std::unique_ptr<lofop::Engine>(new StubEngine(64)), stub_options());
        detector.predict(lofop::Image());
    } catch (const lofop::Error&) {
        threw = true;
    }
    check(threw, "an empty image is rejected");

    if (!lofop::has_onnxruntime()) {
        threw = false;
        std::string message;
        try {
            lofop::Detector detector("missing.onnx");
        } catch (const lofop::Error& exc) {
            threw = true;
            message = exc.what();
        }
        check(threw, "loading a model without an ONNX Runtime build fails");
        check(message.find("LOFOP_WITH_ONNXRUNTIME") != std::string::npos,
              "the error explains how to enable the backend");
    }
}

void test_c_abi() {
    std::cout << "test_c_abi\n";
    check(std::string(lofop_version()) == lofop::kVersion, "C ABI reports the SDK version");

    lofop_options options;
    lofop_options_default(&options);
    check_close(options.score_threshold, 0.25f, "C ABI defaults match the C++ defaults");
    check(options.max_detections == 300, "C ABI default detection cap");

    // Failure path: a model that cannot be loaded must return NULL and leave a
    // message rather than propagating an exception across the boundary.
    lofop_detector* handle = lofop_detector_create("does-not-exist.onnx", &options);
    check(handle == nullptr, "creating a detector for a missing model returns NULL");
    check(std::string(lofop_last_error()).size() > 0, "the failure leaves an error message");

    check(lofop_detector_predict_path(nullptr, "x.ppm", nullptr) == LOFOP_ERROR,
          "null handle is reported, not dereferenced");
    check(lofop_detector_input_size(nullptr) == LOFOP_ERROR, "null handle input size is an error");
    lofop_detector_free(nullptr);  // must be safe
}

}  // namespace

int main() {
    std::cout << "LOFOP C++ SDK tests (onnxruntime backend: "
              << (lofop::has_onnxruntime() ? "on" : "off") << ")\n";
    test_image_construction();
    test_predict_pipeline();
    test_threshold_and_names();
    test_soft_nms_keeps_both();
    test_error_paths();
    test_c_abi();
    std::cout << (g_failures == 0 ? "PASSED " : "FAILED ") << (g_checks - g_failures) << '/'
              << g_checks << " checks\n";
    return g_failures == 0 ? 0 : 1;
}
