// ONNX Runtime execution backend for the LOFOP C++ SDK.
//
// Execution is delegated, not reimplemented. ONNX Runtime is a mature,
// heavily optimised graph executor maintained by people who do nothing else;
// writing convolution kernels to compete with it would cost years and lose.
// This file is therefore thin by design -- it adapts LOFOP's Engine interface
// to an Ort::Session and nothing more.
//
// The whole backend is behind LOFOP_WITH_ONNXRUNTIME. Without it the SDK still
// builds, links, and passes its tests with any C++17 compiler; only the
// "load a .onnx by path" constructor is unavailable, and it says so clearly.

#include <string>
#include <vector>

#include "lofop/lofop.hpp"

#ifdef LOFOP_WITH_ONNXRUNTIME

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstddef>

namespace lofop {
namespace {

class OnnxRuntimeEngine : public Engine {
public:
    OnnxRuntimeEngine(const std::string& model_path, int image_size)
        : env_(ORT_LOGGING_LEVEL_WARNING, "lofop"), session_(nullptr) {
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        try {
#ifdef _WIN32
            const std::wstring wide(model_path.begin(), model_path.end());
            session_ = Ort::Session(env_, wide.c_str(), session_options);
#else
            session_ = Ort::Session(env_, model_path.c_str(), session_options);
#endif
        } catch (const Ort::Exception& exc) {
            throw Error("Cannot load ONNX model '" + model_path + "': " + exc.what());
        }
        if (session_.GetOutputCount() < 2) {
            throw Error("Model does not expose LOFOP's dense (boxes, scores) outputs: " +
                        model_path);
        }

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_.GetInputNameAllocated(0, allocator).get();
        for (std::size_t i = 0; i < 2; ++i) {
            output_names_.push_back(session_.GetOutputNameAllocated(i, allocator).get());
        }
        input_size_ = image_size > 0 ? image_size : infer_input_size();
    }

    DenseOutput run(const float* nchw, std::size_t length) override {
        const std::array<int64_t, 4> shape{1, 3, input_size_, input_size_};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory, const_cast<float*>(nchw), length, shape.data(), shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_names_[0].c_str(), output_names_[1].c_str()};
        std::vector<Ort::Value> outputs;
        try {
            outputs = session_.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                   output_names, 2);
        } catch (const Ort::Exception& exc) {
            throw Error(std::string("Inference failed: ") + exc.what());
        }

        const auto box_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        const auto score_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        if (box_shape.empty() || score_shape.empty()) {
            throw Error("Model returned an output with no shape");
        }

        DenseOutput dense;
        dense.num_candidates = static_cast<int>(box_shape[box_shape.size() - 2]);
        dense.num_classes = static_cast<int>(score_shape[score_shape.size() - 1]);
        const std::size_t box_count = static_cast<std::size_t>(dense.num_candidates) * 4;
        const std::size_t score_count =
            static_cast<std::size_t>(dense.num_candidates) * dense.num_classes;
        const float* box_data = outputs[0].GetTensorData<float>();
        const float* score_data = outputs[1].GetTensorData<float>();
        dense.boxes.assign(box_data, box_data + box_count);
        dense.scores.assign(score_data, score_data + score_count);
        return dense;
    }

    int input_size() const override { return input_size_; }

    std::string backend() const override { return "onnxruntime"; }

private:
    // Read the side length from a fixed-shape graph; dynamic exports carry a
    // symbolic (negative) dimension, where 640 is LOFOP's documented default.
    int infer_input_size() const {
        const auto shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() == 4 && shape[2] > 0) {
            return static_cast<int>(shape[2]);
        }
        return 640;
    }

    Ort::Env env_;
    Ort::Session session_;
    std::string input_name_;
    std::vector<std::string> output_names_;
    int input_size_ = 640;
};

}  // namespace

std::unique_ptr<Engine> make_onnxruntime_engine(const std::string& model_path, int image_size) {
    return std::unique_ptr<Engine>(new OnnxRuntimeEngine(model_path, image_size));
}

bool has_onnxruntime() { return true; }

}  // namespace lofop

#else  // LOFOP_WITH_ONNXRUNTIME

namespace lofop {

std::unique_ptr<Engine> make_onnxruntime_engine(const std::string& model_path, int) {
    throw Error(
        "This LOFOP build has no ONNX Runtime backend, so '" + model_path +
        "' cannot be loaded by path. Rebuild with -DLOFOP_WITH_ONNXRUNTIME=ON, "
        "or construct Detector with your own Engine implementation.");
}

bool has_onnxruntime() { return false; }

}  // namespace lofop

#endif  // LOFOP_WITH_ONNXRUNTIME
