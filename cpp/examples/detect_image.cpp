// Minimal LOFOP C++ SDK example: load a model, detect, print.
//
// Build (from the repository root):
//   cmake -S cpp -B build -DLOFOP_WITH_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime
//   cmake --build build
//   ./build/detect_image model.onnx image.ppm
//
// Produce model.onnx from a trained checkpoint with:
//   lofop export --config lofop/configs/lofop-detect/s.yaml \
//                --checkpoint runs/train/best.pt -o model.onnx

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "lofop/lofop.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: detect_image <model.onnx> <image> [score_threshold]\n";
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string image_path = argv[2];

    lofop::Options options;
    if (argc > 3) {
        options.score_threshold = std::strtof(argv[3], nullptr);
    }
    // Names are optional; without them detections report "class_<index>".
    options.class_names = {};

    try {
        lofop::Detector detector(model_path, options);
        std::cout << "model " << model_path << " | input " << detector.input_size()
                  << "px | backend " << detector.backend() << '\n';

        const auto detections = detector.predict(image_path);
        std::cout << detections.size() << " detection(s) in " << image_path << '\n';
        std::cout << std::fixed << std::setprecision(1);
        for (const auto& hit : detections) {
            std::cout << "  " << hit.name << "  " << std::setprecision(3) << hit.confidence
                      << std::setprecision(1) << "  [" << hit.x1 << ", " << hit.y1 << ", "
                      << hit.x2 << ", " << hit.y2 << "]\n";
        }
    } catch (const lofop::Error& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        return 1;
    }
    return 0;
}
