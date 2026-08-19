# LOFOP C++ SDK

Run a LOFOP detector from a C++ process, with no Python and no PyTorch in the
deployed artifact.

```cpp
#include <lofop/lofop.hpp>

lofop::Detector model("model.onnx");
for (const auto& hit : model.predict("image.ppm")) {
    std::cout << hit.name << ' ' << hit.confidence << '\n';
}
```

The Python equivalent is the same three types with the same names:

```python
from lofop.runtime import Detector

model = Detector("model.onnx")
for hit in model.predict("image.jpg"):
    print(hit.name, hit.confidence, hit.box)
```

Both consume the same `model.onnx` and return the same detections.

## How the two runtimes stay identical

They are not two implementations. They are two thin bindings over one:

| Stage | Implementation | Used by |
| --- | --- | --- |
| Preprocess (letterbox) | `lofop/csrc/preprocess.cpp` | both |
| Execute the graph | ONNX Runtime | both |
| Dense decode | `lofop_decode_dense` in `lofop/csrc/box_ops.cpp` | both |
| NMS / Soft-NMS | `lofop_nms`, `lofop_soft_nms` in `box_ops.cpp` | both |
| Map boxes back | `lofop_unletterbox_boxes` in `preprocess.cpp` | both |

Python reaches those kernels through `ctypes`; the C++ SDK compiles the same
source files directly. Nothing numerical is written twice, so the runtimes
cannot drift.

The one exception is preprocessing, which additionally has a pure Python
reference so LOFOP still runs where no compiler exists. That reference is the
specification, and `tests/ops/test_preprocess.py` asserts the kernel matches it
to float32 epsilon across upscales, downscales, extreme aspect ratios, and
grayscale input. If they ever disagree, CI fails.

## Building

Requires CMake 3.16+ and any C++17 compiler.

```bash
cmake -S cpp -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

That builds and tests the whole pipeline **without ONNX Runtime**. Execution
sits behind the `lofop::Engine` interface, so only loading a `.onnx` by path
needs the backend; everything LOFOP itself implements is exercised through a
stub engine in the test suite.

To run real models, point the build at an
[ONNX Runtime release](https://github.com/microsoft/onnxruntime/releases):

```bash
cmake -S cpp -B build -DLOFOP_WITH_ONNXRUNTIME=ON -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build
./build/detect_image model.onnx image.ppm
```

### Build options

| Option | Default | Effect |
| --- | --- | --- |
| `LOFOP_WITH_ONNXRUNTIME` | `OFF` | Enables loading `.onnx` files by path |
| `LOFOP_WITH_STB_IMAGE` | `OFF` | Adds JPEG/PNG decoding via a vendored `stb_image.h` |
| `LOFOP_BUILD_TESTS` | `ON` | Builds the SDK test suite |
| `LOFOP_BUILD_EXAMPLES` | `ON` | Builds `detect_image` |
| `BUILD_SHARED_LIBS` | `OFF` | Shared instead of static library |

## Producing a model

The SDK consumes the same artifact the Python deployment path does:

```bash
lofop export --config lofop/configs/lofop-detect/s.yaml \
             --checkpoint runs/train/best.pt -o model.onnx
```

The exported graph includes box decoding and is verified numerically against
the torch model at export time, so the handoff between languages is a tested
contract rather than an assumption.

Segmentation and pose variants are not exportable to ONNX yet, so the SDK
covers detection models.

## API

### `lofop::Image`

An 8-bit image in row-major HWC layout.

```cpp
// Pixels you already hold -- a decoded frame, a cv::Mat, a camera buffer.
auto image = lofop::Image::from_pixels(mat.data, mat.cols, mat.rows, 3);

// Or read a file. Binary PPM/PGM needs no dependencies; build with
// LOFOP_WITH_STB_IMAGE for JPEG, PNG, and the rest.
auto image = lofop::Image::load("frame.ppm");
```

Decoding deliberately stays out of LOFOP's scope: production callers almost
always already have pixels, and forcing an image-codec dependency on them would
be a cost with no benefit.

### `lofop::Options`

```cpp
lofop::Options options;
options.score_threshold = 0.5f;      // default 0.25
options.nms_iou         = 0.6f;
options.max_detections  = 300;
options.image_size      = 0;         // 0 = read it from the graph
options.pad_value       = 0.447f;    // letterbox fill
options.soft_nms        = false;     // true = Soft-NMS score decay
options.class_names     = {"cat", "dog"};
```

The defaults match `lofop.runtime.Detector`, so neither side restates them.

### `lofop::Detection`

`x1`, `y1`, `x2`, `y2` in the source image's own pixels, plus `confidence`,
`label`, and `name`, with `width()`, `height()`, and `area()` helpers.

### `lofop::Engine`

Graph execution is an interface, which is what makes ONNX Runtime a choice
rather than a dependency. Implement it to add a backend — TensorRT, OpenVINO,
or a fixture in your own tests:

```cpp
class MyEngine : public lofop::Engine {
    lofop::DenseOutput run(const float* nchw, std::size_t length) override;
    int input_size() const override;
    std::string backend() const override;
};

lofop::Detector detector(std::make_unique<MyEngine>(), options);
```

## The C ABI

`include/lofop/lofop_c.h` exposes the same pipeline through a flat C ABI, for
Go (cgo), Rust (FFI), C# (P/Invoke), and Java (JNI/FFM). No exception ever
crosses the boundary: calls return a status and leave a message in
`lofop_last_error()`.

```c
#include <lofop/lofop_c.h>

lofop_options options;
lofop_options_default(&options);

lofop_detector* det = lofop_detector_create("model.onnx", &options);
if (det == NULL) { fprintf(stderr, "%s\n", lofop_last_error()); return 1; }

const lofop_detection* hits = NULL;
int32_t n = lofop_detector_predict_path(det, "image.ppm", &hits);
for (int32_t i = 0; i < n; ++i) {
    printf("%d %.3f [%.1f %.1f %.1f %.1f]\n", hits[i].label, hits[i].confidence,
           hits[i].x1, hits[i].y1, hits[i].x2, hits[i].y2);
}
lofop_detector_free(det);
```

A C ABI rather than pybind11 is a deliberate choice, and the same one
TensorFlow made for `libtensorflow`: it is stable across compilers and Python
versions, needs no build step at install time, and is the only boundary every
other language can bind to.

## What this does not change

Adding the C++ SDK changed nothing about training. `lofop.Detector` — the
PyTorch build/train/export API — is untouched, and `lofop.runtime.Detector` is
a separate class in a separate module reached by a separate import:

```python
from lofop import Detector            # PyTorch: build, train, export
from lofop.runtime import Detector    # ONNX: deploy, no torch
```

The two never shadow each other, and `lofop.runtime` imports no torch at all —
a property the test suite enforces in a clean interpreter, since that is what
lets a deployment image omit PyTorch entirely.

The `cpp/` directory is likewise self-contained: it can be built, shipped, or
deleted without affecting the Python package.
