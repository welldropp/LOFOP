/* LOFOP C ABI: the language-neutral inference boundary.
 *
 * The C++ API is the natural one for C++ callers; this header exists for
 * everyone else. A flat C ABI is stable across compilers and standard-library
 * versions and is the boundary Go (cgo), Rust (FFI), C# (P/Invoke), and Java
 * (JNI/FFM) can all bind to directly -- the same reason TensorFlow ships a C
 * API next to its C++ one.
 *
 * Ownership: lofop_detector_create returns a handle the caller frees with
 * lofop_detector_free. Detection arrays returned by the predict calls are
 * owned by the handle and stay valid until the next predict call on that
 * handle or until it is freed.
 *
 * Errors: functions return a negative status and leave a message retrievable
 * with lofop_last_error(); no exception ever crosses this boundary.
 */

#ifndef LOFOP_LOFOP_C_H
#define LOFOP_LOFOP_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LOFOP_OK 0
#define LOFOP_ERROR (-1)

/* One detection, in the source image's own pixel coordinates. */
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
    int32_t label;
} lofop_detection;

/* Inference settings; zero-initialise then override what you need, or call
 * lofop_options_default to get the same defaults the C++ and Python APIs use. */
typedef struct {
    float score_threshold;
    float nms_iou;
    int32_t max_detections;
    int32_t image_size;  /* 0 = read from the graph */
    float pad_value;
    int32_t soft_nms;    /* 0 = greedy NMS, 1 = Soft-NMS */
    float soft_sigma;
} lofop_options;

typedef struct lofop_detector lofop_detector;

/* Fill `options` with LOFOP's documented defaults. */
void lofop_options_default(lofop_options* options);

/* Load a model. Returns NULL on failure; see lofop_last_error(). */
lofop_detector* lofop_detector_create(const char* model_path, const lofop_options* options);

/* Release a detector created by lofop_detector_create. Safe on NULL. */
void lofop_detector_free(lofop_detector* detector);

/* Detect objects in an image file.
 * On success returns the detection count (>= 0) and points *out at an array
 * owned by `detector`. Returns LOFOP_ERROR on failure. */
int32_t lofop_detector_predict_path(lofop_detector* detector, const char* image_path,
                                    const lofop_detection** out);

/* Detect objects in a pixel buffer you already hold (row-major HWC, 8-bit,
 * 1 or 3 channels). Same return contract as lofop_detector_predict_path. */
int32_t lofop_detector_predict_pixels(lofop_detector* detector, const uint8_t* pixels,
                                      int32_t width, int32_t height, int32_t channels,
                                      const lofop_detection** out);

/* Model input side length this detector runs at. */
int32_t lofop_detector_input_size(const lofop_detector* detector);

/* Message describing the most recent failure on this thread; never NULL. */
const char* lofop_last_error(void);

/* SDK version string, e.g. "1.2.1". */
const char* lofop_version(void);

/* Non-zero when this build can load .onnx files by path. */
int32_t lofop_has_onnxruntime(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* LOFOP_LOFOP_C_H */
