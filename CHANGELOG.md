# Changelog

All notable changes to LOFOP are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.2] - 2026-09-15

### Added
- NMS-free detection: `lofop-detect-{q-n,q-s}` variants (`model/LofopQuery`
  + `head/SlateHead`). A fixed slate of query slots is matched one-to-one
  with ground truth during training (`PivotMatcher`), so inference emits at
  most one box per object and runs **no suppression pass at all** -- no IoU
  loop, no score decay, no peak test.
- Tracking: `lofop.tracking` -- `MotionEstimator` (constant-velocity Kalman
  filter over `[cx, cy, w, h]` and velocities, written from the estimator
  equations), IoU association built on LOFOP's native C++/CUDA kernel, and
  `LofopTracker`, a confidence-tiered tracker with track lifecycle
  management. Exposed through `Detector.track()`.
- Ops: `lofop.ops.assignment` -- exact `optimal_assignment` (shortest
  augmenting path) and `greedy_assignment`, in pure Python with no scipy.
- Backbone: `backbone/SwiftNet`, an inverted-residual edge backbone with
  squeeze-excite on the deepest stages and ReLU6 for clean INT8
  quantisation. Variants `mb-n`, `mb-s`, and `mbq-n`.
- Ecosystem: `Detections.to_supervision()` and
  `lofop.tracking.from_supervision()` bridge to the supervision library
  through its public API; no supervision code is vendored.
- Data: `Detections.tracker_ids` carries persistent identities.
- Packaging: `lofop[tracking]` and `lofop[supervision]` extras. numpy,
  scipy and supervision are never imported at module scope, so a stock
  `pip install lofop` is unaffected; missing extras raise `LofopError`
  naming the exact install command.
- Project: a `NOTICE` file recording that LOFOP contains no third-party
  source, and `tests/test_dependency_hygiene.py`, which fails the build on
  a module-scope optional import, an undeclared extra, or a copyleft
  marker anywhere in the package.

### Changed
- ONNX/TensorRT export raises a clear error for the query variants; dense
  export is unchanged.
### Added
- C++ inference SDK (`cpp/`): `lofop::Detector`, `lofop::Image`,
  `lofop::Detection`, and a pluggable `lofop::Engine` (ONNX Runtime backend
  included), plus a flat C ABI (`lofop/lofop_c.h`) for Go/Rust/C#/Java
  bindings. Builds with CMake and any C++17 compiler; ONNX Runtime is a
  compile-time option, so the library and its tests build without it.
- Torch-free Python inference runtime (`lofop.runtime`): `Detector`,
  `Detection`, and `Image` mirroring the C++ types, running an exported
  `model.onnx` through ONNX Runtime with no PyTorch import.
- Shared image preprocessing (`lofop/csrc/preprocess.cpp`,
  `lofop.ops.letterbox` / `unletterbox_boxes`): aspect-preserving resize with
  padding and its exact inverse, defined once in C++ and bound by both
  runtimes, with a pure Python reference for compiler-free installs. Parity
  between kernel and reference is asserted to float32 epsilon in CI.
- CI: the C++ SDK is built and tested on Linux, Windows, and macOS.

### Changed
- The native ops library now compiles from multiple translation units
  (`box_ops.cpp` + `preprocess.cpp`); previously built libraries still load,
  as new symbols are bound behind `hasattr` guards.

## [1.2.1] - 2026-07-31

### Added
- Instance segmentation: `lofop-detect-{n,s,ex}-seg` variants
  (`model/LofopSegment` + the prototype-based `head/StencilHead`). Trains
  from COCO polygon annotations (or plain boxes as a fallback) through the
  same SDK/CLI, and `predict` attaches per-detection masks.
- Pose estimation: `lofop-detect-{n,s,ex}-pose` variants (`model/LofopPose`
  + the offset-based `head/VertexHead`, `num_keypoints` configurable).
  Trains from COCO keypoint annotations; `predict` attaches per-detection
  `(x, y, visibility)` keypoints.
- Data: `BoxAnnotation` optionally carries `segmentation` polygons and
  `keypoints`; the COCO adapter round-trips both. `DetectionTorchDataset`
  gains `include_masks` / `include_keypoints` targets.
- Ops: `soft_nms` (gaussian/linear score decay, C++ kernel + verified
  Python reference) and `decode_dense` (native dense best-class decode,
  zero-copy for numpy/tensor inputs); `postprocess_dense(soft=True)`
  enables Soft-NMS on the torch-free deployment path.
- Ops: optional CUDA tier (`build_native(cuda=True)`, needs nvcc) for
  pairwise IoU and dense decode on NVIDIA GPUs; `backend()` now reports
  `cuda`/`native`/`python`, always falling back cleanly.
- Inference: selectable duplicate-removal via `nms_mode` on every model and
  `Detector` -- `"greedy"` (default, unchanged), `"soft"` (Soft-NMS score
  decay), or `"free"` (NMS-free 3x3 peak selection: pure tensor math, no
  suppression loop).

### Changed
- ONNX export raises a clear error for segmentation/pose models (their
  export lands in a later release); detection export is unchanged.

## [1.1.3] - 2026-07-16

### Added
- MLOps: `lofop.mlops` local experiment tracking -- `with track(root):`
  records every training run (settings, environment, per-epoch history,
  best/final metrics, checkpoints) as plain JSON via the event bus, plus
  `lofop runs list/show/compare` CLI. Torch-free registry; no trainer
  changes; original implementation.
- Training: original strong-augmentation recipe (2x2 mosaic + color
  jitter, tensor-native, no external augmentation library) behind an
  opt-in `strong_augment` flag on `DetectionTorchDataset`,
  `Detector.train`, and the training config (`data.strong_augment`).
  Defaults unchanged: existing runs and benchmarks are unaffected.

## [1.1.2] - 2026-07-15

### Added
- CLI: `lofop predict` (image inference), `lofop evaluate` (dataset metrics),
  and `lofop doctor` (environment and backend diagnostics).
- Evaluation: F1 score, per-class precision/recall, and a class confusion
  matrix, exposed on `DetectionMetrics` (all backward-compatible defaults).
- Data: torch-free visualization (`draw_boxes`, `render_sample`,
  `visualize_dataset`) and the `lofop dataset show` command.
- Training: config-driven learning-rate schedulers (`warmup_cosine`,
  `warmup_linear`, `constant`, `step`), opt-in early stopping
  (`patience`/`min_delta`), and an optional `TensorBoardHook`.
- Benchmarking: CSV/JSON export (`render_csv`, `render_json`, `write_reports`)
  and a peak-memory column; `lofop benchmark --results-dir` and a consolidated
  `benchmarks/run_suite.py` runner.
- Deployment: dynamic-shape ONNX export (`export_onnx(..., dynamic=True)` /
  `lofop export --dynamic`) with symbolic batch/height/width axes, verified at
  two resolutions.
- Packaging: `tensorboard` optional-dependency extra; a version-sync check
  (`scripts/check_version_sync.py`) wired into CI.
- Testing: `pytest-cov` with a CI-enforced coverage floor (~90% measured).
- Project: CONTRIBUTING, SECURITY, CODE_OF_CONDUCT, issue/PR templates.

## [0.1.0]

### Added
- Core engine: registry/hub, YAML config (inheritance + interpolation), event
  bus, plugin manager, structured logging, typed exceptions.
- LOFOP-Detect anchor-free detector with `n`/`s`/`ex` variants.
- Native C++ box ops (IoU/NMS) with a pure-Python fallback, cross-platform.
- Data subsystem: COCO/YOLO/VOC adapters, conversion, validation, statistics.
- Training: AMP, EMA, checkpoints/resume, COCO-protocol evaluator, DDP path.
- Deployment: verified ONNX export, TensorRT (FP16/INT8), torch-free
  post-processing.
- Python SDK (`Detector`) and the `lofop` CLI.
- Packaging for PyPI (wheel/sdist) and AUR; CI and release workflows.

[Unreleased]: https://github.com/tedo001/LOFOP/compare/v1.2.2...HEAD
[1.2.2]: https://github.com/tedo001/LOFOP/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/tedo001/LOFOP/compare/v1.1.3...v1.2.1
[1.1.3]: https://github.com/tedo001/LOFOP/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/tedo001/LOFOP/compare/v0.1.0...v1.1.2
[0.1.0]: https://github.com/tedo001/LOFOP/releases/tag/v0.1.0
