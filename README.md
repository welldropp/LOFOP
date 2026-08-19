# LOFOP

[![ci](https://github.com/tedo001/LOFOP/actions/workflows/ci.yml/badge.svg)](https://github.com/tedo001/LOFOP/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/lofop.svg)](https://pypi.org/project/lofop/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**LOFOP** is a modular, enterprise-grade computer vision framework built on PyTorch, with its own
original detector: **LOFOP-Detect**. It is an independent design and implementation that follows
modern computer-vision engineering practices while remaining self-contained.

> **Status:** published on [PyPI](https://pypi.org/project/lofop/) (`pip install lofop`).
> Core engine, data subsystem (with visualization), three-tier native ops (Python /
> C++ / optional CUDA), the LOFOP-Detect family (detection + segmentation + pose
> variants), training engine (schedulers, early stopping, strong augmentation), full
> CLI, Python SDK, and verified ONNX (fixed + dynamic shapes) / TensorRT export.
> 316 tests passing with a coverage-gated CI. See
> [`docs/architecture.md`](docs/architecture.md) for the subsystem map and
> [`CHANGELOG.md`](CHANGELOG.md) for release history.

📖 **New here? Read the [Operator's Manual](MANUAL.md)** — a complete, step-by-step guide to
installing, training, exporting, deploying, and troubleshooting LOFOP.

## What works today

- **Datasets** — COCO, YOLO, and VOC support through one canonical model: any-to-any conversion,
  validation (degenerate/out-of-bounds boxes, missing files, dangling categories), statistics,
  and torch-free visualization (`lofop dataset show` / `lofop.data.draw_boxes`).
- **LOFOP-Detect** — an original anchor-free detector (RidgeNet backbone, DeltaFusion neck with
  attention only on the cheap stride-32 level, ApexHead with an IoU-quality branch, dynamic top-k
  label assignment). Variants are pure config: `n` = 1.3M params, `s` = 3.8M, `ex` = 20.1M.
  Design + trade-offs: [`docs/lofop-detect.md`](docs/lofop-detect.md).
- **Instance segmentation** — `lofop-detect-{n,s,ex}-seg`: a prototype-based mask branch
  (StencilHead) on the same detector; trains from COCO polygons through the same SDK/CLI and
  attaches per-detection masks at inference.
- **Pose estimation** — `lofop-detect-{n,s,ex}-pose`: an offset-based keypoint branch
  (VertexHead, `num_keypoints` configurable); trains from COCO keypoints and attaches
  per-detection `(x, y, visibility)` skeletons.
- **Python SDK** — `from lofop import Detector`: build, train, predict (boxes in original image
  coordinates), evaluate, and export through one documented class. Full reference:
  [`docs/sdk.md`](docs/sdk.md).
- **Training** — AMP, EMA weights, config-driven LR schedulers (`warmup_cosine`,
  `warmup_linear`, `constant`, `step`), early stopping, an optional TensorBoard hook,
  gradient clipping, atomic checkpointing with resume, an opt-in strong-augmentation recipe
  (2x2 mosaic + color jitter, original tensor-native ops), and a COCO-protocol evaluator
  (mAP@50, mAP@50:95, precision, recall, F1, per-class precision/recall, confusion matrix).
- **Three-tier native ops** — IoU, class-aware NMS, Soft-NMS, and dense decode with a
  verified-identical Python fallback (a compiler is never required), a C++ fast path (20-200x;
  g++/clang on Linux/macOS, MinGW/clang/MSVC on Windows), and an optional CUDA tier
  (`build_native(cuda=True)`, needs nvcc) for the parallel ops on NVIDIA GPUs.
  `lofop.ops.backend()` reports the active tier: `cuda` > `native` > `python`.
- **Benchmarking** — `lofop benchmark` renders the standard metric table (mAP, FPS, params,
  FLOPs, model size) with optional CSV/JSON output (`--results-dir`), and never prints a
  number that was not actually measured.
- **Experiment tracking (MLOps)** — `with lofop.mlops.track("runs/registry"):` records every
  training run (settings, environment, per-epoch history, best/final metrics) as plain JSON
  through the event bus; inspect with `lofop runs list / show / compare`. Torch-free.
- **ONNX + TensorRT export** — `lofop export` writes a numerically verified ONNX graph (network +
  box decoding; `--dynamic` for variable input sizes, verified at two resolutions), or a
  TensorRT engine (`--format tensorrt --fp16`) via that same ONNX;
  `postprocess_dense` finishes inference torch-free with the C++ NMS, so serving hosts need only a
  runtime + the LOFOP core. Details: [`docs/deploy.md`](docs/deploy.md).
- **C++ SDK + torch-free Python runtime** — the same detector runs from either language on one
  exported model: `lofop::Detector model("model.onnx")` in C++ (no Python in the artifact) and
  `from lofop.runtime import Detector` in Python (no torch). Both bind the *same* kernels
  (letterbox, dense decode, class-aware NMS), so they cannot drift; a C ABI
  (`lofop/lofop_c.h`) carries the same pipeline to Go/Rust/C#/Java. Details:
  [`docs/cpp-sdk.md`](docs/cpp-sdk.md).
- **Deployment scaffolding** — CPU / CUDA / ONNX Runtime Docker images ([`docker/`](docker/README.md)).

## Installation

```bash
pip install lofop            # from PyPI
yay -S lofop                 # Arch Linux (AUR)
```

Optional feature sets (extras):

```bash
pip install "lofop[models]"      # + PyTorch, for lofop.models / lofop.training
pip install "lofop[deploy]"      # + onnx, onnxruntime, for ONNX export
pip install "lofop[tensorboard]" # + tensorboard, for the training hook
pip install "lofop[all]"         # models + deploy + tensorboard in one go
python -c "from lofop.ops import build_native; build_native()"   # optional C++ fast path
```

From a source checkout instead:

```bash
pip install -e ".[dev]"          # framework + dev tools (pytest, ruff, build, twine)
```

Requires Python 3.9+. The core and data layers run without PyTorch (edge/CI friendly); torch
attaches only to the model and training subsystems. Maintainer release steps live in
[`docs/packaging.md`](docs/packaging.md).

## Quickstart

**Dataset tools** (no torch needed):

```bash
lofop dataset convert  --from coco --source instances.json --to yolo --target out/
lofop dataset validate --format yolo --source out/            # exit 1 on errors
lofop dataset stats    --format coco --source instances.json -o stats.md
lofop dataset show     --format coco --source instances.json -o vis/ --limit 10
```

**Train and benchmark** (five-minute CPU demo that fills the metric table end to end):

```bash
python examples/train_shapes.py --epochs 30 --workdir runs/shapes
lofop benchmark --config lofop/configs/lofop-detect/n.yaml --config lofop/configs/lofop-detect/s.yaml -o table.md
lofop predict  --config n --checkpoint runs/shapes/checkpoints/best.pt --source image.png
lofop evaluate --config n --checkpoint runs/shapes/checkpoints/best.pt --format coco --source val.json
lofop export --config lofop/configs/lofop-detect/n.yaml --checkpoint runs/shapes/checkpoints/best.pt -o model.onnx
lofop export --config lofop/configs/lofop-detect/n.yaml --format tensorrt --fp16 -o model.engine   # NVIDIA GPU
lofop doctor   # environment + backend diagnostics
```

Measured on the fixed-protocol benchmark (`benchmarks/quality_benchmark.py`, 30 CPU epochs,
128px shapes): mAP@50 0.92, best F1 0.90, 0.8 false positives/image at conf 0.25, 115 FPS
end-to-end predict. Accuracy on a real dataset awaits a full GPU training run — the protocol
is documented in `docs/lofop-detect.md`, and the table renders `-` until numbers are measured.

**Python SDK** — the one-import path ([full reference](docs/sdk.md)):

```python
from lofop import Detector

det = Detector("lofop-detect-ex", num_classes=2, class_names=["cat", "dog"])
det.train(data_format="coco", train_source="train.json", image_root="images/", epochs=100)
for hit in det.predict("photo.jpg"):                  # boxes in original image coordinates
    print(hit.boxes, hit.scores, hit.labels)
det.export("model.onnx")
```

**Deploy in C++ or Python** — one exported model, two runtimes ([full guide](docs/cpp-sdk.md)):

```cpp
#include <lofop/lofop.hpp>                       // no Python, no PyTorch
lofop::Detector model("model.onnx");
for (const auto& hit : model.predict("image.ppm")) std::cout << hit.confidence;
```

```python
from lofop.runtime import Detector               # no PyTorch
model = Detector("model.onnx")
results = model.predict("image.jpg")
```

Lower-level control remains fully public — registries, `Config`, `Trainer`, and the deploy
functions are the same objects the SDK uses:

```python
from lofop import Config, HUB
import lofop.models                                   # registers model components

cfg = Config.load("lofop/configs/lofop-detect/s.yaml")
model = HUB.build(cfg.model)                          # ready LofopDetect
```

Custom components plug in without touching the framework:

```python
from lofop.registries import BACKBONES

@BACKBONES.register()
class MyNet: ...
# then in YAML:  backbone: {type: backbone/MyNet, ...}
```

## Repository layout

```
lofop/
  core/          # registry, config, events, plugins, logging, exceptions (torch-free)
  data/          # canonical dataset model, COCO/YOLO/VOC adapters, validator, statistics
  models/        # LOFOP-Detect: RidgeNet, DeltaFusion, ApexHead, losses, assigner
  training/      # trainer, EMA, checkpoints, torch data bridge, COCO-protocol evaluator
  deploy/        # ONNX + TensorRT export, torch-free post-processing
  ops/ + csrc/   # native C++ IoU/NMS with Python fallback
  utils/         # model benchmarking (metric table, FLOPs, FPS)
  sdk.py         # high-level Python SDK: the Detector class (docs/sdk.md)
  runtime/       # torch-free ONNX inference: Detector, Detection, Image
  csrc/          # C++ kernels: box ops + shared letterbox preprocessing
  configs/       # packaged model family definitions (n, s, ex)
  cli.py         # `lofop` command: dataset / train / benchmark / export
cpp/             # C++ inference SDK (CMake): headers, C ABI, tests, example
configs/         # training config examples
docker/          # CPU, CUDA, and ONNX Runtime images
docs/            # architecture, per-module references, LOFOP-Detect design doc
benchmarks/      # reusable performance measurement scripts
examples/        # end-to-end runnable demos
tests/           # pytest suite mirroring the package layout (316 tests)
```

## Development

```bash
python -m pytest                                   # run the test suite
ruff check lofop tests benchmarks examples         # lint
python benchmarks/bench_core.py -o report.md       # core engine micro-benchmarks
python benchmarks/bench_ops.py                     # C++ vs Python ops speedups
python benchmarks/bench_detect.py                  # detector params + latency
python benchmarks/run_suite.py                     # full suite -> results.md/.csv/.json
```

Every push and pull request runs the full test suite (Python 3.9/3.11/3.12, native C++ ops
built), lint, a version-consistency check, and a distribution build check via GitHub Actions
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)), so `main` stays releasable.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and the quality gates,
[CHANGELOG.md](CHANGELOG.md) for release notes, and [SECURITY.md](SECURITY.md) for reporting
vulnerabilities. Bug reports and feature requests use the issue templates.

## Roadmap

Ordered by expected return: published pretrained checkpoints (GPU training runs),
letterboxing, a torch-free tracking module, ONNX export for the segmentation/pose variants,
then inference sources (video/RTSP/webcam), OpenVINO engines, and REST serving. The full
subsystem map with per-phase status lives in [`docs/architecture.md`](docs/architecture.md).

## Authors

DURGAMANI SASIKUMAR and Nishanandhini A. (Assistant Professor).

## License

Apache License 2.0. See [LICENSE](LICENSE).
