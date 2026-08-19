# LOFOP Operator's Manual

A complete, practical guide to installing, running, training, exporting, and deploying LOFOP and
its flagship detector **LOFOP-Detect**. This is the hands-on manual; for architecture and design
rationale see [`docs/architecture.md`](docs/architecture.md) and the per-module docs it links.

- **Version:** 1.2.1
- **Python:** 3.9+
- **Platforms:** Linux, macOS, Windows

## Table of contents

1. [Requirements](#1-requirements)
2. [Installation](#2-installation)
3. [Verify your install](#3-verify-your-install)
4. [Command-line reference](#4-command-line-reference)
5. [Configuration system](#5-configuration-system)
6. [Datasets](#6-datasets)
7. [Training](#7-training)
8. [Evaluation and benchmarking](#8-evaluation-and-benchmarking)
9. [Export and deployment](#9-export-and-deployment)
10. [Native C++ ops](#10-native-c-ops)
11. [Python SDK](#11-python-sdk)
12. [Extending LOFOP](#12-extending-lofop)
13. [Troubleshooting](#13-troubleshooting)
14. [Contribution workflow](#14-contribution-workflow)

---

## 1. Requirements

| Component | Needed for | Notes |
|---|---|---|
| Python 3.9+ | everything | 3.10/3.11 recommended |
| PyYAML, Pillow | core + data | installed automatically |
| PyTorch >= 2.0 | models, training, export | `pip install lofop[models]` |
| onnx, onnxruntime | ONNX export/verify | `pip install lofop[deploy]` |
| tensorrt + CUDA GPU | TensorRT engines | NVIDIA only; `pip install lofop[tensorrt]` |
| A C++ compiler | optional native ops speedup | g++/clang/MSVC; never required |

The **core and data layers need no PyTorch** — dataset tooling runs on an edge box or in CI without
a deep-learning runtime. Torch attaches only to `lofop.models` and `lofop.training`.

## 2. Installation

Install from a package manager:

```bash
pip install lofop             # PyPI
yay -S lofop                  # Arch Linux (AUR)
```

Add optional feature sets with pip extras:

```bash
pip install "lofop[models]"     # + PyTorch, for lofop.models / lofop.training
pip install "lofop[deploy]"     # + onnx, onnxruntime, for ONNX export
pip install "lofop[tensorrt]"   # + tensorrt, numpy (NVIDIA GPU machines only)
pip install "lofop[tensorboard]"   # + tensorboard, for the training hook
pip install "lofop[all]"        # models + deploy + tensorboard
```

On the AUR the optional features are `optdepends` (`python-pytorch`, `python-onnx`,
`python-onnxruntime`, `gcc`) — install them as needed.

From a source checkout instead:

```bash
pip install -e ".[dev]"        # framework + dev tools (pytest, ruff, build, twine)
pip install -e ".[dev,models,deploy]"   # combine extras
```

Maintainers: release steps (PyPI + AUR) are in [`docs/packaging.md`](packaging.md).

### Platform notes

- **Windows:** everything above works in PowerShell. Use a virtual environment (`python -m venv
  .venv; .venv\Scripts\Activate.ps1`) to avoid clashing with other projects. The optional C++ ops
  need a compiler — see [section 10](#10-native-c-ops).
- **GPU:** install the CUDA build of PyTorch from the official index if you want GPU training, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
- **CPU-only:** `pip install torch --index-url https://download.pytorch.org/whl/cpu` keeps the
  download small.

## 3. Verify your install

```bash
lofop version                  # prints 1.2.1
python -m pytest               # runs the test suite (316 passed, 3 skipped without GPU/TensorBoard)
```

Then run the self-contained demo — it generates its own data, trains LOFOP-Detect end to end, and
writes a metric table. No external files needed:

```bash
python examples/train_shapes.py --epochs 30 --workdir runs/shapes
```

You should see per-epoch loss and evaluation lines, and a table at `runs/shapes/metric-table.md`.
If this runs, your install is fully working.

## 4. Command-line reference

The `lofop` command is installed with the package. Global option: `--log-level DEBUG|INFO|WARNING`
(or set the `LOFOP_LOG_LEVEL` environment variable).

```
lofop {version, dataset, train, benchmark, predict, evaluate, export, doctor, runs}
```

### 4.1 `lofop version`

```bash
lofop version
```

### 4.2 `lofop dataset`

Format-agnostic dataset tools. Supported formats: `coco`, `yolo`, `voc`.

**Convert** between any two formats:

```bash
lofop dataset convert --from coco --source instances.json --to yolo --target out/ \
    --image-root images/
```

| Flag | Meaning |
|---|---|
| `--from` | source format (`coco`/`yolo`/`voc`) |
| `--source` | source file (COCO JSON) or dataset root (YOLO/VOC) |
| `--to` | target format |
| `--target` | output file or directory |
| `--image-root` | image directory for COCO sources (optional) |

**Validate** a dataset (exit code 1 if it has errors — CI-friendly):

```bash
lofop dataset validate --format yolo --source dataset_root/
lofop dataset validate --format coco --source instances.json --no-check-images
```

Reports degenerate/out-of-bounds/non-finite boxes, unknown category references, missing image
files, duplicate entries, and empty datasets, split into blocking errors and advisory warnings.

**Statistics**:

```bash
lofop dataset stats --format coco --source instances.json -o stats.md   # markdown
lofop dataset stats --format coco --source instances.json --json        # JSON
```

Reports per-class counts, boxes-per-image, COCO small/medium/large size breakdown, and image-size
distribution.

**Visualize** (draw the ground-truth boxes onto the images):

```bash
lofop dataset show --format coco --source instances.json --image-root images/ -o vis/
lofop dataset show --format yolo --source dataset_root/ -o vis/ --limit 20
```

Writes one PNG per sample (`<id>_<image>.png`) with each box outlined and captioned by class name;
colors are stable per class. Pillow only, so it runs without the `models` extra. The same drawing
primitive is available in code as `lofop.data.draw_boxes` for rendering predictions.

### 4.3 `lofop train`

```bash
lofop train --config configs/train/example.yaml
lofop train --config configs/train/example.yaml --resume   # continue from last.pt
```

Reads a training config (see [section 7](#7-training)), trains, checkpoints to
`training.checkpoint_dir`, and evaluates if a validation set is configured.

### 4.4 `lofop benchmark`

Produce the standard metric table (params, FLOPs, model size, CPU/GPU FPS) for one or more model
configs:

```bash
lofop benchmark --config lofop/configs/lofop-detect/n.yaml --config lofop/configs/lofop-detect/s.yaml \
    --size 640 -o table.md
lofop benchmark --config lofop/configs/lofop-detect/s.yaml --checkpoint runs/train/best.pt
lofop benchmark --config lofop/configs/lofop-detect/n.yaml --results-dir benchmarks/results
```

Accuracy rows (mAP/precision/recall) require a trained checkpoint and an evaluation run; unmeasured
cells render as `-`. `--results-dir` additionally writes `results.md`, `results.csv`, and
`results.json` (params, FLOPs, size, CPU/GPU FPS, peak memory, and any accuracy) so a run is both
human- and machine-readable. The full variant sweep is `python benchmarks/run_suite.py`
(add `--isolated` for clean per-model peak memory).

### 4.5 `lofop export`

```bash
# ONNX (verified against the torch model by default):
lofop export --config lofop/configs/lofop-detect/s.yaml --checkpoint runs/train/best.pt \
    --size 640 -o model.onnx

# TensorRT engine (NVIDIA GPU):
lofop export --config lofop/configs/lofop-detect/s.yaml --checkpoint runs/train/best.pt \
    --format tensorrt --fp16 -o model.engine
```

| Flag | Meaning |
|---|---|
| `--format` | `onnx` (default) or `tensorrt` |
| `--size` | input resolution baked into the graph |
| `--opset` | ONNX opset (default 18) |
| `--dynamic` | ONNX: symbolic batch/height/width axes for variable input sizes (multiples of 32) |
| `--no-verify` | skip the onnxruntime verification step |
| `--fp16` | TensorRT: enable FP16 kernels |
| `--checkpoint` | load weights (EMA weights are used automatically if present) |

### 4.6 `lofop predict`

Run detection on one or more images. Boxes come back in the original image
coordinates. Accepts a variant name (`n`/`s`/`ex`) or a config path for
`--config`.

```bash
lofop predict --config s --checkpoint runs/train/best.pt \
    --source photo1.jpg photo2.jpg --score-threshold 0.3
lofop predict --config s --checkpoint runs/train/best.pt \
    --source photo.jpg --json -o detections.json
```

| Flag | Meaning |
|---|---|
| `--source` | one or more image paths |
| `--num-classes` | classes the head predicts (default 80) |
| `--size` | inference resolution (default 640) |
| `--score-threshold` | override the confidence cut for this run |
| `--json` / `-o` | print JSON / also write it to a file |

### 4.7 `lofop evaluate`

COCO-protocol evaluation on a dataset. Prints mAP@50, mAP@50:95, precision,
recall and F1; the JSON form additionally carries per-class precision/recall
and the confusion matrix.

```bash
lofop evaluate --config s --checkpoint runs/train/best.pt \
    --format coco --source instances_val.json --image-root images/
lofop evaluate --config s --checkpoint best.pt \
    --format yolo --source dataset_root/ --json -o metrics.json
```

### 4.8 `lofop doctor`

Print the environment LOFOP sees: version, Python/OS, the active box-ops
backend (native C++ or pure Python), and which optional dependencies (torch,
onnx, onnxruntime, tensorrt, ...) are installed. Runs without the `models`
extra, so it is the first command to reach for when diagnosing an install.

```bash
lofop doctor
```

## 5. Configuration system

LOFOP experiments are plain YAML, instantiated through the registry. Files support:

- **Inheritance** via `extends:` (paths relative to the file; parents deep-merged in order, child
  wins).
- **Interpolation**: `${a.b.c}` references another config value; `${env:VAR:default}` reads the
  environment.
- **Freezing** (SDK): `cfg.freeze()` makes a config read-only.

### Model variants

The detector family lives in `lofop/configs/lofop-detect/`:

| File | Variant | Params |
|---|---|---|
| `base.yaml` | shared settings (not used directly) | - |
| `n.yaml` | nano (edge-first) | ~1.3M |
| `s.yaml` | small (reference) | ~3.8M |
| `ex.yaml` | extra (higher-accuracy GPU) | ~20.1M |

Each variant only overrides widths/depths; the architecture is identical. Build one with
`HUB.build(Config.load("lofop/configs/lofop-detect/s.yaml").model)`.

Every size also has a task variant on the same detector:

- `n-seg.yaml` / `s-seg.yaml` / `ex-seg.yaml` -- instance segmentation
  (`model/LofopSegment`): a prototype mask branch trained from COCO polygon
  annotations (plain boxes work as a coarse fallback). `predict` adds a
  `masks` field; checkpoints are named by the variant, e.g.
  `lofop-detect-n-seg.pt`.
- `n-pose.yaml` / `s-pose.yaml` / `ex-pose.yaml` -- pose estimation
  (`model/LofopPose`): per-detection keypoints trained from COCO keypoint
  annotations. `num_keypoints` defaults to 17 (COCO person) and is
  configurable (`Detector(..., num_keypoints=K)`). `predict` adds a
  `keypoints` field of `(x, y, visibility)` rows.

The SDK trains them exactly like detection -- mask/keypoint targets are read
automatically from the dataset's annotations. Note: horizontal-flip
augmentation is disabled for pose training (flipping would need left/right
keypoint swapping), and `strong_augment` is not yet supported for seg/pose.
ONNX export currently covers detection models only.

### Inference NMS modes

Every model (and `Detector(..., nms_mode=...)`) supports three
duplicate-removal strategies at inference:

| Mode | What it does | When to use |
|---|---|---|
| `greedy` (default) | classic class-aware NMS | general use; unchanged behavior |
| `soft` | decays overlapping scores (gaussian, `soft_nms_sigma`) instead of dropping | crowded scenes with touching objects |
| `free` | NMS-free: keeps only 3x3 local score peaks -- pure tensor math, no suppression loop | latency-critical paths; large candidate counts |

Config keys: `model.nms_mode`, `model.soft_nms_sigma`. At runtime:
`det.model.nms_mode = "soft"`.

### Training config

See [`configs/train/example.yaml`](configs/train/example.yaml). It `extends` a model variant and
adds two blocks:

```yaml
extends: [../../lofop/configs/lofop-detect/s.yaml]
num_classes: 80                    # your dataset's class count

data:
  format: coco                     # coco | yolo | voc
  train_source: /path/to/instances_train.json
  val_source: /path/to/instances_val.json   # optional
  image_root: /path/to/images      # COCO only
  image_size: 640

training:
  epochs: 100
  batch_size: 16
  lr: 0.01
  optimizer: SGD                   # SGD | AdamW
  weight_decay: 0.0005
  warmup_epochs: 3
  scheduler: warmup_cosine         # warmup_cosine | warmup_linear | constant | step
  scheduler_kwargs: {min_factor: 0.05}
  patience: 20                     # early stop after N epochs without improvement (omit to disable)
  min_delta: 0.0
  amp: true
  workers: 4
  checkpoint_dir: runs/train
```

Every `training:` key maps directly to a `Trainer` argument (section 7).

## 6. Datasets

LOFOP normalizes every format into one canonical model (`Dataset` / `Sample` / `BoxAnnotation`,
boxes in absolute xyxy pixels), so conversion, validation, and training work identically across
formats.

| Format | `--source` / `--target` | Expected layout |
|---|---|---|
| `coco` | annotation JSON file | standard `images`/`annotations`/`categories` |
| `yolo` | dataset root directory | `classes.txt`, `images/`, `labels/*.txt` (normalized cxcywh) |
| `voc` | dataset root directory | `Annotations/*.xml`, `JPEGImages/` |

Round-trip caveats: COCO preserves category/image ids and `iscrowd`; YOLO reassigns contiguous
class indices (inherent to the format) and reads image sizes from the files; VOC preserves the
`difficult` flag. Full reference: [`docs/data.md`](docs/data.md).

**Always validate before training** — it catches the defects that otherwise appear hours in as NaN
losses:

```bash
lofop dataset validate --format coco --source instances_train.json --image-root images/
```

## 7. Training

### 7.1 Workflow

1. Prepare a dataset in COCO/YOLO/VOC form and validate it.
2. Copy `configs/train/example.yaml`, set `num_classes` and the `data.*` paths.
3. Run `lofop train --config your_config.yaml`.
4. Checkpoints (`last.pt`, `best.pt`) land in `training.checkpoint_dir`. `best.pt` tracks the
   highest mAP@50:95 (or lowest loss when no validation set is configured).

### 7.2 What the trainer does

- **Mixed precision (AMP)** on CUDA; automatic no-op on CPU.
- **EMA weights** — evaluation and export use the exponential moving average, which scores higher
  than the raw weights.
- **Augmentation** — horizontal flip by default; set `data.strong_augment: true`
  (or `Detector.train(strong_augment=True)`) for the richer recipe: 2x2 mosaic
  (four images combined, boxes remapped) plus brightness/contrast/saturation
  jitter. Recommended for real-data training; off by default.
- **Config-driven LR schedule** — `scheduler:` selects `warmup_cosine` (default; linear warmup then
  cosine decay to 5% of peak), `warmup_linear`, `constant`, or `step`; tune via `scheduler_kwargs`.
  New schedules can be registered in the `scheduler` group.
- **Early stopping** — set `patience` (epochs without a validation-metric gain beyond `min_delta`)
  to stop a plateaued run and emit `train.early_stop`. Disabled when `patience` is omitted.
- **Gradient clipping** at norm 10.
- **Atomic checkpointing** with `--resume` support.
- **Lifecycle events** on the LOFOP event bus (`train.start`, `train.epoch_end`, `eval.end`,
  `checkpoint.saved`, `train.end`, `train.early_stop`) — attach trackers/plugins without modifying
  the trainer. **TensorBoard**: `from lofop.training import attach_tensorboard;
  attach_tensorboard("runs/tb")` before `fit()` logs loss and metrics (needs the `tensorboard`
  extra).

### 7.3 Resuming

```bash
lofop train --config your_config.yaml --resume
```

Restores model, EMA, optimizer, and epoch from `checkpoint_dir/last.pt`.

### 7.4 Multi-GPU

The trainer wraps the model in `DistributedDataParallel` when `torch.distributed` is initialized:

```bash
torchrun --nproc_per_node=4 -m lofop.cli train --config your_config.yaml
```

The DDP path follows the standard recipe; validate on your cluster before long runs.

### 7.5 Programmatic training

See [section 11.3](#113-train-in-python).

### 7.5 Experiment tracking (`lofop.mlops`)

Wrap any training call to record the run permanently -- settings, machine,
per-epoch loss/mAP history, best epoch, final metrics, checkpoint paths:

```python
from lofop.mlops import track

with track("runs/registry", name="person-v2", tags=["coco"], meta={"lr": 0.002}):
    det.train(...)
```

Then from any machine (no PyTorch needed -- records are plain JSON):

```bash
lofop runs list --root runs/registry
lofop runs show 20260715_183000 --root runs/registry
lofop runs compare 20260715_183000 20260714_120000 --root runs/registry   # metric table
```

Tracking observes the standard training events, so it works with the SDK, the
CLI, or any custom loop that emits them; a crashed run is recorded with
`status: failed` instead of disappearing.

## 8. Evaluation and benchmarking

`lofop benchmark` fills the standard comparison table. Structural metrics (params, FLOPs, model
size, CPU/GPU FPS) are always measured; accuracy metrics fill in when a checkpoint and evaluation
are supplied, and show `-` otherwise so the table never displays numbers that were not measured.

```bash
lofop benchmark --config lofop/configs/lofop-detect/n.yaml --config lofop/configs/lofop-detect/s.yaml -o table.md
```

The evaluator implements the COCO protocol: greedy score-descending matching, 101-point
interpolated AP, mAP@50:95, and micro precision/recall. It also returns F1 (the harmonic mean
of the micro precision/recall), per-class precision/recall, and a class confusion matrix
(cross-class matching, with a trailing background row/column for spurious detections and missed
ground truths). Use it directly via `lofop.training.evaluate_detections` (section 11), or through
`lofop evaluate` for a dataset on disk.

## 9. Export and deployment

Full reference: [`docs/deploy.md`](docs/deploy.md). Two-stage by design: **model -> ONNX ->
engine**, so ONNX Runtime and TensorRT share one verified graph and one post-processing routine.

### 9.1 ONNX

```bash
lofop export --config lofop/configs/lofop-detect/s.yaml --checkpoint runs/train/best.pt -o model.onnx
```

The graph contains the network **plus box decoding**; NMS stays outside (runtimes disagree on NMS
ops, and thresholds are deployment decisions). Export verifies the graph against the torch model by
default.

### 9.2 TensorRT

```bash
lofop export --config lofop/configs/lofop-detect/s.yaml --checkpoint runs/train/best.pt \
    --format tensorrt --fp16 -o model.engine
```

FP16 is one flag; INT8 needs calibration data (`export_tensorrt(..., int8=True,
calibration_inputs=[...])` in the SDK). Engines are GPU- and TensorRT-version-specific — build on
the target hardware.

### 9.3 Inference without PyTorch

Serving hosts need only a runtime (onnxruntime/TensorRT) plus the torch-free LOFOP core:

```python
import onnxruntime
from lofop.deploy import postprocess_dense

session = onnxruntime.InferenceSession("model.onnx")
boxes, scores = session.run(None, {"images": batch})     # (1, N, 4), (1, N, C)
det = postprocess_dense(boxes[0], scores[0], score_threshold=0.25, nms_iou=0.6)
det.boxes, det.scores, det.labels
```

### 9.4 Docker

Three images ([`docker/`](docker/README.md)), all with the `lofop` CLI as entrypoint:

```bash
docker build -f docker/Dockerfile-cpu  -t lofop:cpu  .   # CPU inference + dataset tools
docker build -f docker/Dockerfile-gpu  -t lofop:gpu  .   # CUDA training/inference
docker build -f docker/Dockerfile-onnx -t lofop:onnx .   # ONNX Runtime, smallest footprint

docker run --rm -v /data:/data lofop:cpu dataset validate --format yolo --source /data/yolo
docker run --rm --gpus all lofop:gpu version
```

## 10. Native C++ ops

Box IoU and NMS have a pure-Python implementation (always available, every OS) and an optional C++
fast path (20-200x faster) loaded via ctypes. **A compiler is never required** — the Python path is
the automatic fallback.

Check and build:

```python
from lofop.ops import backend, build_native
print(backend())     # "python" or "native"
build_native()       # compiles the C++ library if a compiler is available
```

The builder auto-selects a toolchain:

| OS | Compilers tried |
|---|---|
| Linux / macOS | `g++`, `clang++`, `c++` |
| Windows | `g++` / `clang++` (MinGW-w64 or LLVM), then MSVC `cl.exe` |

Override with `build_native(compiler="clang++")` or the `CXX` environment variable.

**Windows:** to get the C++ path, either install [MinGW-w64](https://www.mingw-w64.org/) (so `g++`
is on PATH) or run from a *Developer PowerShell for Visual Studio* (so `cl.exe` is on PATH), then
call `build_native()`. Without a compiler you simply stay on the Python path — nothing breaks.

## 11. Python SDK

The high-level API is one class — full reference with every argument documented:
[`docs/sdk.md`](docs/sdk.md).

### 11.1 The Detector class

```python
from lofop import Detector

det = Detector("lofop-detect-ex", num_classes=2, class_names=["cat", "dog"])
det.train(data_format="coco", train_source="train.json",
          val_source="val.json", image_root="images/", epochs=100)
for hit in det.predict("photo.jpg"):        # boxes in ORIGINAL image coordinates
    print(hit.boxes, hit.scores, hit.labels)
det.export("model.onnx")                    # or .engine for TensorRT
det.save("weights.pt")
```

Variant names: `lofop-detect-n` (1.3M) / `-s` (3.8M) / `-ex` (20.1M), with or without the
prefix; also accepts a config YAML path, a config dict, or a prebuilt module. `checkpoint=`
loads Trainer checkpoints (EMA weights win automatically).

### 11.1b Build a model at the low level

```python
import lofop.models                        # registers model components
from lofop import Config, HUB

cfg = Config.load("lofop/configs/lofop-detect/s.yaml")
model = HUB.build(cfg.model).eval()         # a LofopDetect
results = model.predict(images)             # per image: {"boxes","scores","labels"}
```

### 11.2 Load and convert datasets

```python
from lofop.data import load_dataset, convert_dataset, validate_dataset, compute_stats

ds = load_dataset("coco", "instances.json", image_root="images/")
convert_dataset("coco", "instances.json", "yolo", "out/")
report = validate_dataset(ds)               # report.ok, report.errors, report.warnings
print(compute_stats(ds).to_markdown())
```

### 11.3 Train in Python

```python
import lofop.models
from lofop import Config, HUB
from lofop.data import load_dataset
from lofop.training import DetectionTorchDataset, Trainer

cfg = Config.load("lofop/configs/lofop-detect/s.yaml")
model = HUB.build(cfg.model)
train = DetectionTorchDataset(load_dataset("coco", "train.json", image_root="imgs/"),
                              image_size=640, augment=True)
val = DetectionTorchDataset(load_dataset("coco", "val.json", image_root="imgs/"), image_size=640)

trainer = Trainer(model, train, val, epochs=100, batch_size=16, lr=0.01,
                  checkpoint_dir="runs/train")
metrics = trainer.fit()
print(metrics.map50, metrics.map50_95)
```

### 11.4 Evaluate and export

```python
from lofop.training import evaluate_detections
from lofop.deploy import export_onnx, export_tensorrt

metrics = evaluate_detections(predictions, targets)     # COCO-protocol metrics
export_onnx(model, "model.onnx", image_size=640)
export_tensorrt(model, "model.engine", image_size=640, fp16=True)   # NVIDIA GPU
```

## 12. Extending LOFOP

Every component is a registry entry. Add your own without touching the framework:

```python
from lofop.registries import BACKBONES

@BACKBONES.register()
class MyNet(...): ...
# then in YAML:  backbone: {type: backbone/MyNet, ...}
```

Registry groups: `model`, `backbone`, `neck`, `head`, `loss`, `metric`, `optimizer`, `scheduler`,
`dataset`, `transform`, `hook`, `dataset_format`.

**Plugins** package extensions for distribution — declare a `lofop.plugins` entry point, or register
programmatically with `PLUGINS.add(...)`. See [`docs/core-engine.md`](docs/core-engine.md).

## 13. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No module named pytest` | dev extra not installed | `pip install -e ".[dev]"` |
| `No C++ compiler found` on `build_native()` | no compiler on PATH | Install g++/clang/MSVC, or ignore — the Python ops path works regardless (section 10) |
| `Cannot read COCO file: instances.json` | the example path does not exist on your disk | point `--source` at a real file; the README commands are templates |
| `YOLO root has no classes.txt` | pointing at a directory without the expected layout | ensure `classes.txt` + `images/` + `labels/` (section 6) |
| `TensorRT is not installed` | building an engine without the tensorrt package/GPU | install `lofop[tensorrt]` on an NVIDIA machine, or export ONNX instead |
| DataLoader hangs/errors on Windows | multiprocessing workers | set `workers: 0` in the training config |
| `Config is frozen and cannot be modified` | mutating a frozen config | clone before editing, or freeze later |
| Import error for `lofop.models` | torch not installed | `pip install -e ".[models]"` |
| Slow NMS on large images | Python ops path active | build the C++ ops (section 10); check with `lofop.ops.backend()` |

Raise the log level for detail: `lofop --log-level DEBUG ...` or `set LOFOP_LOG_LEVEL=DEBUG`
(Windows) / `export LOFOP_LOG_LEVEL=DEBUG` (Unix).

## 14. Contribution workflow

Development happens on the `tedo` branch; `main` is the reviewed base.

```bash
git checkout tedo
git pull origin tedo
# ... make changes ...
python -m pytest && ruff check lofop tests benchmarks examples
git commit -am "Describe the change"
git push origin tedo
```

Then open a pull request `tedo -> main` for review. Every change ships with tests and, when it
adds or changes user-facing behavior, a README/docs update. Keep the test suite green and the lint
clean before requesting review.

---

*For architecture and design rationale, continue to [`docs/architecture.md`](docs/architecture.md),
[`docs/lofop-detect.md`](docs/lofop-detect.md), and the module references in [`docs/`](docs/).*
