# The LOFOP SDK Book

**A complete guide to the LOFOP Python SDK, from your first detection to extending the framework.**

LOFOP 1.2.2 &middot; Python 3.9+ &middot; Apache-2.0

---

This book teaches the whole SDK in order. Part I gets you detecting objects in five
minutes. Part II covers the everyday work: data, training, evaluation. Part III covers
the task variants: segmentation, pose, NMS-free detection, and tracking. Part IV covers
deployment, experiment tracking, and the ops layer. Part V is for people extending
LOFOP. Part VI is reference material you will come back to.

Every code block runs as written. Every number quoted was measured, not estimated; where
something has not been measured, this book says so rather than guessing.

## Contents

**Part I &mdash; Getting started**
1. [Install and verify](#1-install-and-verify)
2. [Your first detection](#2-your-first-detection)
3. [The Detector class](#3-the-detector-class)
4. [The Detections result](#4-the-detections-result)
5. [Choosing a model](#5-choosing-a-model)

**Part II &mdash; The core workflow**
6. [Datasets](#6-datasets)
7. [Training](#7-training)
8. [Evaluation and benchmarking](#8-evaluation-and-benchmarking)

**Part III &mdash; Task variants**
9. [Instance segmentation](#9-instance-segmentation)
10. [Pose estimation](#10-pose-estimation)
11. [NMS-free detection](#11-nms-free-detection)
12. [Tracking](#12-tracking)
13. [The supervision bridge](#13-the-supervision-bridge)

**Part IV &mdash; Production**
14. [Export and deployment](#14-export-and-deployment)
15. [The torch-free runtime](#15-the-torch-free-runtime)
16. [Experiment tracking](#16-experiment-tracking)
17. [The ops layer](#17-the-ops-layer)

**Part V &mdash; Extending LOFOP**
18. [Configs and registries](#18-configs-and-registries)
19. [Writing your own components](#19-writing-your-own-components)
20. [Events and hooks](#20-events-and-hooks)

**Part VI &mdash; Reference**
21. [The CLI](#21-the-cli)
22. [Cookbook](#22-cookbook)
23. [Troubleshooting](#23-troubleshooting)
24. [API reference](#24-api-reference)

---

# Part I &mdash; Getting started

## 1. Install and verify

```bash
pip install lofop
```

That base install is deliberately small: PyYAML and Pillow, nothing else. It gives you
the dataset tools, the config system, the ops layer, and torch-free deployment
post-processing. Everything heavier is an opt-in extra.

| Extra | Adds | Unlocks |
|---|---|---|
| `lofop[models]` | PyTorch | `Detector`, training, all models |
| `lofop[deploy]` | onnx, onnxruntime | ONNX export and verification |
| `lofop[tracking]` | numpy | `Detector.track`, `lofop.tracking` |
| `lofop[supervision]` | supervision, numpy | the ecosystem bridge |
| `lofop[tensorboard]` | tensorboard | the training hook |
| `lofop[tensorrt]` | tensorrt, numpy | TensorRT engines (NVIDIA only) |
| `lofop[all]` | models + deploy + tensorboard + numpy | most things at once |

For SDK work you want at least:

```bash
pip install "lofop[models]"
```

Verify:

```bash
lofop version      # 1.2.2
lofop doctor       # environment + which ops backend is active
```

**A promise this book relies on.** A stock `pip install lofop` never raises
`ModuleNotFoundError`. If you call something that needs an extra you do not have, LOFOP
raises `LofopError` telling you the exact install command:

```python
>>> import lofop.tracking          # fine, always works
>>> lofop.tracking.LofopTracker()
LofopError: LOFOP tracking needs numpy for its motion model.
Install the tracking extra:  pip install "lofop[tracking]"
```

Optionally, compile the native ops for a 20&ndash;200&times; speedup on box operations. It is
never required &mdash; there is a pure-Python fallback that produces identical results:

```python
from lofop.ops import build_native, backend
build_native()                    # needs g++/clang/MSVC
print(backend())                  # 'native', or 'cuda' if you built with cuda=True
```

## 2. Your first detection

```python
from lofop import Detector

det = Detector("lofop-detect-s", num_classes=80)
results = det.predict("photo.jpg")

for hit in results:
    print(hit.boxes)     # [[x1, y1, x2, y2], ...] in the ORIGINAL image's pixels
    print(hit.scores)    # descending confidence
    print(hit.labels)    # class indices
```

`predict` always returns a **list**, one `Detections` per input image, even for a single
image. That keeps the single and batch paths identical.

Without a trained checkpoint this model has random weights and will detect nothing
useful. To get real results, either train it (chapter 7) or load weights:

```python
det = Detector("lofop-detect-s", num_classes=80, checkpoint="best.pt")
```

The five-minute end-to-end demo, which generates its own data and fills a metric table:

```bash
python examples/train_shapes.py --epochs 30 --workdir runs/shapes
```

## 3. The Detector class

`Detector` is the whole framework behind one class. It wraps the lower layers without
hiding them: `det.model` is the real `torch.nn.Module`, and every lower-level API stays
public.

### Construction

```python
Detector(
    model="lofop-detect-s",     # variant name, YAML path, Config/dict, or built module
    *,
    num_classes=80,
    checkpoint=None,            # best.pt, last.pt, or a bare state_dict
    class_names=None,           # names for your classes
    image_size=640,             # square inference resolution
    device=None,                # "cpu" / "cuda"; auto-selects CUDA when available
    num_keypoints=None,         # pose variants only
    nms_mode=None,              # "greedy" | "soft" | "free"; dense variants only
)
```

Four ways to say which model:

```python
Detector("lofop-detect-n")                          # variant name
Detector("n")                                       # the prefix is optional
Detector("path/to/my-model.yaml")                   # your own config
Detector(already_built_module)                      # a module you built yourself
```

`class_names` attaches human labels and is validated against `num_classes`:

```python
det = Detector("n", num_classes=3, class_names=["cat", "dog", "bird"])
hit = det.predict("pets.jpg")[0]
print([det.class_names[i] for i in hit.labels])     # ['cat', 'dog']
```

### The eight methods

| Method | Purpose | Chapter |
|---|---|---|
| `predict(source)` | detect in images | 3 |
| `track(frames)` | detect + assign identities across frames | 12 |
| `train(...)` | train the model | 7 |
| `evaluate(val_data)` | COCO-protocol metrics | 8 |
| `export(path)` | ONNX or TensorRT | 14 |
| `save(path)` | write a loadable checkpoint | 7 |
| `optimize()` | CPU inference optimisation | 14 |
| `num_parameters` | parameter count (property) | 5 |

### predict in detail

```python
predict(source, *, score_threshold=None) -> list[Detections]
```

`source` accepts a file path, a PIL image, a CHW float tensor in `[0, 1]`, or a sequence
of any of those:

```python
det.predict("a.jpg")                            # one path
det.predict(["a.jpg", "b.jpg", "c.jpg"])        # several
det.predict(Image.open("a.jpg"))                # PIL
det.predict(torch.rand(3, 640, 640))            # tensor
```

`score_threshold` overrides the model's threshold **for that call only**; the model is
restored afterwards even if the call raises.

```python
det.predict("crowd.jpg", score_threshold=0.1)   # recall-first, more boxes
det.predict("crowd.jpg", score_threshold=0.6)   # precision-first, fewer
```

Boxes come back in the original image's coordinate system. LOFOP resizes internally to
`image_size` and maps the boxes back, so you never do that arithmetic yourself.

### Duplicate removal on dense variants

```python
det = Detector("n", num_classes=80, nms_mode="soft")
det.model.nms_mode = "greedy"          # changeable at runtime
```

| Mode | Behaviour | Use when |
|---|---|---|
| `greedy` *(default)* | classic class-aware NMS | general use |
| `soft` | decays overlapping scores instead of dropping them | crowded scenes, touching objects |
| `free` | keeps only 3&times;3 local score peaks; no suppression loop | latency matters |

Measured on `n` at 640px with 8,400 candidates: greedy 52.4 ms, soft 53.5 ms, free
49.2 ms per image on CPU.

`nms_mode` does not apply to the query variants of chapter 11 &mdash; they never run
suppression at all &mdash; and passing it raises a clear error.

## 4. The Detections result

```python
@dataclass
class Detections:
    boxes: list[list[float]]      # (K, 4) xyxy in original image pixels
    scores: list[float]           # K, descending
    labels: list[int]             # K class indices
    masks: Any = None             # (K, H, W) bool tensor  -- segmentation variants
    keypoints: Any = None         # (K, num_kp, 3)         -- pose variants
    tracker_ids: list[int] = None # K identities           -- from track()
```

```python
hit = det.predict("photo.jpg")[0]
print(len(hit))                          # number of detections
for box, score, label in zip(hit.boxes, hit.scores, hit.labels):
    print(f"{det.class_names[label]}: {score:.2f} at {box}")
```

The optional fields are `None` unless the model that produced them fills them in, so
code written against plain detection keeps working with every variant.

```python
sv = hit.to_supervision()                # chapter 13
```

## 5. Choosing a model

Fourteen variants, all pure configuration. Measured parameter counts at 80 classes:

| Variant | Class | Params | Task | Notes |
|---|---|---:|---|---|
| `n` | LofopDetect | 1.31M | detection | edge-first default |
| `s` | LofopDetect | 3.84M | detection | reference variant |
| `ex` | LofopDetect | 20.12M | detection | highest accuracy |
| `n-seg` | LofopSegment | 1.46M | + masks | prototype mask branch |
| `s-seg` | LofopSegment | 4.19M | + masks | |
| `ex-seg` | LofopSegment | 21.48M | + masks | |
| `n-pose` | LofopPose | 1.41M | + keypoints | 17 keypoints by default |
| `s-pose` | LofopPose | 4.05M | + keypoints | |
| `ex-pose` | LofopPose | 20.87M | + keypoints | |
| `q-n` | LofopQuery | 1.28M | detection | **NMS-free** |
| `q-s` | LofopQuery | 3.84M | detection | **NMS-free** |
| `mb-n` | LofopDetect | 0.89M | detection | SwiftNet backbone, smallest |
| `mb-s` | LofopDetect | 3.39M | detection | SwiftNet backbone |
| `mbq-n` | LofopQuery | 0.86M | detection | SwiftNet + NMS-free |

Measured CPU latency at 640px, batch 1 (p50): `n` 39 ms, `s` 75 ms, `ex` 291 ms. The
seg and pose branches add roughly 5&ndash;11% parameters and almost no latency, because
their extra heads run only in training and in `predict`.

**How to choose.** Start at `s`. Move to `n` or `mb-n` if CPU or NPU latency binds; move
to `ex` if you have GPU budget and accuracy matters more than speed. Pick `-seg` or
`-pose` when you need masks or keypoints. Pick `q-*` when post-processing cost or crowded
scenes are the problem &mdash; but budget more training epochs, because set-prediction
heads converge more slowly.

```python
det = Detector("q-n", num_classes=80)
print(det.num_parameters)     # 1276701
print(det)                    # Detector(classes=80, parameters=1,276,701, ...)
```

---

# Part II &mdash; The core workflow

## 6. Datasets

LOFOP normalises COCO, YOLO and VOC into one canonical model, so N formats need N
adapters instead of N&sup2; converters. This layer needs no PyTorch.

```python
from lofop.data import load_dataset, save_dataset, convert_dataset

data = load_dataset("coco", "instances.json", image_root="images/")
print(data)          # Dataset(name=..., categories=80, samples=5000, annotations=36781)
```

### The canonical model

```python
from lofop.data import Dataset, Sample, BoxAnnotation, Category

dataset = Dataset(
    name="my-data",
    categories=[Category(id=1, name="widget")],
    image_root="images/",
)
dataset.add_sample(Sample(
    image="0001.jpg", width=1920, height=1080,
    annotations=[BoxAnnotation(bbox=(100.0, 50.0, 300.0, 400.0), category_id=1)],
))
```

Boxes are always `(x1, y1, x2, y2)` in **absolute pixels**. Normalised formats convert at
the adapter boundary, where image sizes are known.

`BoxAnnotation` optionally carries richer labels:

```python
BoxAnnotation(
    bbox=(10.0, 10.0, 90.0, 120.0),
    category_id=1,
    segmentation=[[10, 10, 90, 10, 90, 120]],      # polygons, absolute pixels
    keypoints=[(30.0, 40.0, 2), (50.0, 80.0, 1)],  # (x, y, visibility)
)
```

### Converting, validating, inspecting

```python
convert_dataset("coco", "instances.json", "yolo", "out/", image_root="images/")

from lofop.data import validate_dataset
report = validate_dataset(data)
print(report.errors)        # blocking: degenerate boxes, dangling categories, missing files
print(report.warnings)      # advisory

from lofop.data import compute_stats
stats = compute_stats(data)
print(stats.per_class_counts, stats.size_breakdown)

from lofop.data import visualize_dataset
visualize_dataset(data, "vis/", limit=20)     # ground-truth boxes drawn onto images
```

Validation returns a report rather than raising, so you can decide what is fatal. The
CLI equivalent exits 1 on errors, which is what you want in CI.

### Synthetic data for smoke tests

```python
from lofop.data.synthetic import generate_shapes_dataset
train = generate_shapes_dataset("/tmp/t", num_images=64, image_size=128, seed=1)
```

## 7. Training

The one-call path:

```python
from lofop import Detector

det = Detector("lofop-detect-s", num_classes=2, class_names=["cat", "dog"])
metrics = det.train(
    data_format="coco",
    train_source="train.json",
    val_source="val.json",
    image_root="images/",
    epochs=100,
    batch_size=16,
    lr=0.01,
    checkpoint_dir="runs/train",
    strong_augment=True,
)
print(metrics.map50, metrics.map50_95)
```

Or from canonical datasets you already have:

```python
det.train(train_data=train, val_data=val, epochs=50)
```

After training, the detector holds the trained EMA weights and is immediately usable.
Checkpoints are written atomically to `checkpoint_dir` as `best.pt` and `last.pt`.

### Everything you can pass

`train` forwards unknown keyword arguments straight to `Trainer`:

```python
det.train(
    train_data=train, val_data=val,
    epochs=300, batch_size=32, lr=0.01,
    optimizer="SGD",                 # or "AdamW"
    weight_decay=0.0005,
    warmup_epochs=3.0,
    scheduler="warmup_cosine",       # warmup_linear | constant | step
    scheduler_kwargs={},
    amp=True,                        # mixed precision
    workers=8,                       # dataloader workers
    patience=30, min_delta=0.001,    # early stopping
    device="cuda",
)
```

### Augmentation

`strong_augment=True` enables LOFOP's original recipe: a 2&times;2 mosaic combining four
images plus colour jitter, on top of the default horizontal flip. Tensor-native, no
external augmentation library.

```python
det.train(train_data=train, epochs=200, strong_augment=True)
```

Recommended for real-data runs. It is off by default so existing runs and benchmarks are
unchanged. It is **not** yet supported for segmentation or pose variants.

### Dropping to the Trainer

When you need full control:

```python
from lofop.training import Trainer, DetectionTorchDataset

train_ds = DetectionTorchDataset(train, image_size=640, augment=True, strong_augment=True)
val_ds = DetectionTorchDataset(val, image_size=640)

trainer = Trainer(det.model, train_ds, val_ds, epochs=100, batch_size=16, lr=0.01)
metrics = trainer.fit()
trainer.resume()        # continue from last.pt
```

`DetectionTorchDataset` also produces the task-variant targets:

```python
DetectionTorchDataset(data, include_masks=True)      # for -seg
DetectionTorchDataset(data, include_keypoints=True)  # for -pose
```

The SDK sets these automatically based on your model, so you rarely pass them yourself.

### Saving and loading

```python
det.save("my-model.pt")
later = Detector("lofop-detect-s", num_classes=2, checkpoint="my-model.pt")
```

`Detector` reads `best.pt`, `last.pt`, and bare `state_dict` files, preferring EMA
weights when present because they evaluate higher.

### A note on expectations

Training from scratch on a small dataset gives weak results &mdash; that is a property of
training from scratch, not of LOFOP. LOFOP does not yet publish pretrained backbone
weights, so plan for a substantial dataset and a real training budget, or narrow your
class count. Measured on the fixed synthetic benchmark (128px shapes, 30 CPU epochs):
mAP@50 0.92, best F1 0.90.

## 8. Evaluation and benchmarking

```python
metrics = det.evaluate(val_data, batch_size=8)

metrics.map50            # mAP at IoU 0.50
metrics.map50_95         # COCO primary metric
metrics.precision
metrics.recall
metrics.f1
metrics.per_class_ap50
metrics.per_class_precision
metrics.per_class_recall
metrics.confusion_matrix
metrics.confusion_classes
```

Evaluation follows the COCO protocol. Note a current limit: the evaluator scores
**boxes only**. Mask mAP and keypoint OKS are not implemented yet, so segmentation and
pose quality cannot be measured natively &mdash; this is on the roadmap and this book will
not pretend otherwise.

### Benchmarking models

```python
from lofop.utils import benchmark_model, render_table, write_reports

result = benchmark_model(det.model, image_size=640)
print(render_table([result]))          # params, FLOPs, size, CPU/GPU FPS, peak memory
write_reports([result], "benchmarks/results")   # .md, .csv, .json
```

The table renders `-` for anything not actually measured. LOFOP never prints a number it
did not measure, which is why accuracy columns stay empty until you supply a trained
checkpoint.

---

# Part III &mdash; Task variants

## 9. Instance segmentation

```python
det = Detector("lofop-detect-n-seg", num_classes=80)
det.train(data_format="coco", train_source="instances.json", image_root="images/", epochs=100)

hit = det.predict("photo.jpg")[0]
print(hit.masks.shape)      # (K, H, W) bool tensor at the original image size
```

Masks come from **StencilHead**: a small tower on the finest pyramid level produces a
shared bank of 16 prototype masks, and a per-location coefficient tower predicts how to
combine them. A detection's mask is `sigmoid(coefficients · prototypes)` cropped to its
box, so mask cost grows with the number of detections, not with the number of pyramid
locations.

Training reads polygons from your COCO annotations automatically. If an annotation has
no polygon, its box is used as a coarse mask fallback, so a box-only dataset still
trains without crashing.

```python
import numpy as np
mask = hit.masks[0].numpy()
overlay = image * 0.5 + mask[..., None] * np.array([0, 255, 0]) * 0.5
```

## 10. Pose estimation

```python
det = Detector("lofop-detect-n-pose", num_classes=1, num_keypoints=17)
det.train(data_format="coco", train_source="person_keypoints.json",
          image_root="images/", epochs=200)

hit = det.predict("person.jpg")[0]
print(hit.keypoints[0])     # [[x, y, visibility], ...] in original image pixels
```

**VertexHead** regresses, at every pyramid location and for each keypoint, an `(x, y)`
offset from the location centre plus a visibility logit. A detection's skeleton is read
off at its winning location &mdash; no heatmaps, no second-stage crop.

`num_keypoints` defaults to 17 (the COCO person skeleton) and is configurable:

```python
Detector("n-pose", num_classes=1, num_keypoints=5)
```

Visibility comes back as a probability in `[0, 1]`:

```python
for x, y, visibility in hit.keypoints[0].tolist():
    if visibility > 0.5:
        draw_point(x, y)
```

**Known limit:** horizontal-flip augmentation is disabled for pose training, because
flipping requires swapping left/right keypoint pairs and that mapping is not implemented
yet. Pose runs therefore see less augmentation than detection runs.

## 11. NMS-free detection

This is new in 1.2.2 and it is worth understanding what it changes.

A dense head predicts at every pyramid location, so many locations fire on one object and
a suppression pass is needed to clean up. **SlateHead** instead keeps a fixed slate of
100 query slots. During training, `PivotMatcher` assigns each ground-truth object to
exactly **one** slot. The slate therefore learns to divide the image between its slots,
and at inference the model emits one box per object and **runs no suppression at all**
&mdash; no IoU pass, no score decay, no peak test. `predict` thresholds and sorts.

```python
det = Detector("lofop-detect-q-n", num_classes=80)
det.train(train_data=train, val_data=val, epochs=300)

hit = det.predict("photo.jpg")[0]
print(hit.boxes)            # already duplicate-free
```

The API is identical to any other variant. Nothing about your calling code changes.

### Why you would choose it

- **Post-processing cost stops depending on scene density.** Greedy NMS is quadratic in
  surviving candidates; a crowded frame costs more than an empty one. Peak selection is a
  fixed tensor operation.
- **It exports cleanly**, in principle, because there is no data-dependent loop &mdash;
  though ONNX export for this family is not implemented yet (see below).
- **No IoU threshold to tune.** There is no `nms_iou` because there is no NMS.

### Why you might not

- **Slower convergence.** Set-prediction heads need more epochs than dense heads. Budget
  for it.
- **The slate caps detections per image.** 100 slots is ample for ordinary scenes; raise
  it for dense crowds:
  ```python
  from lofop.core.config import Config
  from lofop.registries import HUB
  cfg = Config.load("lofop/configs/lofop-detect/q-s.yaml", resolve=False)
  cfg.num_classes = 80
  cfg.model.head.num_queries = 300
  cfg.resolve()
  det = Detector(HUB.build(cfg.model))
  ```
- **`nms_mode` is rejected** with a clear error, because it is meaningless here.
- **ONNX and TensorRT export raise a clear error** for this family today.

### Seeding the slate

If you know your dataset's box distribution, you can start the slots somewhere sensible
instead of at a random spread:

```python
import torch
priors = torch.tensor([[0.5, 0.5, 0.2, 0.3]] * 100)   # normalised cxcywh
det.model.head.initialise_reference_from(priors)
```

### The claim, tested

LOFOP's test suite trains `q-n` on two well-separated objects and asserts the model
returns **exactly two boxes with a maximum pairwise IoU below 0.3**. With no suppression
running, a duplicate slot would survive into the output, so this test fails loudly if the
one-to-one matching ever stops working.

## 12. Tracking

```bash
pip install "lofop[tracking]"
```

```python
det = Detector("lofop-detect-n", num_classes=80)

for frame in det.track(frame_paths):
    print(frame.tracker_ids, frame.boxes)
```

`track` is a **generator**, yielding one `Detections` per frame with `tracker_ids`
populated. It accepts any iterable of frames &mdash; a list of paths, a directory listing,
or a generator from your own video reader. LOFOP does not decode video itself, by design:
you keep whatever reader you already use.

```python
import cv2
def frames(path):
    capture = cv2.VideoCapture(path)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()

for tracked in det.track(frames("video.mp4")):
    ...
```

### How it works

Three layers, each usable alone:

1. **`MotionEstimator`** &mdash; a constant-velocity Kalman filter over
   `[cx, cy, w, h]` and their four velocities. Width and height are estimated
   independently rather than through an aspect-ratio term, so a box that changes shape is
   modelled as two ordinary velocities instead of one ratio whose derivative is
   ill-conditioned for short boxes. Noise scales with the box's geometric mean side.
2. **Association** &mdash; `1 - IoU` cost matrices built on LOFOP's native C++/CUDA IoU
   kernel, matched by LOFOP's own exact assignment solver. This layer needs nothing
   beyond the LOFOP core: no numpy, no scipy.
3. **`LofopTracker`** &mdash; confidence-tiered matching plus track lifecycle.

### Confidence tiers

Detectors emit a long tail of low-confidence boxes. Most are noise, but some are real
objects that briefly became hard to see. LOFOP matches in two tiers per frame:

- **Strong tier** &mdash; confident detections match against every live track. They can
  both continue and open tracks.
- **Weak tier** &mdash; low-confidence detections match only against tracks the strong
  tier left unmatched. They can **sustain** a track through a difficult moment but can
  never **start** one, so the noise tail never manufactures identities.

This is why `score_threshold` defaults to `0.1` in `track`, below `strong_threshold`: the
weak tier needs that tail as input. Raise the detector threshold and you disable the
mechanism.

### Tuning

```python
det.track(
    frames,
    score_threshold=0.1,      # detector threshold; keep below strong_threshold
    strong_threshold=0.5,     # at or above: can open a track
    weak_threshold=0.1,       # above this but below strong: can only continue one
    max_cost=0.8,             # association ceiling as 1 - IoU (0.8 accepts 20% overlap)
    grace_frames=30,          # frames a track survives unmatched before ending
    confirm_after=3,          # matching frames before a track is reported
    class_aware=True,         # forbid associations across classes
)
```

`grace_frames` scales with frame rate; 30 suits about 30 fps. Raise `confirm_after` if
detector flicker is creating spurious tracks; lower it if tracks start too late.

### Track lifecycle

```
Tentative  ->  Active  ->  Dormant  ->  Ended
```

A new track is **Tentative** until confirmed on `confirm_after` frames, which suppresses
one-frame flickers. An unmatched confirmed track becomes **Dormant** and keeps being
predicted forward, so a reappearing object recovers its original id. After
`grace_frames` it **Ends** and is never revived. Only **Active** tracks are returned.

### Using the tracker directly

```python
from lofop.tracking import LofopTracker

tracker = LofopTracker(confirm_after=2, grace_frames=45)
for boxes, scores, labels in my_detections:
    for track in tracker.update(boxes, scores, labels):
        print(track.track_id, track.box, track.phase, track.hits, track.age)

tracker.reset()
```

Or pass a preconfigured one into the SDK:

```python
det.track(frames, tracker=LofopTracker(class_aware=False))
```

### The layers on their own

```python
from lofop.tracking import associate, iou_cost_matrix

pairs, unmatched_tracks, unmatched_detections = associate(
    track_boxes, detection_boxes,
    track_labels=[0, 1], detection_labels=[1, 0],
    max_cost=0.7,
)
```

```python
from lofop.tracking import MotionEstimator

estimator = MotionEstimator(position_noise=0.055, velocity_noise=0.0095)
mean, covariance = estimator.initiate([100.0, 100.0, 40.0, 60.0])   # cx, cy, w, h
mean, covariance = estimator.predict(mean, covariance)
mean, covariance = estimator.correct(mean, covariance, [105.0, 100.0, 40.0, 60.0])
distances = estimator.residual_distance(mean, covariance, candidate_measurements)
```

Raise `measurement_noise` for a jittery detector; raise `velocity_noise` for erratic
motion.

## 13. The supervision bridge

```bash
pip install "lofop[supervision]"
```

```python
hit = det.predict("photo.jpg")[0]
sv_detections = hit.to_supervision()

import supervision as sv
annotated = sv.BoxAnnotator().annotate(scene=image, detections=sv_detections)
```

Boxes, scores, labels, masks and tracker ids all cross over. Coming back:

```python
from lofop.tracking import from_supervision
lofop_detections = from_supervision(sv_detections)
```

LOFOP vendors no supervision code and uses only its documented public API, so the two
projects stay independently licensed and independently versioned.

```python
from lofop.tracking import tracks_to_detections
detections = tracks_to_detections(tracker.update(boxes, scores, labels))
```

---

# Part IV &mdash; Production

## 14. Export and deployment

```python
det.export("model.onnx")                                  # verified ONNX
det.export("model.onnx", dynamic=True)                    # variable input sizes
det.export("model.engine", format="tensorrt", fp16=True)  # NVIDIA GPU
```

ONNX export is **numerically verified** against the torch model at export time, and
dynamic exports are verified at two resolutions. An export that would silently diverge
fails instead.

Export covers dense detection models. Segmentation, pose, and the query variants raise a
clear error today rather than producing a broken graph.

### Torch-free serving

The exported graph gives you dense `(boxes, scores)`. Finishing the job needs no PyTorch:

```python
from lofop.deploy import postprocess_dense

boxes, scores = session.run(None, {"images": batch})      # onnxruntime
detections = postprocess_dense(
    boxes[0], scores[0],
    score_threshold=0.25,
    nms_iou=0.6,
    max_detections=300,
    soft=False,           # True for Soft-NMS in crowded scenes
    soft_sigma=0.5,
)
```

A serving host needs only a runtime plus the LOFOP core &mdash; no torch, no numpy. If the
native ops are compiled there, this path uses the C++ (or CUDA) kernels automatically.

### CPU inference optimisation

```python
det.optimize()      # eval mode + channels_last
```

Measured 1.6&times; forward speedup at 640px on CPU, neutral at 128px, outputs identical
to float tolerance.

## 15. The torch-free runtime

Chapter 14 exported a graph. This chapter runs it in production, with **no
PyTorch installed at all**.

`lofop.runtime` is a separate, minimal inference path: it loads an exported
ONNX model, preprocesses images through LOFOP's native C++ kernels, and
post-processes with the same ops the training path uses. The dependency
footprint is the LOFOP core plus an ONNX runtime.

```python
from lofop.runtime import Detector

det = Detector(
    "model.onnx",
    score_threshold=0.25,
    nms_iou=0.6,
    max_detections=300,
    class_names=["cat", "dog"],
    soft_nms=False,
)

for hit in det.predict("photo.jpg"):
    print(hit.box, hit.score, hit.label, hit.name)
```

Note the different result shape from `lofop.Detector`: the runtime returns a
**list of individual `Detection` records**, each with `box`, `score`, `label`
and `name`, rather than one batched `Detections` per image. The runtime is
built for serving one image at a time, where per-detection records are what
callers want.

```python
@dataclass
class Detection:
    box: list[float]     # xyxy in original image pixels
    score: float
    label: int
    name: str | None     # resolved from class_names when given
```

### Choosing an execution provider

```python
det = Detector("model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
```

### Letterboxing

The runtime preprocesses with **letterbox** resizing, which preserves aspect
ratio by padding rather than stretching, and maps boxes back afterwards:

```python
from lofop.ops.preprocess import letterbox, unletterbox_boxes

tensor, meta = letterbox(pixels, width, height, size=640, pad_value=0.447)
boxes = unletterbox_boxes(boxes, meta)
```

`letterbox` runs on the native C++ kernel when it is built and falls back to
Python otherwise, like every other LOFOP op.

This is worth knowing even if you never call it directly: the training path
in Part II still resizes to a square and distorts aspect ratio, so a model
trained through `Detector.train` and served through `lofop.runtime` sees
slightly different geometry. Until letterboxing reaches the training path,
prefer matching the two if you are measuring accuracy carefully.

### Images without Pillow

```python
from lofop.runtime import Image

image = Image(width=640, height=480, channels=3, pixels=raw_bytes)
detections = det.predict(image)
```

### The C++ SDK

The same runtime exists as a C++ library for hosts with no Python at all. It
links the same `lofop/csrc` kernels, so preprocessing and post-processing are
bit-identical to the Python path. See [`docs/cpp-sdk.md`](docs/cpp-sdk.md).

### Which path should you use

| | `lofop.Detector` | `lofop.runtime.Detector` |
|---|---|---|
| Needs PyTorch | yes | **no** |
| Trains | yes | no |
| Input | checkpoints and configs | exported ONNX |
| Batching | yes | one image at a time |
| Returns | `list[Detections]` | `list[Detection]` |
| Preprocessing | square resize | letterbox |
| Use for | development, training, evaluation | serving, edge devices |

## 16. Experiment tracking

```python
from lofop.mlops import track

with track("runs/registry", name="baseline-s", tags=["coco", "s"], meta={"lr": 0.01}):
    det.train(train_data=train, val_data=val, epochs=100)
```

Every run is recorded as plain JSON through the event bus: settings, environment
(LOFOP/torch/CUDA versions), per-epoch history, best and final metrics, and checkpoint
paths. No server, no account, no lock-in. A crashed run is recorded with status
`failed` rather than vanishing.

```python
from lofop.mlops import list_runs, load_run, compare_runs

for run in list_runs("runs/registry"):                # newest first
    print(run["run_id"], run["status"], run["best_map50"])

record, history = load_run("runs/registry", run_id)
print(compare_runs("runs/registry", [run_a, run_b]))  # markdown table
```

These query functions are torch-free, so you can inspect runs from anywhere.

```bash
lofop runs list --root runs/registry
lofop runs show <run_id> --root runs/registry
lofop runs compare <id_a> <id_b> --root runs/registry
```

## 17. The ops layer

LOFOP's performance-critical primitives have three tiers, selected fastest-first:

```
cuda    ->  NVIDIA GPU        (optional, needs nvcc)
native  ->  C++, any CPU      (optional, needs a compiler)
python  ->  always available  (no build required)
```

```python
from lofop.ops import backend, build_native
print(backend())              # 'cuda' | 'native' | 'python'

build_native()                # build the C++ tier
build_native(cuda=True)       # build the CUDA tier (needs nvcc)
```

Every tier produces **identical results**; the test suite asserts it. If a tier is
missing or a CUDA call fails, LOFOP falls back silently and correctly.

```python
from lofop.ops import iou_matrix, nms, batched_nms, soft_nms, decode_dense

iou_matrix(boxes_a, boxes_b)
nms(boxes, scores, iou_threshold=0.5, max_keep=300)
batched_nms(boxes, scores, class_ids, iou_threshold=0.5)
keep, decayed = soft_nms(boxes, scores, sigma=0.5, method="gaussian")
indices, labels, best = decode_dense(class_scores, score_threshold=0.25)
```

Measured speedups, C++ versus pure Python: NMS 38&ndash;145&times;, Soft-NMS 23&ndash;80&times;,
IoU 22&ndash;25&times;, dense decode 35&ndash;48&times; on the zero-copy numpy path.

One honest caveat on `decode_dense`: for nested Python lists, copying into C memory costs
more than the Python loop saves, so the native kernel is auto-selected only for
contiguous float32 numpy arrays or CPU tensors &mdash; which is the deployment case that
matters.

### The assignment solver

```python
from lofop.ops.assignment import optimal_assignment, greedy_assignment

optimal_assignment([[4, 1, 3], [2, 0, 5], [3, 2, 2]])   # [1, 0, 2], total cost 5
greedy_assignment(cost, max_cost=0.8)                    # cheaper, can be worse
```

`optimal_assignment` solves the assignment problem exactly by shortest augmenting paths
over dual potentials, in pure Python with no scipy. It is verified against brute force
across randomised matrices. Rows and columns may differ in count; unassigned rows return
`-1`.

---

# Part V &mdash; Extending LOFOP

## 18. Configs and registries

LOFOP is config-driven. Anything executable is a registered component; YAML holds only
data.

```python
from lofop import Config, HUB
import lofop.models

cfg = Config.load("lofop/configs/lofop-detect/s.yaml", resolve=False)
cfg.num_classes = 12
cfg.resolve()
model = HUB.build(cfg.model)
```

Configs support inheritance and interpolation:

```yaml
extends: [base.yaml]
num_classes: 12

model:
  head:
    num_classes: ${num_classes}     # interpolated at resolve()
    width: 96
```

Cross-group references are explicitly qualified, so a plain dict that happens to have a
`type` key is never instantiated by surprise:

```yaml
backbone: {type: backbone/RidgeNet, widths: [48, 96, 192, 384]}
neck:     {type: neck/DeltaFusion, width: 96}
head:     {type: head/ApexHead, num_classes: 12}
```

The standard component groups: `model`, `backbone`, `neck`, `head`, `loss`, `metric`,
`optimizer`, `scheduler`, `dataset`, `transform`, `hook`.

```python
cfg.freeze()        # make a config read-only
cfg.to_dict()
```

## 19. Writing your own components

Register a component and it is immediately usable from YAML. No framework changes.

```python
from torch import nn
from lofop.registries import BACKBONES

@BACKBONES.register()
class MyNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        ...

    def forward(self, x):
        return [c3, c4, c5]       # the backbone contract: three levels
```

```yaml
backbone:
  type: backbone/MyNet
  widths: [32, 64, 128, 256]
```

The contracts your component must honour:

| Group | Contract |
|---|---|
| backbone | `forward(images) -> [C3, C4, C5]` at strides 8/16/32 |
| neck | `forward([C3, C4, C5]) -> [P3, P4, P5]`, all at one width |
| head | `forward([P3, P4, P5]) -> ` whatever your model consumes |
| model | `compute_losses(images, targets) -> dict` with `"total"`; `predict(images) -> list[dict]` |

A custom head, using the same pattern the built-in ones use:

```python
from lofop.registries import HEADS

@HEADS.register()
class MyHead(nn.Module):
    def __init__(self, num_classes, width=96):
        super().__init__()
        self.num_classes = num_classes
        ...
```

This is exactly how the 1.2.2 task variants were added: `SlateHead`, `StencilHead`,
`VertexHead` and `SwiftNet` are registrations plus YAML, with no changes to the core
engine.

## 20. Events and hooks

The `Trainer` emits events; subscribers attach without touching it.

| Topic | Fired when | Payload |
|---|---|---|
| `train.start` | training begins | `epochs`, `device` |
| `train.epoch_end` | each epoch ends | `epoch`, `loss`, `metrics` |
| `eval.end` | evaluation completes | `metrics` |
| `checkpoint.saved` | a checkpoint is written | `epoch`, `path` |
| `train.end` | training finishes | `metrics` |

```python
from lofop import EVENTS

def on_epoch(epoch, loss, metrics, **kwargs):
    print(f"epoch {epoch}: loss {loss:.4f} mAP50 {metrics.get('map50', 0):.4f}")

EVENTS.subscribe("train.epoch_end", on_epoch)
```

```python
from lofop.training import attach_tensorboard
attach_tensorboard("runs/tb")        # needs lofop[tensorboard]
```

This is how `lofop.mlops` works, and how you should build anything similar: no trainer
subclassing, no monkey-patching.

---

# Part VI &mdash; Reference

## 21. The CLI

Everything in this book has a command-line equivalent.

```
lofop {version, dataset, train, benchmark, predict, evaluate, runs, doctor, export}
```

```bash
lofop version
lofop doctor                       # environment + active ops backend

lofop dataset convert  --from coco --source instances.json --to yolo --target out/
lofop dataset validate --format yolo --source out/           # exit 1 on errors
lofop dataset stats    --format coco --source instances.json -o stats.md
lofop dataset show     --format coco --source instances.json -o vis/ --limit 10

lofop train    --config configs/train/example.yaml
lofop train    --config configs/train/example.yaml --resume

lofop predict  --config n --checkpoint best.pt --source image.png
lofop evaluate --config n --checkpoint best.pt --format coco --source val.json

lofop benchmark --config lofop/configs/lofop-detect/n.yaml --results-dir results/
lofop export    --config lofop/configs/lofop-detect/n.yaml --checkpoint best.pt -o model.onnx

lofop runs list --root runs/registry
```

The dataset commands need no PyTorch, so they run on an edge box or in CI.

## 22. Cookbook

**Detect on a folder of images**

```python
from pathlib import Path
paths = sorted(Path("images").glob("*.jpg"))
for path, hit in zip(paths, det.predict(paths)):
    print(path.name, len(hit))
```

**Filter to one class above a confidence**

```python
person = det.class_names.index("person")
keep = [(b, s) for b, s, label in zip(hit.boxes, hit.scores, hit.labels)
        if label == person and s > 0.5]
```

**Count unique objects in a video**

```python
seen = set()
for frame in det.track(frames):
    seen.update(frame.tracker_ids)
print(f"{len(seen)} distinct objects")
```

**Fine-tune from an existing checkpoint**

```python
det = Detector("lofop-detect-s", num_classes=3, checkpoint="runs/train/best.pt")
det.train(train_data=new_data, epochs=50, lr=0.002)      # lower LR for fine-tuning
```

The variant and class count must match the checkpoint.

**Track a specific class only**

```python
for frame in det.track(frames):
    cars = [(i, b) for i, b, label in
            zip(frame.tracker_ids, frame.boxes, frame.labels) if label == car_id]
```

**Compare two training runs**

```python
from lofop.mlops import track, list_runs, compare_runs

for lr in (0.01, 0.005):
    with track("runs/registry", name=f"lr-{lr}", meta={"lr": lr}):
        Detector("s", num_classes=2).train(train_data=train, val_data=val, lr=lr, epochs=50)

ids = [r["run_id"] for r in list_runs("runs/registry")[:2]]
print(compare_runs("runs/registry", ids))
```

**Run entirely without PyTorch at serving time**

```python
from lofop.deploy import postprocess_dense
boxes, scores = onnx_session.run(None, {"images": batch})
detections = postprocess_dense(boxes[0], scores[0], score_threshold=0.3)
```

**Build the native ops in a Dockerfile**

```dockerfile
RUN pip install lofop && \
    python -c "from lofop.ops import build_native; build_native()"
```

## 23. Troubleshooting

**`LofopError: ... pip install "lofop[tracking]"`**
Working as designed. Install the named extra.

**`Unknown model 'lofop-detect-x'`**
The error lists every valid variant. See chapter 5.

**`nms_mode does not apply to the NMS-free query variants`**
Query variants never suppress. Drop the argument, or use a dense variant.

**`ONNX export does not yet cover the NMS-free query variants`**
Accurate, not a bug. Export a dense variant for now.

**`class_names length must match num_classes`**
Exactly what it says; the error reports both numbers.

**Training is slow.** Check `workers` (default 2 &mdash; raise it), whether `amp=True`,
and how large your validation set is, since it is evaluated every epoch. Variant `ex` is
roughly 7&times; the cost of `n`.

**Predictions are duplicated.** Lower `nms_iou`, try `nms_mode="soft"`, or move to a
query variant which cannot duplicate by construction.

**Everything is detected as one class.** A single-class model detects one class. Train
with the class count you actually need.

**mAP is near zero from scratch.** Expected on small datasets without pretrained weights.
Use more data, fewer classes, more epochs, and `strong_augment=True`.

**`backend()` says `python`.** The native library is not built. Run
`build_native()`. Results stay correct either way &mdash; only speed changes.

## 24. API reference

### lofop

`Config`, `HUB`, `EVENTS`, `PLUGINS`, `LofopError`, `get_logger`, `configure_logging`

### lofop.sdk

```python
Detector(model="lofop-detect-s", *, num_classes=80, checkpoint=None, class_names=None,
         image_size=640, device=None, num_keypoints=None, nms_mode=None)

.predict(source, *, score_threshold=None) -> list[Detections]
.track(source, *, score_threshold=0.1, strong_threshold=0.5, weak_threshold=0.1,
       max_cost=0.8, grace_frames=30, confirm_after=3, class_aware=True,
       tracker=None) -> Iterator[Detections]
.train(*, train_data=None, val_data=None, data_format=None, train_source=None,
       val_source=None, image_root=None, epochs=100, batch_size=16, lr=0.01,
       checkpoint_dir="runs/train", strong_augment=False, **trainer_kwargs)
.evaluate(val_data, *, batch_size=8) -> DetectionMetrics
.export(path, *, format=None, **kwargs) -> Path
.save(path) -> Path
.optimize() -> Detector
.num_parameters -> int
```

### lofop.data

`Dataset`, `Sample`, `BoxAnnotation`, `Category`, `load_dataset`, `save_dataset`,
`convert_dataset`, `get_adapter`, `FORMATS`, `DatasetAdapter`, `validate_dataset`,
`ValidationReport`, `Issue`, `Severity`, `compute_stats`, `DatasetStats`, `draw_boxes`,
`render_sample`, `visualize_dataset`

### lofop.models

`RidgeNet`, `SwiftNet`, `InvertedResidual`, `DeltaFusion`, `ApexHead`, `StencilHead`,
`VertexHead`, `SlateHead`, `LofopDetect`, `LofopSegment`, `LofopPose`, `LofopQuery`,
`DynamicTopKAssigner`, `PivotMatcher`, `sigmoid_focal_loss`, `giou_loss`, `pairwise_iou`

### lofop.training

`Trainer`, `DetectionTorchDataset`, `detection_collate`, `image_to_tensor`, `ModelEMA`,
`CheckpointManager`, `DetectionMetrics`, `evaluate_detections`, `build_scheduler`,
`warmup_cosine_lr`, `TensorBoardHook`, `attach_tensorboard`

### lofop.tracking

`MotionEstimator`, `LofopTracker`, `Tracklet`, `TrackPhase`, `track_detections`,
`associate`, `iou_cost_matrix`, `match`, `to_centre_size`, `to_corners`,
`to_supervision`, `from_supervision`, `tracks_to_detections`

### lofop.ops

`iou_matrix`, `nms`, `batched_nms`, `soft_nms`, `decode_dense`, `backend`,
`build_native`, `find_library`, `load_native`, `load_native_cuda`,
`assignment.optimal_assignment`, `assignment.greedy_assignment`

### lofop.deploy

`Detections`, `postprocess_dense`, `export_onnx`, `export_tensorrt`,
`DenseExportWrapper`, `build_engine_from_onnx`

### lofop.runtime

`Detector`, `Detection`, `Image` &mdash; the torch-free ONNX serving path.
Preprocessing helpers live in `lofop.ops.preprocess`: `letterbox`,
`unletterbox_boxes`, `LetterboxMeta`.

### lofop.mlops

`track`, `RunTracker`, `RunRecord`, `list_runs`, `load_run`, `compare_runs`

### Exceptions

All inherit from `LofopError` and carry a `context` dict: `ModelError`, `DataError`,
`ConfigError`, `RegistryError`, `BuildError`.

```python
from lofop import LofopError
try:
    det = Detector("nonexistent")
except LofopError as error:
    print(error)            # message
    print(error.context)    # structured detail, e.g. the list of valid variants
```

---

## What is not done yet

Stated plainly, so you can plan around it:

- **No published pretrained checkpoints.** Every model trains from scratch today. This is
  LOFOP's largest gap and its top roadmap item.
- **Letterboxing is serving-only.** `lofop.runtime` letterboxes, but the
  training path still resizes to a square, so the two see different geometry.
- **Box-only evaluation.** No mask mAP, no keypoint OKS.
- **Export gaps.** ONNX and TensorRT cover dense detection only.
- **Pose flip augmentation disabled**, pending left/right keypoint pair mapping.
- **`strong_augment` unsupported** for segmentation and pose.
- **CUDA ops tier and the DDP training path** are implemented and guarded but have not
  been exercised on real GPU hardware.

---

*LOFOP is developed by DURGAMANI SASIKUMAR and Nishanandhini A. (Assistant Professor).
Apache-2.0. Source, issues and the changelog:
[github.com/tedo001/LOFOP](https://github.com/tedo001/LOFOP).*
