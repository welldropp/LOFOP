"""Box operations: IoU and non-maximum suppression.

These are the CPU-side hot path of detection post-processing. A multi-scale
detector predicts candidates for tiny objects on fine pyramid levels and huge
objects on coarse levels; after score filtering, thousands of overlapping
candidates per image reach NMS. Each op therefore has two implementations:

* a pure Python reference (always available, used for verification), and
* the native C++ kernel from ``lofop/csrc`` (used automatically when built --
  see :mod:`lofop.ops.native`).

Both produce identical results; tests assert it. Inputs are any sequences of
floats -- ``list``s, ``array``s, numpy arrays -- with boxes in absolute xyxy.
"""

from __future__ import annotations

import ctypes
import math
from collections.abc import Sequence
from typing import Any

from lofop.core.exceptions import LofopError
from lofop.ops.native import load_native, load_native_cuda

BoxLike = Sequence[Sequence[float]]
IouMatrix = list[list[float]]

# Class-aware NMS separates classes by shifting each class onto its own
# coordinate island; the offset just needs to exceed any real coordinate.
# Public so callers doing the shift themselves (e.g. in tensor math, to hit
# the zero-copy nms path) stay consistent with batched_nms.
CLASS_OFFSET = 1e7

# Below this element count the host<->device copies cost more than the GPU
# saves, so the CUDA tier only engages on larger problems.
_CUDA_MIN_ELEMENTS = 1 << 16


def _parse_boxes(boxes: BoxLike) -> list[tuple[float, float, float, float]]:
    parsed = []
    for box in boxes:
        if len(box) != 4:
            raise LofopError("Boxes must be xyxy quadruples", context={"got": list(box)})
        parsed.append(tuple(float(v) for v in box))
    return parsed


def _flatten_boxes(boxes: BoxLike) -> list[float]:
    return [v for box in _parse_boxes(boxes) for v in box]


def _c_float_view(obj: Any, count: int):
    """ctypes float pointer for ``obj``, zero-copy when possible.

    A contiguous float32 CPU tensor-like (duck-typed via ``data_ptr``) is
    viewed in place -- no per-element Python conversion -- which is what makes
    the native NMS path fast on detector outputs. Anything else is copied
    through the generic list path. Returns ``(pointer, keepalive)``; the
    keepalive must stay referenced while the pointer is in use.
    """
    if callable(getattr(obj, "data_ptr", None)):
        try:
            zero_copy = (
                str(obj.dtype) == "torch.float32"
                and obj.device.type == "cpu"
                and obj.is_contiguous()
                and obj.numel() == count
            )
        except AttributeError:
            zero_copy = False
        if zero_copy:
            return ctypes.cast(obj.data_ptr(), ctypes.POINTER(ctypes.c_float)), obj
        flat = obj.detach().reshape(-1).tolist()
        array = (ctypes.c_float * count)(*flat)
        return array, array
    if hasattr(obj, "__array_interface__"):
        # numpy-like (e.g. onnxruntime outputs): view a contiguous float32
        # buffer in place, same zero-copy contract as the tensor path.
        iface = obj.__array_interface__
        try:
            zero_copy = (
                iface.get("typestr") in ("<f4", "|f4")
                and iface.get("strides") is None
                and obj.size == count
            )
        except AttributeError:
            zero_copy = False
        if zero_copy:
            return ctypes.cast(iface["data"][0], ctypes.POINTER(ctypes.c_float)), obj
        array = (ctypes.c_float * count)(*[float(v) for v in obj.reshape(-1)])
        return array, array
    return None, None


def iou_matrix(boxes_a: BoxLike, boxes_b: BoxLike, *, native: bool | None = None) -> IouMatrix:
    """Pairwise IoU between two box sets.

    Args:
        boxes_a: N boxes, xyxy.
        boxes_b: M boxes, xyxy.
        native: Force the native (``True``) or Python (``False``) path;
            ``None`` auto-selects native when built.

    Returns:
        N x M nested lists of IoU values in ``[0, 1]``.
    """
    lib = load_native() if native in (None, True) else None
    if native is True and lib is None:
        raise LofopError("Native ops library is not built; run lofop.ops.native.build_native()")
    n, m = len(boxes_a), len(boxes_b)
    if n == 0 or m == 0:
        return [[0.0] * m for _ in range(n)]
    cuda = load_native_cuda() if native in (None, True) and n * m >= _CUDA_MIN_ELEMENTS else None
    if cuda is not None:
        a = (ctypes.c_float * (n * 4))(*_flatten_boxes(boxes_a))
        b = (ctypes.c_float * (m * 4))(*_flatten_boxes(boxes_b))
        out = (ctypes.c_float * (n * m))()
        if cuda.lofop_cuda_iou_matrix(a, n, b, m, out) == 0:
            return [list(out[i * m:(i + 1) * m]) for i in range(n)]
        # Any CUDA failure falls through to the C++/Python tiers.
    if lib is not None:
        a = (ctypes.c_float * (n * 4))(*_flatten_boxes(boxes_a))
        b = (ctypes.c_float * (m * 4))(*_flatten_boxes(boxes_b))
        out = (ctypes.c_float * (n * m))()
        lib.lofop_iou_matrix(a, n, b, m, out)
        return [list(out[i * m:(i + 1) * m]) for i in range(n)]
    parsed_a, parsed_b = _parse_boxes(boxes_a), _parse_boxes(boxes_b)
    return [[_pair_iou(pa, pb) for pb in parsed_b] for pa in parsed_a]


def nms(
    boxes: BoxLike,
    scores: Sequence[float],
    *,
    iou_threshold: float = 0.5,
    max_keep: int = 0,
    native: bool | None = None,
) -> list[int]:
    """Greedy non-maximum suppression.

    Args:
        boxes: N boxes, xyxy.
        scores: N confidence scores.
        iou_threshold: Boxes overlapping a kept box above this are dropped.
        max_keep: Stop after keeping this many boxes (0 = no limit).
        native: Force the native (``True``) or Python (``False``) path;
            ``None`` auto-selects native when built.

    Returns:
        Indices of kept boxes, highest score first.
    """
    if len(boxes) != len(scores):
        raise LofopError(
            "boxes and scores must have equal length",
            context={"boxes": len(boxes), "scores": len(scores)},
        )
    n = len(boxes)
    if n == 0:
        return []
    lib = load_native() if native in (None, True) else None
    if native is True and lib is None:
        raise LofopError("Native ops library is not built; run lofop.ops.native.build_native()")
    if lib is not None:
        c_boxes, keep_boxes = _c_float_view(boxes, n * 4)
        if c_boxes is None:
            c_boxes = (ctypes.c_float * (n * 4))(*_flatten_boxes(boxes))
            keep_boxes = c_boxes
        c_scores, keep_scores = _c_float_view(scores, n)
        if c_scores is None:
            c_scores = (ctypes.c_float * n)(*[float(s) for s in scores])
            keep_scores = c_scores
        keep = (ctypes.c_int32 * n)()
        kept = lib.lofop_nms(c_boxes, c_scores, n, float(iou_threshold), int(max_keep), keep)
        del keep_boxes, keep_scores  # buffers alive through the native call
        return list(keep[:kept])
    if hasattr(boxes, "tolist"):
        boxes, scores = boxes.tolist(), [float(s) for s in scores]
    return _nms_python(boxes, scores, iou_threshold, max_keep)


def batched_nms(
    boxes: BoxLike,
    scores: Sequence[float],
    class_ids: Sequence[int],
    *,
    iou_threshold: float = 0.5,
    max_keep: int = 0,
    native: bool | None = None,
) -> list[int]:
    """Class-aware NMS: boxes only suppress boxes of the same class.

    Implemented with the standard coordinate-offset trick, so one native NMS
    call handles all classes at once.
    """
    if not (len(boxes) == len(scores) == len(class_ids)):
        raise LofopError(
            "boxes, scores, and class_ids must have equal length",
            context={"boxes": len(boxes), "scores": len(scores), "classes": len(class_ids)},
        )
    shifted = [
        [float(v) + CLASS_OFFSET * int(cls) for v in box]
        for box, cls in zip(boxes, class_ids)
    ]
    return nms(shifted, scores, iou_threshold=iou_threshold, max_keep=max_keep, native=native)


def soft_nms(
    boxes: BoxLike,
    scores: Sequence[float],
    *,
    iou_threshold: float = 0.3,
    sigma: float = 0.5,
    score_threshold: float = 0.001,
    method: str = "gaussian",
    max_keep: int = 0,
    native: bool | None = None,
) -> tuple[list[int], list[float]]:
    """Soft non-maximum suppression: decay overlapping scores, don't drop.

    Where greedy :func:`nms` deletes every box overlapping a kept box, Soft-NMS
    multiplies its score down -- ``gaussian``: ``s *= exp(-iou^2 / sigma)``;
    ``linear``: ``s *= 1 - iou`` when ``iou > iou_threshold`` -- and only
    discards boxes whose decayed score falls below ``score_threshold``. This
    recovers true detections in crowded scenes that greedy NMS wrongly merges.

    Args:
        boxes: N boxes, xyxy.
        scores: N confidence scores.
        iou_threshold: Overlap above which the ``linear`` method decays
            (ignored by ``gaussian``).
        sigma: Gaussian decay width (ignored by ``linear``).
        score_threshold: Decayed boxes below this are discarded.
        method: ``"gaussian"`` or ``"linear"``.
        max_keep: Stop after keeping this many boxes (0 = no limit).
        native: Force the native (``True``) or Python (``False``) path;
            ``None`` auto-selects native when built.

    Returns:
        ``(keep, kept_scores)``: indices of kept boxes in selection order
        (decayed-score descending) and their final, decayed scores.
    """
    if len(boxes) != len(scores):
        raise LofopError(
            "boxes and scores must have equal length",
            context={"boxes": len(boxes), "scores": len(scores)},
        )
    if method not in ("gaussian", "linear"):
        raise LofopError("method must be 'gaussian' or 'linear'", context={"got": method})
    n = len(boxes)
    if n == 0:
        return [], []
    lib = load_native() if native in (None, True) else None
    if native is True and lib is None:
        raise LofopError("Native ops library is not built; run lofop.ops.native.build_native()")
    if lib is not None and hasattr(lib, "lofop_soft_nms"):
        c_boxes, keep_boxes = _c_float_view(boxes, n * 4)
        if c_boxes is None:
            c_boxes = (ctypes.c_float * (n * 4))(*_flatten_boxes(boxes))
            keep_boxes = c_boxes
        c_scores, keep_scores = _c_float_view(scores, n)
        if c_scores is None:
            c_scores = (ctypes.c_float * n)(*[float(s) for s in scores])
            keep_scores = c_scores
        keep = (ctypes.c_int32 * n)()
        out_scores = (ctypes.c_float * n)()
        kept = lib.lofop_soft_nms(
            c_boxes, c_scores, n, float(iou_threshold), float(sigma), float(score_threshold),
            1 if method == "gaussian" else 0, int(max_keep), keep, out_scores,
        )
        del keep_boxes, keep_scores  # buffers alive through the native call
        return list(keep[:kept]), list(out_scores[:kept])
    if hasattr(boxes, "tolist"):
        boxes, scores = boxes.tolist(), [float(s) for s in scores]
    parsed = _parse_boxes(boxes)
    live = [float(s) for s in scores]
    done = [False] * n
    keep_idx: list[int] = []
    kept_scores: list[float] = []
    while True:
        best, best_score = -1, score_threshold
        for i in range(n):
            if not done[i] and live[i] > best_score:
                best, best_score = i, live[i]
        if best < 0:
            break
        done[best] = True
        keep_idx.append(best)
        kept_scores.append(best_score)
        if max_keep > 0 and len(keep_idx) >= max_keep:
            break
        for j in range(n):
            if done[j]:
                continue
            iou = _pair_iou(parsed[best], parsed[j])
            if iou <= 0.0:
                continue
            if method == "gaussian":
                live[j] *= math.exp(-(iou * iou) / sigma)
            elif iou > iou_threshold:
                live[j] *= 1.0 - iou
    return keep_idx, kept_scores


def decode_dense(
    class_scores: Sequence[Sequence[float]],
    *,
    score_threshold: float = 0.25,
    native: bool | None = None,
) -> tuple[list[int], list[int], list[float]]:
    """Best-class decode of a dense (N, C) score matrix with thresholding.

    The first stage of torch-free deployment post-processing: for each of the
    N candidate locations, take the highest-scoring class and keep the
    candidate when that score clears ``score_threshold``. A contiguous float32
    numpy array or CPU tensor is passed to the native kernel zero-copy.

    Args:
        class_scores: (N, C) per-class scores (nested sequences, a numpy
            array, or a CPU tensor).
        score_threshold: Minimum best-class score to keep a candidate.
        native: Force the native (``True``) or Python (``False``) path;
            ``None`` auto-selects native when built.

    Returns:
        ``(indices, labels, scores)``: kept candidate row indices (ascending),
        their best class ids, and the corresponding scores.
    """
    n = len(class_scores)
    if n == 0:
        return [], [], []
    num_classes = len(class_scores[0])
    if num_classes == 0:
        return [], [], []
    lib = load_native() if native in (None, True) else None
    if native is True and lib is None:
        raise LofopError("Native ops library is not built; run lofop.ops.native.build_native()")
    count = n * num_classes
    if lib is not None and hasattr(lib, "lofop_decode_dense"):
        c_scores, keepalive = _c_float_view(class_scores, count)
        copied = c_scores is None or isinstance(c_scores, ctypes.Array)
        if copied and native is not True:
            # Measured: flattening N*C Python floats into a C buffer costs
            # more than the pure Python argmax loop, so the native kernel is
            # auto-selected only for zero-copy inputs (contiguous float32
            # numpy arrays / CPU tensors -- the deployment hot path, 35-48x).
            c_scores = None
        if c_scores is None and native is not True:
            lib = None
        elif c_scores is None:
            flat = [float(v) for row in class_scores for v in row]
            if len(flat) != count:
                raise LofopError("class_scores rows must have equal length")
            c_scores = (ctypes.c_float * count)(*flat)
            keepalive = c_scores
    if lib is not None and hasattr(lib, "lofop_decode_dense"):
        idx = (ctypes.c_int32 * n)()
        labels = (ctypes.c_int32 * n)()
        out_scores = (ctypes.c_float * n)()
        cuda = load_native_cuda() if native is None and count >= _CUDA_MIN_ELEMENTS else None
        kept = -1
        if cuda is not None:
            kept = cuda.lofop_cuda_decode_dense(
                c_scores, n, num_classes, float(score_threshold), idx, labels, out_scores,
            )
        if kept < 0:  # no CUDA tier, or it failed: use the C++ kernel
            kept = lib.lofop_decode_dense(
                c_scores, n, num_classes, float(score_threshold), idx, labels, out_scores,
            )
        del keepalive
        return list(idx[:kept]), list(labels[:kept]), list(out_scores[:kept])
    indices: list[int] = []
    labels_out: list[int] = []
    scores_out: list[float] = []
    for i, row in enumerate(class_scores):
        best, best_score = 0, float(row[0])
        for c, value in enumerate(row):
            if float(value) > best_score:
                best, best_score = c, float(value)
        if best_score > score_threshold:
            indices.append(i)
            labels_out.append(best)
            scores_out.append(best_score)
    return indices, labels_out, scores_out


def _pair_iou(a: tuple, b: tuple) -> float:
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _nms_python(boxes: BoxLike, scores: Sequence[float], iou_threshold: float,
                max_keep: int) -> list[int]:
    order = sorted(range(len(boxes)), key=lambda i: -float(scores[i]))
    parsed = _parse_boxes(boxes)
    suppressed = [False] * len(boxes)
    keep: list[int] = []
    for oi, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        if max_keep > 0 and len(keep) >= max_keep:
            break
        for j in order[oi + 1:]:
            if not suppressed[j] and _pair_iou(parsed[i], parsed[j]) > iou_threshold:
                suppressed[j] = True
    return keep
