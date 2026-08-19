"""LOFOP ops: performance-critical primitives with native acceleration.

Three tiers, auto-selected fastest-first (``cuda`` > ``native`` C++ >
``python``): every op has a pure Python reference implementation; compiling
``lofop/csrc`` (``build_native()``) adds a drop-in C++ fast path producing
identical results; and ``build_native(cuda=True)`` adds CUDA kernels for the
parallel ops on NVIDIA GPUs. :func:`backend` reports the active tier.
"""

from lofop.ops.boxes import batched_nms, decode_dense, iou_matrix, nms, soft_nms
from lofop.ops.native import backend, build_native, find_library, load_native, load_native_cuda
from lofop.ops.preprocess import LetterboxMeta, letterbox, unletterbox_boxes

__all__ = [
    "iou_matrix", "nms", "batched_nms", "soft_nms", "decode_dense",
    "letterbox", "unletterbox_boxes", "LetterboxMeta",
    "backend", "build_native", "find_library", "load_native", "load_native_cuda",
]
