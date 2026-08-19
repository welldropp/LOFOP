"""Torch-free ONNX inference for LOFOP detectors.

This is the Python half of LOFOP's deployment runtime. Its C++ twin
(``cpp/include/lofop/lofop.hpp``) exposes the same three types with the same
names and the same semantics, and both drive the *same* native kernels:
letterboxing from ``csrc/preprocess.cpp``, dense decoding and class-aware NMS
from ``csrc/box_ops.cpp``. Given one ``model.onnx`` the two runtimes return
identical detections, which is what makes "train in Python, serve in C++" a
guarantee rather than an aspiration.

It is entirely separate from :class:`lofop.Detector`, the PyTorch research and
training API. Nothing here imports torch, and nothing here changes that class::

    from lofop import Detector                 # PyTorch: build, train, export
    from lofop.runtime import Detector         # ONNX: deploy, no torch

Usage::

    from lofop.runtime import Detector

    model = Detector("model.onnx", class_names=["cat", "dog"])
    for hit in model.predict("image.jpg"):
        print(hit.name, hit.confidence, hit.box)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from lofop.core.exceptions import LofopError
from lofop.core.logging import get_logger
from lofop.deploy.postprocess import postprocess_dense
from lofop.ops import backend as ops_backend
from lofop.ops.preprocess import letterbox, unletterbox_boxes
from lofop.runtime.image import Image

logger = get_logger(__name__)

__all__ = ["Detection", "Detector"]

ImageSource = Union[str, Path, Image]

_DEFAULT_SIZE = 640


@dataclass(frozen=True)
class Detection:
    """One detected object, in original image coordinates.

    Attributes:
        box: ``(x1, y1, x2, y2)`` in the source image's own pixels.
        score: Confidence in ``[0, 1]``.
        label: Class index.
        name: Class name, or ``"class_<index>"`` when no names were given.
    """

    box: tuple[float, float, float, float]
    score: float
    label: int
    name: str

    @property
    def confidence(self) -> float:
        """Alias for :attr:`score`, matching the C++ SDK's field name."""
        return self.score

    def __repr__(self) -> str:
        coords = ", ".join(f"{value:.1f}" for value in self.box)
        return f"Detection({self.name} {self.score:.3f} [{coords}])"


class Detector:
    """Run a exported LOFOP detector on images, without PyTorch.

    Args:
        model: Path to a ``.onnx`` file produced by
            :func:`lofop.deploy.export_onnx`.
        score_threshold: Minimum class score for a candidate to survive.
        nms_iou: IoU threshold for class-aware NMS.
        max_detections: Cap on detections returned per image.
        class_names: Optional class-index to name mapping.
        image_size: Model input side length. Inferred from the graph when it
            is fixed; required (or defaulted to 640) for dynamic-shape models.
        providers: ONNX Runtime execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``. Defaults to
            whatever the installed runtime offers.
        soft_nms: Use Soft-NMS score decay instead of greedy suppression.

    Raises:
        LofopError: If onnxruntime is missing, or the model cannot be loaded
            or does not expose LOFOP's dense ``(boxes, scores)`` outputs.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        score_threshold: float = 0.25,
        nms_iou: float = 0.6,
        max_detections: int = 300,
        class_names: Sequence[str] | None = None,
        image_size: int | None = None,
        providers: Sequence[str] | None = None,
        soft_nms: bool = False,
    ) -> None:
        self.model_path = Path(model)
        if not self.model_path.is_file():
            raise LofopError(
                "ONNX model not found", context={"path": str(self.model_path)}
            )
        self.score_threshold = float(score_threshold)
        self.nms_iou = float(nms_iou)
        self.max_detections = int(max_detections)
        self.soft_nms = bool(soft_nms)

        self._session = self._open_session(providers)
        outputs = [o.name for o in self._session.get_outputs()]
        if len(outputs) < 2:
            raise LofopError(
                "Model does not expose LOFOP's dense (boxes, scores) outputs",
                context={"outputs": outputs, "path": str(self.model_path)},
            )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = outputs[:2]
        self.image_size = image_size or self._infer_image_size()
        self._class_names = list(class_names) if class_names else None
        logger.info(
            "Loaded %s (input %d, ops backend %s)",
            self.model_path.name, self.image_size, ops_backend(),
        )

    # -- construction helpers ----------------------------------------------

    def _open_session(self, providers: Sequence[str] | None):
        """Create the ONNX Runtime session, with an actionable error if absent."""
        try:
            import onnxruntime
        except ImportError as exc:
            raise LofopError(
                "onnxruntime is required for lofop.runtime: "
                "pip install onnxruntime (or onnxruntime-gpu)",
            ) from exc
        try:
            kwargs: dict[str, Any] = {}
            if providers is not None:
                kwargs["providers"] = list(providers)
            return onnxruntime.InferenceSession(str(self.model_path), **kwargs)
        except Exception as exc:  # onnxruntime raises bare RuntimeError subclasses
            raise LofopError(
                f"Cannot load ONNX model: {exc}", context={"path": str(self.model_path)}
            ) from exc

    def _infer_image_size(self) -> int:
        """Read the input side length from the graph, or fall back to 640."""
        shape = self._session.get_inputs()[0].shape
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] > 0:
            return int(shape[2])
        logger.debug("Model has dynamic input height; using %d", _DEFAULT_SIZE)
        return _DEFAULT_SIZE

    # -- inference ----------------------------------------------------------

    def predict(
        self, source: ImageSource, *, score_threshold: float | None = None
    ) -> list[Detection]:
        """Detect objects in one image; boxes come back in its own pixels.

        Args:
            source: An image path, or an :class:`~lofop.runtime.Image` whose
                pixels you already hold.
            score_threshold: Override the confidence cut for this call only.

        Returns:
            Detections sorted by descending score, at most
            ``max_detections`` of them.
        """
        image = source if isinstance(source, Image) else Image.load(source)
        threshold = self.score_threshold if score_threshold is None else float(score_threshold)

        tensor, meta = letterbox(
            image.pixels, image.width, image.height,
            channels=image.channels, size=self.image_size,
        )
        boxes, scores = self._run(tensor)
        # The same torch-free post-processing the documented deployment path
        # uses: dense best-class decode plus class-aware NMS, both native.
        detections = postprocess_dense(
            boxes, scores,
            score_threshold=threshold,
            nms_iou=self.nms_iou,
            max_detections=self.max_detections,
            soft=self.soft_nms,
        )
        mapped = unletterbox_boxes(detections.boxes, meta)
        return [
            Detection(
                box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                score=float(score),
                label=int(label),
                name=self.class_name(int(label)),
            )
            for box, score, label in zip(mapped, detections.scores, detections.labels)
        ]

    def predict_batch(
        self, sources: Sequence[ImageSource], *, score_threshold: float | None = None
    ) -> list[list[Detection]]:
        """Detect objects in several images; one result list per input."""
        return [self.predict(item, score_threshold=score_threshold) for item in sources]

    def _run(self, tensor: Sequence[float]) -> tuple[list[list[float]], list[list[float]]]:
        """Execute the graph; return dense boxes (N,4) and scores (N,C)."""
        size = self.image_size
        try:
            import numpy as np
        except ImportError as exc:
            raise LofopError(
                "numpy is required for lofop.runtime: pip install numpy"
            ) from exc
        try:
            batch = np.asarray(tensor, dtype=np.float32).reshape(1, 3, size, size)
            raw_boxes, raw_scores = self._session.run(
                self._output_names, {self._input_name: batch}
            )
        except Exception as exc:
            raise LofopError(
                f"Inference failed: {exc}", context={"path": str(self.model_path)}
            ) from exc
        boxes = np.asarray(raw_boxes).reshape(-1, 4).tolist()
        scores_array = np.asarray(raw_scores)
        scores = scores_array.reshape(-1, int(scores_array.shape[-1])).tolist()
        return boxes, scores

    # -- introspection -------------------------------------------------------

    def class_name(self, label: int) -> str:
        """Name for a class index, falling back to ``class_<index>``."""
        if self._class_names and 0 <= label < len(self._class_names):
            return self._class_names[label]
        return f"class_{label}"

    @property
    def providers(self) -> list[str]:
        """Execution providers the session is actually using."""
        return list(self._session.get_providers())

    @property
    def ops_backend(self) -> str:
        """Active native ops tier: ``cuda``, ``native``, or ``python``."""
        return ops_backend()

    def __repr__(self) -> str:
        return (
            f"Detector(model={self.model_path.name!r}, image_size={self.image_size}, "
            f"providers={self.providers}, ops={self.ops_backend})"
        )
