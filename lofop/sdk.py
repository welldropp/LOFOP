"""LOFOP Python SDK: the high-level, one-import API.

Everything the framework can do -- build a detector, train it, run inference
on images, evaluate, export -- through one documented class::

    from lofop import Detector

    det = Detector("lofop-detect-s", num_classes=2)
    det.train(data_format="coco", train_source="train.json",
              image_root="images/", epochs=50)
    for hit in det.predict("photo.jpg"):
        print(hit.boxes, hit.scores, hit.labels)
    det.export("model.onnx")

The SDK wraps the lower layers (registries, configs, trainer, deploy) without
hiding them: ``Detector.model`` is the underlying torch module, and every
lower-level API remains public for users who need more control. Requires the
``lofop[models]`` extra (PyTorch). Full guide: docs/sdk.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import torch
from PIL import Image

import lofop.models  # noqa: F401  (registers model components)
from lofop.core.config import Config
from lofop.core.exceptions import LofopError, ModelError
from lofop.core.logging import get_logger
from lofop.data.dataset import Dataset
from lofop.deploy.postprocess import Detections
from lofop.models.detector import LofopDetect
from lofop.registries import HUB
from lofop.training.torch_data import image_to_tensor

_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "lofop-detect"
_VARIANTS = (
    "n", "s", "ex",
    "n-seg", "s-seg", "ex-seg",
    "n-pose", "s-pose", "ex-pose",
)

ImageSource = Union[str, Path, "Image.Image", torch.Tensor]

logger = get_logger(__name__)


def _resolve_model_spec(model: str | Path | Config | dict) -> Config:
    """Turn a variant name / YAML path / mapping into a full model config."""
    if isinstance(model, (Config, dict)):
        cfg = model if isinstance(model, Config) else Config(model)
        return cfg if "model" in cfg else Config({"model": cfg.to_dict()})
    name = str(model)
    short = name.replace("lofop-detect-", "")
    candidate = _CONFIG_DIR / f"{short}.yaml"
    path = Path(name) if Path(name).suffix in (".yaml", ".yml") else candidate
    if not path.is_file():
        raise LofopError(
            f"Unknown model {model!r}",
            context={"variants": [f"lofop-detect-{v}" for v in _VARIANTS]},
        )
    return Config.load(path, resolve=False)


def load_checkpoint_state(path: str | Path) -> dict:
    """Read a LOFOP training checkpoint and return the best weights in it.

    Prefers the EMA weights (they evaluate higher) and falls back to the raw
    model weights, so ``best.pt``/``last.pt`` from :class:`~lofop.training.Trainer`
    and plain ``state_dict`` files all load.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "ema" in payload:
        return payload["ema"]["module"]
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


class Detector:
    """A LOFOP object detector with a task-level API.

    Args:
        model: What to build. One of:

            * a variant name -- detection ``"lofop-detect-n" | "s" | "ex"``,
              segmentation ``"lofop-detect-n-seg" | "s-seg" | "ex-seg"``, or
              pose ``"lofop-detect-n-pose" | "s-pose" | "ex-pose"``
              (with or without the ``lofop-detect-`` prefix);
            * a path to a model config YAML;
            * a :class:`~lofop.core.config.Config`/dict with a ``model`` spec;
            * an already-built :class:`~lofop.models.LofopDetect` module.
        num_classes: Object categories the head predicts. Ignored when
            ``model`` is an already-built module.
        checkpoint: Optional path to trained weights (``best.pt``,
            ``last.pt``, or a bare ``state_dict``); EMA weights are used
            automatically when present.
        class_names: Optional label names, used to attach names to
            predictions; defaults to ``"class_<i>"``.
        image_size: Inference resolution; images are resized to this square
            and boxes are mapped back to original coordinates.
        device: ``"cpu"`` / ``"cuda"``; auto-selects CUDA when available.
        num_keypoints: Keypoints per instance for pose variants (default 17,
            the COCO person skeleton). Ignored by detection/segmentation
            models.
        nms_mode: Duplicate-removal strategy: ``"greedy"`` (default),
            ``"soft"`` (score decay, better in crowds), or ``"free"``
            (NMS-free peak selection -- no suppression loop at all).

    Attributes:
        model: The underlying :class:`LofopDetect` torch module -- fully
            accessible for anything the SDK does not cover.
    """

    def __init__(
        self,
        model: str | Path | Config | dict | LofopDetect = "lofop-detect-s",
        *,
        num_classes: int = 80,
        checkpoint: str | Path | None = None,
        class_names: Sequence[str] | None = None,
        image_size: int = 640,
        device: str | None = None,
        num_keypoints: int | None = None,
        nms_mode: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        if isinstance(model, LofopDetect):
            self.model = model
        else:
            cfg = _resolve_model_spec(model)
            cfg.num_classes = num_classes
            if num_keypoints is not None and "num_keypoints" in cfg:
                cfg.num_keypoints = num_keypoints
            cfg.resolve()
            self.model = HUB.build(cfg.model)
        if class_names is not None and len(class_names) != self.model.head.num_classes:
            raise ModelError(
                "class_names length must match num_classes",
                context={"names": len(class_names), "classes": self.model.head.num_classes},
            )
        self.class_names = list(class_names) if class_names else [
            f"class_{i}" for i in range(self.model.head.num_classes)
        ]
        if nms_mode is not None:
            if nms_mode not in LofopDetect._NMS_MODES:
                raise ModelError(
                    "Unknown nms_mode",
                    context={"got": nms_mode, "known": list(LofopDetect._NMS_MODES)},
                )
            self.model.nms_mode = nms_mode
        if checkpoint is not None:
            self.model.load_state_dict(load_checkpoint_state(checkpoint))
            logger.info("Loaded weights from %s", checkpoint)
        self.model.to(self.device).eval()

    # -- inference ---------------------------------------------------------

    def predict(
        self,
        source: ImageSource | Sequence[ImageSource],
        *,
        score_threshold: float | None = None,
    ) -> list[Detections]:
        """Detect objects; boxes come back in ORIGINAL image coordinates.

        Args:
            source: One image or a sequence of images. Each may be a file
                path, a PIL image, or a CHW/1xCHW float tensor in ``[0, 1]``.
            score_threshold: Override the model's confidence threshold for
                this call only.

        Returns:
            One :class:`~lofop.deploy.postprocess.Detections` per input image
            (``boxes`` xyxy in the image's own pixels, ``scores`` descending,
            ``labels`` class indices; map to names via ``class_names``).
        """
        singular = isinstance(source, (str, Path, Image.Image, torch.Tensor))
        items = [source] if singular else list(source)
        old_threshold = self.model.score_threshold
        if score_threshold is not None:
            self.model.score_threshold = score_threshold
        try:
            results = [self._predict_one(item) for item in items]
        finally:
            self.model.score_threshold = old_threshold
        return results

    def _predict_one(self, source: ImageSource) -> Detections:
        tensor, (width, height) = self._load_image(source)
        raw = self.model.predict(tensor.unsqueeze(0).to(self.device))[0]
        scale_x = width / self.image_size
        scale_y = height / self.image_size
        boxes = []
        for x1, y1, x2, y2 in raw["boxes"].cpu().tolist():
            boxes.append([
                min(max(x1 * scale_x, 0.0), width),
                min(max(y1 * scale_y, 0.0), height),
                min(max(x2 * scale_x, 0.0), width),
                min(max(y2 * scale_y, 0.0), height),
            ])
        masks = None
        if "masks" in raw:  # segmentation variant: resize to original pixels
            masks = raw["masks"].cpu()
            if masks.shape[0] and (height, width) != masks.shape[-2:]:
                masks = torch.nn.functional.interpolate(
                    masks.unsqueeze(1).float(), size=(height, width), mode="nearest"
                ).squeeze(1) > 0.5
            elif not masks.shape[0]:
                masks = torch.zeros((0, height, width), dtype=torch.bool)
        keypoints = None
        if "keypoints" in raw:  # pose variant: map to original pixels
            keypoints = raw["keypoints"].cpu().clone()
            keypoints[..., 0] *= scale_x
            keypoints[..., 1] *= scale_y
            keypoints = keypoints.tolist()
        return Detections(
            boxes=boxes,
            scores=raw["scores"].cpu().tolist(),
            labels=raw["labels"].cpu().tolist(),
            masks=masks,
            keypoints=keypoints,
        )

    def _load_image(self, source: ImageSource) -> tuple[torch.Tensor, tuple[int, int]]:
        if isinstance(source, torch.Tensor):
            tensor = source[0] if source.ndim == 4 else source
            height, width = tensor.shape[-2:]
            if (height, width) != (self.image_size, self.image_size):
                tensor = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0), size=(self.image_size, self.image_size),
                    mode="bilinear", align_corners=False,
                )[0]
            return tensor, (width, height)
        image = source if isinstance(source, Image.Image) else Image.open(source)
        with image:
            size = image.size
            resized = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
        return image_to_tensor(resized), size

    # -- training / evaluation ----------------------------------------------

    def train(
        self,
        *,
        train_data: Dataset | None = None,
        val_data: Dataset | None = None,
        data_format: str | None = None,
        train_source: str | Path | None = None,
        val_source: str | Path | None = None,
        image_root: str | Path | None = None,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.01,
        checkpoint_dir: str | Path = "runs/train",
        strong_augment: bool = False,
        **trainer_kwargs: Any,
    ):
        """Train this detector; the trained (EMA) weights replace the current ones.

        Provide data either as canonical datasets (``train_data``/``val_data``,
        from :func:`lofop.data.load_dataset` or your own construction) OR as
        ``data_format`` + ``train_source`` (+ optional ``val_source`` /
        ``image_root``) to load from COCO/YOLO/VOC on disk.

        ``strong_augment=True`` enables the richer augmentation recipe
        (mosaic + color jitter on top of the default flip) -- recommended
        for real-data training runs.

        Extra keyword arguments pass straight to
        :class:`~lofop.training.Trainer` (``optimizer``, ``warmup_epochs``,
        ``amp``, ``workers``, ...).

        Returns:
            The final :class:`~lofop.training.DetectionMetrics`, or ``None``
            when no validation data was provided.
        """
        from lofop.data import load_dataset
        from lofop.models import LofopPose, LofopSegment
        from lofop.training import DetectionTorchDataset, Trainer

        if train_data is None:
            if not (data_format and train_source):
                raise LofopError(
                    "Provide train_data, or data_format + train_source",
                )
            kwargs = {"image_root": image_root} if image_root else {}
            train_data = load_dataset(data_format, train_source, **kwargs)
            if val_source:
                val_data = load_dataset(data_format, val_source, **kwargs)
        # Task variants pull their extra supervision from the same canonical
        # dataset: seg rasterizes polygons, pose scales keypoints.
        extras = {
            "include_masks": isinstance(self.model, LofopSegment),
            "include_keypoints": isinstance(self.model, LofopPose),
        }
        train_ds = DetectionTorchDataset(
            train_data, image_size=self.image_size, augment=True,
            strong_augment=strong_augment, **extras,
        )
        val_ds = (
            DetectionTorchDataset(val_data, image_size=self.image_size)
            if val_data is not None else None
        )
        trainer = Trainer(
            self.model, train_ds, val_ds, epochs=epochs, batch_size=batch_size,
            lr=lr, checkpoint_dir=checkpoint_dir, device=str(self.device),
            **trainer_kwargs,
        )
        metrics = trainer.fit()
        self.model.load_state_dict(trainer.ema.module.state_dict())
        self.model.to(self.device).eval()
        return metrics

    def evaluate(self, val_data: Dataset, *, batch_size: int = 8):
        """COCO-protocol evaluation on a canonical dataset.

        Returns:
            :class:`~lofop.training.DetectionMetrics` (``map50``,
            ``map50_95``, ``precision``, ``recall``, ``per_class_ap50``).
        """
        from torch.utils.data import DataLoader

        from lofop.training import DetectionTorchDataset, evaluate_detections
        from lofop.training.torch_data import detection_collate

        loader = DataLoader(
            DetectionTorchDataset(val_data, image_size=self.image_size),
            batch_size=batch_size, collate_fn=detection_collate,
        )
        predictions, targets = [], []
        with torch.inference_mode():
            for images, batch_targets in loader:
                for result in self.model.predict(images.to(self.device)):
                    predictions.append({k: v.cpu() for k, v in result.items()})
                targets.extend(batch_targets)
        return evaluate_detections(predictions, targets)

    # -- deployment ----------------------------------------------------------

    def export(self, path: str | Path, *, format: str | None = None, **kwargs: Any) -> Path:
        """Export to ONNX (default) or a TensorRT engine.

        The format is inferred from the file suffix (``.onnx`` / ``.engine``)
        unless given explicitly. Keyword arguments pass to
        :func:`~lofop.deploy.export_onnx` or
        :func:`~lofop.deploy.export_tensorrt` (``opset``, ``verify``,
        ``fp16``, ...). ``image_size`` defaults to this detector's.
        """
        from lofop.deploy import export_onnx, export_tensorrt

        path = Path(path)
        fmt = format or ("tensorrt" if path.suffix == ".engine" else "onnx")
        kwargs.setdefault("image_size", self.image_size)
        model = self.model.cpu()
        try:
            if fmt == "tensorrt":
                return export_tensorrt(model, path, **kwargs)
            if fmt == "onnx":
                return export_onnx(model, path, **kwargs)
            known = ["onnx", "tensorrt"]
            raise LofopError(f"Unknown export format {fmt!r}", context={"known": known})
        finally:
            self.model.to(self.device)

    def save(self, path: str | Path) -> Path:
        """Write the current weights as a checkpoint loadable by ``Detector``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict()}, path)
        return path

    # -- introspection -------------------------------------------------------

    def optimize(self) -> Detector:
        """Apply CPU inference optimizations (channels_last); see
        :meth:`LofopDetect.optimize_for_inference`."""
        self.model.optimize_for_inference()
        return self

    @property
    def num_parameters(self) -> int:
        """Total parameter count of the underlying model."""
        return sum(p.numel() for p in self.model.parameters())

    def __repr__(self) -> str:
        return (
            f"Detector(classes={self.model.head.num_classes}, "
            f"parameters={self.num_parameters:,}, device={self.device.type}, "
            f"image_size={self.image_size})"
        )
