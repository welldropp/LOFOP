"""LOFOP inference runtime: torch-free ONNX deployment.

The serving counterpart to the training stack. Where :class:`lofop.Detector`
builds and trains PyTorch models, this package runs an exported ``model.onnx``
with no deep-learning framework attached -- ONNX Runtime executes the graph and
LOFOP's own native kernels do the preprocessing and post-processing.

It mirrors the C++ SDK in ``cpp/`` type for type (``Detector``, ``Detection``,
``Image``) and shares its kernels, so the same model file produces the same
detections from either language::

    from lofop.runtime import Detector

    model = Detector("model.onnx")
    for hit in model.predict("image.jpg"):
        print(hit.name, hit.confidence, hit.box)

Requires ``onnxruntime`` and ``numpy``; both stay optional for the rest of
LOFOP, and importing this package is what pulls them in.
"""

from lofop.runtime.detector import Detection, Detector
from lofop.runtime.image import Image

__all__ = ["Detector", "Detection", "Image"]
