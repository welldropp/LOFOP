"""LOFOP model subsystem: the LOFOP-Detect family and its parts.

Requires PyTorch (install with ``pip install lofop[models]``); the rest of
the framework stays importable without it. Importing this package registers
``backbone/RidgeNet``, ``neck/DeltaFusion``, ``head/ApexHead``,
``head/StencilHead``, ``head/VertexHead``, ``model/LofopDetect``,
``model/LofopSegment``, and ``model/LofopPose``.
Design document: docs/lofop-detect.md.
"""

from lofop.models.assigner import DynamicTopKAssigner
from lofop.models.backbone import RidgeNet
from lofop.models.detector import LofopDetect
from lofop.models.head import ApexHead
from lofop.models.losses import giou_loss, pairwise_iou, sigmoid_focal_loss
from lofop.models.matcher import PivotMatcher
from lofop.models.mobile_backbone import InvertedResidual, SwiftNet
from lofop.models.neck import DeltaFusion
from lofop.models.pose_head import VertexHead
from lofop.models.poser import LofopPose
from lofop.models.query_detector import LofopQuery
from lofop.models.query_head import SlateHead
from lofop.models.seg_head import StencilHead
from lofop.models.segmenter import LofopSegment

__all__ = [
    "RidgeNet",
    "SwiftNet",
    "InvertedResidual",
    "DeltaFusion",
    "ApexHead",
    "StencilHead",
    "VertexHead",
    "SlateHead",
    "LofopDetect",
    "LofopSegment",
    "LofopPose",
    "LofopQuery",
    "DynamicTopKAssigner",
    "PivotMatcher",
    "sigmoid_focal_loss",
    "giou_loss",
    "pairwise_iou",
]
