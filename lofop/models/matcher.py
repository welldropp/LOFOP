"""One-to-one matching between query slots and ground-truth objects.

A set-prediction head emits a fixed number of slots and must decide, for each
training image, which slot is responsible for which object. Because exactly
one slot is assigned per object, the head learns to suppress its own
duplicates and inference needs no NMS at all.

The pairing is an assignment problem: build a cost matrix over
(slot, object) pairs and pick the cheapest one-to-one selection. LOFOP solves
it with :func:`lofop.ops.assignment.optimal_assignment`, its own pure-Python
exact solver, so the head needs no scipy.

Costs combine three terms with configurable weights: classification (how
badly the slot scores the object's class), L1 distance between normalised box
coordinates, and the GIoU of the boxes. Classification uses a focal-style
cost so that easy background slots do not dominate the matrix.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lofop.core.exceptions import ModelError
from lofop.models.losses import pairwise_iou
from lofop.ops.assignment import optimal_assignment


class PivotMatcher:
    """Cheapest one-to-one assignment of query slots to objects.

    Args:
        class_weight: Weight of the classification term.
        box_weight: Weight of the normalised L1 box distance.
        giou_weight: Weight of the GIoU term.
        focal_alpha: Focal balancing factor in the classification cost.
        focal_gamma: Focal focusing exponent in the classification cost.

    Raises:
        ModelError: If every weight is zero, which would make the cost matrix
            uniform and the assignment arbitrary.
    """

    def __init__(
        self,
        *,
        class_weight: float = 2.0,
        box_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        if class_weight == 0.0 and box_weight == 0.0 and giou_weight == 0.0:
            raise ModelError("PivotMatcher needs at least one non-zero cost weight")
        self.class_weight = float(class_weight)
        self.box_weight = float(box_weight)
        self.giou_weight = float(giou_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    @torch.no_grad()
    def match(
        self,
        class_logits: Tensor,
        boxes: Tensor,
        target_boxes: Tensor,
        target_labels: Tensor,
        image_size: float,
    ) -> tuple[Tensor, Tensor]:
        """Assign slots to the objects of one image.

        Args:
            class_logits: (Q, C) class logits for this image's slots.
            boxes: (Q, 4) predicted xyxy boxes in pixels.
            target_boxes: (M, 4) ground-truth xyxy boxes in pixels.
            target_labels: (M,) ground-truth class indices.
            image_size: Side length used to normalise the L1 term, so it is
                scale-free and comparable with the GIoU term.

        Returns:
            ``(slot_index, target_index)``: two 1-D long tensors of equal
            length listing the matched pairs. Both are empty when the image
            has no objects.
        """
        num_slots = class_logits.shape[0]
        num_targets = target_boxes.shape[0]
        device = class_logits.device
        empty = torch.zeros(0, dtype=torch.long, device=device)
        if num_slots == 0 or num_targets == 0:
            return empty, empty

        probabilities = class_logits.sigmoid()
        # Focal-style classification cost: the price of *not* already scoring
        # this class highly. Constant terms are dropped since only relative
        # cost matters to the assignment.
        positive = (
            self.focal_alpha
            * (1.0 - probabilities) ** self.focal_gamma
            * (-(probabilities + 1e-8).log())
        )
        negative = (
            (1.0 - self.focal_alpha)
            * probabilities**self.focal_gamma
            * (-(1.0 - probabilities + 1e-8).log())
        )
        class_cost = positive[:, target_labels] - negative[:, target_labels]

        scale = max(float(image_size), 1.0)
        box_cost = torch.cdist(boxes / scale, target_boxes / scale, p=1)
        giou_cost = -_generalized_iou(boxes, target_boxes)

        total = (
            self.class_weight * class_cost
            + self.box_weight * box_cost
            + self.giou_weight * giou_cost
        )
        # Guard the solver against non-finite cells produced by degenerate
        # predictions early in training.
        total = torch.nan_to_num(total, nan=1e4, posinf=1e4, neginf=-1e4)

        assignment = optimal_assignment(total.detach().cpu().tolist())
        pairs = [(slot, target) for slot, target in enumerate(assignment) if target >= 0]
        if not pairs:
            return empty, empty
        slot_index = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
        target_index = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
        return slot_index, target_index


def _generalized_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise GIoU: IoU minus the enclosing box's wasted area fraction."""
    iou = pairwise_iou(boxes_a, boxes_b)
    top_left = torch.min(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.max(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    sides = (bottom_right - top_left).clamp(min=0)
    enclosing = sides[..., 0] * sides[..., 1]

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clamp(min=0)
    union = (area_a[:, None] + area_b[None, :] - _intersection(boxes_a, boxes_b)).clamp(min=1e-7)
    return iou - (enclosing - union).clamp(min=0) / enclosing.clamp(min=1e-7)


def _intersection(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    top_left = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    sides = (bottom_right - top_left).clamp(min=0)
    return sides[..., 0] * sides[..., 1]
