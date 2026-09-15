"""Cost matrices and matching for frame-to-frame track association.

The per-frame hot loop of any tracker is the same shape as detection
post-processing: compare every track against every detection. LOFOP already
owns a fast kernel for exactly that -- :func:`lofop.ops.iou_matrix`, which
runs on the native C++ (or CUDA) tier when built and falls back to pure
Python otherwise -- so association reuses it rather than growing a second
implementation.

Everything in this module is plain Python over lists: no numpy, no scipy. The
matching itself comes from :mod:`lofop.ops.assignment`. That keeps the
association path importable anywhere LOFOP is, and means the expensive part
is the part that is already compiled.
"""

from __future__ import annotations

from collections.abc import Sequence

from lofop.ops import iou_matrix
from lofop.ops.assignment import greedy_assignment, optimal_assignment

BoxList = Sequence[Sequence[float]]

# An IoU of 0 costs 1.0, so this sits just above any real pairing and marks
# a cell the matcher must never choose.
FORBIDDEN_COST = 10.0


def iou_cost_matrix(
    track_boxes: BoxList,
    detection_boxes: BoxList,
    *,
    track_labels: Sequence[int] | None = None,
    detection_labels: Sequence[int] | None = None,
) -> list[list[float]]:
    """Association cost as ``1 - IoU``, optionally class-aware.

    Args:
        track_boxes: T predicted track boxes, xyxy.
        detection_boxes: D detection boxes, xyxy.
        track_labels: Optional class id per track. When both label sequences
            are given, pairings across different classes are set to
            :data:`FORBIDDEN_COST` so a car can never inherit a person's id.
        detection_labels: Optional class id per detection.

    Returns:
        A T x D matrix of costs in ``[0, 1]``, or ``FORBIDDEN_COST`` for
        class-mismatched cells.
    """
    if not track_boxes or not detection_boxes:
        return [[] for _ in track_boxes]

    overlaps = iou_matrix(track_boxes, detection_boxes)
    class_aware = track_labels is not None and detection_labels is not None
    costs: list[list[float]] = []
    for row_index, row in enumerate(overlaps):
        costs.append(
            [
                FORBIDDEN_COST
                if class_aware and track_labels[row_index] != detection_labels[column_index]
                else 1.0 - float(overlap)
                for column_index, overlap in enumerate(row)
            ]
        )
    return costs


def match(
    cost: Sequence[Sequence[float]],
    *,
    max_cost: float = 0.8,
    optimal: bool = True,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pair tracks with detections under a cost ceiling.

    Args:
        cost: T x D cost matrix, typically from :func:`iou_cost_matrix`.
        max_cost: Pairs above this cost are rejected and both sides are
            reported unmatched. With IoU costs, ``0.8`` accepts pairings
            down to 20% overlap.
        optimal: Use the exact assignment solver. ``False`` selects the
            cheaper greedy pass, which can return a worse total cost.

    Returns:
        ``(pairs, unmatched_tracks, unmatched_detections)`` where ``pairs``
        holds ``(track_index, detection_index)`` tuples.
    """
    track_count = len(cost)
    detection_count = len(cost[0]) if track_count and cost[0] is not None else 0
    if track_count == 0 or detection_count == 0:
        return [], list(range(track_count)), list(range(detection_count))

    if optimal:
        assignment = optimal_assignment(cost)
    else:
        assignment = greedy_assignment(cost, max_cost=max_cost)

    pairs: list[tuple[int, int]] = []
    matched_detections: set[int] = set()
    for track_index, detection_index in enumerate(assignment):
        # The exact solver fills every row it can; the ceiling is applied
        # here so both solvers honour the same acceptance rule.
        if detection_index < 0 or cost[track_index][detection_index] > max_cost:
            continue
        pairs.append((track_index, detection_index))
        matched_detections.add(detection_index)

    matched_tracks = {track_index for track_index, _ in pairs}
    unmatched_tracks = [index for index in range(track_count) if index not in matched_tracks]
    unmatched_detections = [
        index for index in range(detection_count) if index not in matched_detections
    ]
    return pairs, unmatched_tracks, unmatched_detections


def associate(
    track_boxes: BoxList,
    detection_boxes: BoxList,
    *,
    track_labels: Sequence[int] | None = None,
    detection_labels: Sequence[int] | None = None,
    max_cost: float = 0.8,
    optimal: bool = True,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Build the IoU cost matrix and match in one call.

    Convenience wrapper over :func:`iou_cost_matrix` and :func:`match`; see
    those for the arguments. Returns the same triple as :func:`match`.
    """
    # An empty cost matrix cannot express how many detections exist, so the
    # degenerate sides are resolved here where both lengths are known.
    if not track_boxes or not detection_boxes:
        return [], list(range(len(track_boxes))), list(range(len(detection_boxes)))

    cost = iou_cost_matrix(
        track_boxes,
        detection_boxes,
        track_labels=track_labels,
        detection_labels=detection_labels,
    )
    return match(cost, max_cost=max_cost, optimal=optimal)
