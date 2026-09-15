"""Optimal and greedy one-to-one assignment over a cost matrix.

Set-prediction detection heads and multi-object trackers both need the same
primitive: given an N x M matrix of costs, choose at most one column per row
(and one row per column) so the total cost is minimal. LOFOP implements it
once, here, in pure Python -- no numpy, no scipy -- so the op stays usable on
the torch-free core and inside the deployment post-processing path.

:func:`optimal_assignment` solves the problem exactly with a shortest
augmenting path method over dual potentials (the classical O(n^3) formulation
of the assignment problem). :func:`greedy_assignment` trades optimality for a
simpler pass and is useful when the cost matrix is already close to diagonal.
"""

from __future__ import annotations

from collections.abc import Sequence

from lofop.core.exceptions import LofopError

CostMatrix = Sequence[Sequence[float]]

_INFINITY = float("inf")


def _validate(cost: CostMatrix) -> tuple[int, int]:
    rows = len(cost)
    if rows == 0:
        return 0, 0
    columns = len(cost[0])
    for row in cost:
        if len(row) != columns:
            raise LofopError(
                "Cost matrix rows must all have the same length",
                context={"expected": columns, "got": len(row)},
            )
    return rows, columns


def optimal_assignment(cost: CostMatrix) -> list[int]:
    """Minimum-total-cost one-to-one assignment.

    Args:
        cost: N x M matrix of costs. Finite values only; use a large finite
            sentinel rather than ``inf`` to forbid a pairing.

    Returns:
        A list of length N. Entry ``i`` is the column assigned to row ``i``,
        or ``-1`` when row ``i`` is left unassigned (only possible when
        ``M < N``).

    The solver works on a row-major copy padded to a square matrix, so
    rectangular inputs are handled without special-casing at the call site.
    """
    rows, columns = _validate(cost)
    if rows == 0 or columns == 0:
        return [-1] * rows

    # Pad to square with zero-cost dummy entries; dummies absorb the surplus
    # side and are stripped from the result below.
    size = max(rows, columns)
    padded = [
        [float(cost[i][j]) if i < rows and j < columns else 0.0 for j in range(size)]
        for i in range(size)
    ]

    # row_potential/column_potential are the dual variables; column_match[j]
    # holds the 1-based row currently matched to column j (0 = unmatched).
    # Index 0 of the column arrays is the virtual "entry" column the search
    # starts from, which is why they are sized size + 1.
    row_potential = [0.0] * (size + 1)
    column_potential = [0.0] * (size + 1)
    column_match = [0] * (size + 1)
    predecessor = [0] * (size + 1)

    for row in range(1, size + 1):
        column_match[0] = row
        current_column = 0
        slack = [_INFINITY] * (size + 1)
        visited = [False] * (size + 1)

        while True:
            visited[current_column] = True
            current_row = column_match[current_column]
            best_delta = _INFINITY
            next_column = -1

            for column in range(1, size + 1):
                if visited[column]:
                    continue
                reduced = (
                    padded[current_row - 1][column - 1]
                    - row_potential[current_row]
                    - column_potential[column]
                )
                if reduced < slack[column]:
                    slack[column] = reduced
                    predecessor[column] = current_column
                if slack[column] < best_delta:
                    best_delta = slack[column]
                    next_column = column

            if next_column < 0:
                raise LofopError("Assignment failed: cost matrix has no finite completion")

            # Shift the potentials so the chosen edge becomes tight.
            for column in range(size + 1):
                if visited[column]:
                    row_potential[column_match[column]] += best_delta
                    column_potential[column] -= best_delta
                else:
                    slack[column] -= best_delta

            current_column = next_column
            if column_match[current_column] == 0:
                break

        # Walk the augmenting path back to the entry column, flipping edges.
        while current_column:
            previous = predecessor[current_column]
            column_match[current_column] = column_match[previous]
            current_column = previous

    assignment = [-1] * rows
    for column in range(1, size + 1):
        row = column_match[column]
        if 0 < row <= rows and column <= columns:
            assignment[row - 1] = column - 1
    return assignment


def greedy_assignment(cost: CostMatrix, *, max_cost: float | None = None) -> list[int]:
    """Repeatedly take the cheapest remaining pair.

    Args:
        cost: N x M matrix of costs.
        max_cost: Pairs costing more than this are never made, leaving both
            sides unassigned. ``None`` accepts any finite cost.

    Returns:
        A list of length N; entry ``i`` is row ``i``'s column, or ``-1``.

    Cheaper than :func:`optimal_assignment` and order-independent, but it can
    return a worse total cost: an early cheap pair may block a better global
    solution.
    """
    rows, columns = _validate(cost)
    assignment = [-1] * rows
    if rows == 0 or columns == 0:
        return assignment

    pairs = sorted(
        (
            (float(cost[i][j]), i, j)
            for i in range(rows)
            for j in range(columns)
            if max_cost is None or float(cost[i][j]) <= max_cost
        ),
        key=lambda item: item[0],
    )
    taken_rows: set[int] = set()
    taken_columns: set[int] = set()
    for _, row, column in pairs:
        if row in taken_rows or column in taken_columns:
            continue
        assignment[row] = column
        taken_rows.add(row)
        taken_columns.add(column)
        if len(taken_rows) == rows or len(taken_columns) == columns:
            break
    return assignment
