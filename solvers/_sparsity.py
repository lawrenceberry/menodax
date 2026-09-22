"""Turning a Jacobian sparsity pattern into the fewest forward sweeps.

Forward-mode AD does not hand back a Jacobian; it hands back ``J v`` for a
direction ``v``. Seeding the unit vectors one at a time costs ``n_vars`` sweeps.
But two columns that share no row are *structurally orthogonal*: their
contributions to ``J v`` never land on the same component, so seeding both at
once -- ``v = e_i + e_j`` -- returns both columns uncorrupted in a single sweep,
and the sparsity pattern says which component belongs to which column.

Partitioning the columns into as few such groups as possible is exactly vertex
colouring of the column intersection graph ``S^T S``, which NetworkX's greedy
colouring does well enough here: the bound that matters is the largest set of
mutually overlapping columns, and greedy hits it on the patterns these solvers
see.

The pattern a caller supplies must be a *superset* of the true nonzeros --
colouring a superset is conservative, colouring a subset silently corrupts
entries where two columns in a group turn out to overlap after all. It need not
cover the factorisation's fill-in: that has slots of its own, laid out by
[`solvers._sparse_direct`][], and no column of ``J`` writes them.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

# The strategies NetworkX offers that are worth trying. They are cheap -- this
# runs once per kernel build, on the host -- so the best of them is taken
# rather than one being picked in advance.
_STRATEGIES = (
    "largest_first",
    "smallest_last",
    "independent_set",
    "saturation_largest_first",
    "DSATUR",
)


@dataclass(frozen=True)
class CompressedJacobian:
    """A column-compressed Jacobian layout, and the seeds that fill it.

    Entry ``(r, c)`` of the Jacobian lives at slot ``r * n_colours + colour[c]``.
    The layout is dense in the rows and compressed in the columns, so a slot
    exists for every row of every colour; slots whose colour group has no
    nonzero in that row are written but never read, which costs nothing and
    keeps the write a straight run over a column block.

    With no pattern every column conflicts with every other, colouring gives
    ``n_colours == n_vars``, and this degenerates exactly to the dense
    row-major matrix -- so the dense path is not a special case in the kernel,
    just the uninformative end of the same mechanism.

    [`solvers._sparse_direct`][] overrides the grid with ``packed``: the slot
    of each entry in its own CSR image of ``L + U``, so the sweeps deposit ``-J``
    straight into the buffer the factorisation will work in.
    """

    n_vars: int
    n_colours: int
    colour: tuple[int, ...]
    # Set by `solvers._sparse_direct.compressed_jacobian`: the slot each
    # ``(row, colour)`` grid position collapses to, ``-1`` where the group has
    # nothing in that row. ``None`` leaves the layout on the grid, where a slot
    # exists for every position.
    packed: tuple[int, ...] | None = None
    # Slots for diagonal entries the pattern does not declare. ``I/(h*gamma)``
    # lands on every diagonal whether or not ``J`` has anything there, so a
    # packed layout has to keep room for them; nothing writes them, which is
    # why a packed layout with any of these is cleared before the sweeps.
    packed_diagonal: tuple[int, ...] | None = None
    # Slots the buffer holds beyond the ones some entry claims: the direct
    # sparse factorisation needs room for its fill-in, which belongs to no
    # column of ``J`` and so appears in no slot table. ``None`` means the
    # claimed slots are the whole buffer.
    n_slots: int | None = None

    @property
    def size(self) -> int:
        """Elements in one trajectory's matrix."""
        if self.n_slots is not None:
            return self.n_slots
        if self.packed is not None:
            return 1 + max(max(self.packed), max(self.packed_diagonal))
        return self.n_vars * self.n_colours

    @property
    def diagonal(self) -> tuple[int, ...]:
        """Slot of each ``(i, i)`` entry, for the ``1/(h*gamma)`` term."""
        return tuple(self.slot(i, i) for i in range(self.n_vars))

    def slot(self, row: int, col: int) -> int:
        """Where entry ``(row, col)`` lives."""
        grid = row * self.n_colours + self.colour[col]
        if self.packed is None:
            return grid
        slot = self.packed[grid]
        if slot < 0 and row == col:
            return self.packed_diagonal[row]
        if slot < 0:
            raise ValueError(
                f"entry ({row}, {col}) is not in the pattern, so a packed "
                "layout has no slot for it; declare it in the sparsity pattern"
            )
        return slot

    def store_slots(self) -> np.ndarray | None:
        """Where each ``(row, colour)`` sweep value goes, or ``None`` for a run.

        On the grid a colour group's values are a contiguous block and the
        kernel writes them straight down; packed, each one is scattered to its
        own slot and the positions no entry claims are simply not written.
        """
        if self.packed is None:
            return None
        return np.asarray(self.packed, dtype=np.int32)

    def seed_table(self, min_width: int = 0) -> np.ndarray:
        """``(n_colours + 1, max(n_vars, min_width))`` of tangent directions.

        Row ``g`` is the indicator of colour group ``g``, so one sweep seeded
        with it yields that whole group. The extra final row is all zeros, for
        the sweeps that vary ``t`` or a parameter rather than the state -- which
        is why the rows are widened to ``min_width``: the zero row doubles as
        the null *parameter* direction, and there may be more parameters than
        state variables.
        """
        width = max(self.n_vars, min_width)
        seeds = np.zeros((self.n_colours + 1, width), dtype=np.float64)
        for c, g in enumerate(self.colour):
            seeds[g, c] = 1.0
        return seeds


def dense_jacobian(n_vars: int) -> CompressedJacobian:
    """Every column its own colour: the row-major dense matrix."""
    return CompressedJacobian(
        n_vars=n_vars, n_colours=n_vars, colour=tuple(range(n_vars))
    )


def normalize_sparsity(sparsity, n_vars: int) -> tuple[tuple[int, ...], ...]:
    """Accept a dense mask, a scipy sparse matrix, or ``(row, col)`` pairs.

    Returned as a hashable tuple of row tuples so the colouring can be cached
    on it: a kernel is rebuilt per pattern, not per call.
    """
    if hasattr(sparsity, "tocoo"):  # scipy sparse
        coo = sparsity.tocoo()
        pairs = zip(coo.row.tolist(), coo.col.tolist())
    else:
        arr = np.asarray(sparsity)
        if arr.ndim == 2 and arr.shape == (n_vars, n_vars):
            rows, cols = np.nonzero(arr)
            pairs = zip(rows.tolist(), cols.tolist())
        elif arr.ndim == 2 and arr.shape[1] == 2:
            pairs = ((int(r), int(c)) for r, c in arr.tolist())
        else:
            raise ValueError(
                "sparsity must be an (n_vars, n_vars) mask, a scipy sparse "
                f"matrix, or an (nnz, 2) array of (row, col); got shape "
                f"{arr.shape} for n_vars={n_vars}"
            )
    by_row: list[set[int]] = [set() for _ in range(n_vars)]
    for r, c in pairs:
        if not (0 <= r < n_vars and 0 <= c < n_vars):
            raise ValueError(
                f"sparsity entry ({r}, {c}) lies outside an {n_vars}x{n_vars} Jacobian"
            )
        by_row[r].add(c)
    return tuple(tuple(sorted(cs)) for cs in by_row)


@functools.cache
def colour_sparsity(pattern: tuple[tuple[int, ...], ...]) -> CompressedJacobian:
    """Colour a pattern's column intersection graph, fewest colours wins."""
    import networkx as nx

    n_vars = len(pattern)
    # Columns conflict when some row holds both. Building the adjacency from the
    # rows directly is cheaper than forming S^T S for the patterns seen here,
    # where a row has a handful of entries out of hundreds of columns.
    graph = nx.Graph()
    graph.add_nodes_from(range(n_vars))
    for cols in pattern:
        for a in range(len(cols)):
            for b in range(a + 1, len(cols)):
                graph.add_edge(cols[a], cols[b])

    best: dict[int, int] | None = None
    for strategy in _STRATEGIES:
        colouring = nx.coloring.greedy_color(graph, strategy=strategy)
        if best is None or max(colouring.values()) < max(best.values()):
            best = colouring
    assert best is not None

    colour = tuple(int(best.get(c, 0)) for c in range(n_vars))
    compressed = CompressedJacobian(
        n_vars=n_vars, n_colours=max(colour) + 1, colour=colour
    )
    _check_orthogonal(pattern, compressed)
    return compressed


def _check_orthogonal(pattern, compressed: CompressedJacobian) -> None:
    """No row may hold two columns of the same colour.

    A violation means two entries would share a slot and silently overwrite one
    another, which is the one way this scheme can be wrong. It costs a pass over
    the pattern once per kernel build to rule out.
    """
    for row, cols in enumerate(pattern):
        seen: dict[int, int] = {}
        for c in cols:
            g = compressed.colour[c]
            if g in seen:
                raise ValueError(
                    f"columns {seen[g]} and {c} share colour {g} but both have "
                    f"an entry in row {row}; the colouring is not structurally "
                    "orthogonal for this pattern"
                )
            seen[g] = c
