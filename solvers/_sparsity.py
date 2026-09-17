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
entries where two columns in a group turn out to overlap after all. For a
structured solver the natural pattern is therefore everything its factorisation
reads or writes, fill-in included, which is a superset by construction.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace

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
    """

    n_vars: int
    n_colours: int
    colour: tuple[int, ...]
    # The pattern this was coloured from, kept so the layout can say where a
    # compressed entry belongs in an ordinary dense matrix. ``None`` for a
    # layout built without one, which is every layout that is already dense.
    pattern: tuple[tuple[int, ...], ...] | None = None
    # Set by :func:`pack`: the packed slot each ``(row, colour)`` grid position
    # collapses to, ``-1`` where the group has nothing in that row. ``None``
    # leaves the layout on the grid, where a slot exists for every position.
    packed: tuple[int, ...] | None = None
    # Slots for diagonal entries the pattern does not declare. ``I/(h*gamma)``
    # lands on every diagonal whether or not ``J`` has anything there, so a
    # packed layout has to keep room for them; nothing writes them, which is
    # why a packed layout with any of these is cleared before the sweeps.
    packed_diagonal: tuple[int, ...] | None = None

    @property
    def is_packed(self) -> bool:
        return self.packed is not None

    @property
    def size(self) -> int:
        """Elements in one trajectory's matrix."""
        if self.packed is not None:
            return 1 + max(max(self.packed), max(self.packed_diagonal))
        return self.n_vars * self.n_colours

    @property
    def is_dense(self) -> bool:
        return self.n_colours == self.n_vars

    @property
    def is_row_major_dense(self) -> bool:
        """True when a slot *is* ``row * n_vars + col``, so nothing needs moving.

        Stronger than :attr:`is_dense`: a colouring can use ``n_vars`` colours
        and still permute the columns, and a permuted dense matrix is not one
        a dense LU may be pointed at.
        """
        return (
            not self.is_packed
            and self.is_dense
            and self.colour == tuple(range(self.n_vars))
        )

    def dense_slots(self) -> np.ndarray:
        """``(n_vars * n_colours,)`` of row-major destinations, ``-1`` for none.

        Entry ``(row, colour)`` of the compressed buffer belongs at
        ``row * n_vars + col`` of a dense matrix, where ``col`` is the one
        column of that colour group with a nonzero in that row -- unique
        because that is exactly what the colouring guarantees. Slots whose
        group has no entry in the row hold nothing and map to ``-1``.
        """
        if self.pattern is None:
            raise ValueError(
                "this layout was built without a pattern, so it cannot say "
                "where its entries belong in a dense matrix"
            )
        table = np.full(self.n_vars * self.n_colours, -1, dtype=np.int32)
        for row, cols in enumerate(self.pattern):
            for c in cols:
                table[row * self.n_colours + self.colour[c]] = row * self.n_vars + c
        return table

    @property
    def diagonal(self) -> tuple[int, ...]:
        """Slot of each ``(i, i)`` entry, for the ``1/(h*gamma)`` term."""
        return tuple(self.slot(i, i) for i in range(self.n_vars))

    def slot(self, row: int, col: int) -> int:
        """Where entry ``(row, col)`` lives. For solvers written by hand."""
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


def pack(compressed: CompressedJacobian) -> CompressedJacobian:
    """Squeeze a colour grid down to one slot per declared entry.

    The grid stores ``n_vars * n_colours`` slots because that makes the AD
    write a straight run; a pattern that colours well leaves most of them
    holding nothing. Packing keeps everything that makes the grid cheap to
    address -- a slot is still a compile-time constant per ``(row, col)``, with
    no ``rowptr`` to chase and no search -- and simply stops paying for the
    holes, which matters because the matrix is per-thread local memory.

    The cost is that the write becomes a scatter rather than a run, and that a
    solver may only touch entries the pattern declares: on the grid an
    undeclared ``(r, c)`` silently aliases another column's slot, and packed it
    raises. For a structured solver that is the right trade, since its pattern
    already declares its fill-in.

    Entries are packed row-major and, within a row, in colour order, so a row
    of the factorisation walks contiguous memory.
    """
    if compressed.pattern is None:
        raise ValueError(
            "packing needs the pattern the layout was coloured from; build it "
            "with colour_sparsity rather than by hand"
        )
    n_vars, n_colours = compressed.n_vars, compressed.n_colours
    table = np.full(n_vars * n_colours, -1, dtype=np.int64)
    diagonal = np.full(n_vars, -1, dtype=np.int64)
    nxt = 0
    for row, cols in enumerate(compressed.pattern):
        for c in sorted(cols, key=lambda c: compressed.colour[c]):
            table[row * n_colours + compressed.colour[c]] = nxt
            if c == row:
                diagonal[row] = nxt
            nxt += 1
    # ``I/(h*gamma)`` lands on every diagonal, including the ones J leaves
    # structurally zero, so those get a slot of their own here. Nothing writes
    # them, so the kernel clears the matrix when there are any.
    for row in range(n_vars):
        if diagonal[row] < 0:
            diagonal[row] = nxt
            nxt += 1
    return replace(
        compressed,
        packed=tuple(int(s) for s in table),
        packed_diagonal=tuple(int(s) for s in diagonal),
    )


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
        n_vars=n_vars, n_colours=max(colour) + 1, colour=colour, pattern=pattern
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
