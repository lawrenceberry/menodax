"""A compiled sparse direct solver for the Rodas5P iteration matrix.

``dense_lu_solver`` factorises ``M = I/(h*gamma) - J`` as though it were dense,
which is ``n_vars ** 3 / 3`` operations however few nonzeros ``M`` has. A
hand-written solver such as DISCO-EB's Schur/Einstein-Boltzmann block-LU does far
better, but only for the one block structure it was written against. This module
takes the *sparsity pattern* the kernel is already given and compiles a direct
solver for it -- any pattern, no structure assumed -- so the saving a hand-written
solver bought is available to any caller who can say where its nonzeros are.

The analysis happens once, on the host, when the kernel is built:

1. **Order.** The pattern is symmetrised and handed to SuiteSparse's AMD, which
   returns a fill-reducing permutation (see
   [`fill_reducing_order`][menodax._sparse_direct.fill_reducing_order] for
   why AMD and not COLAMD).
2. **Factor symbolically.** The exact pattern of ``L + U`` for the permuted
   matrix, fill-in included, falls out of a pure-pattern Gaussian elimination
   ([`fill_pattern`][menodax._sparse_direct.fill_pattern]) -- no numbers, no
   device, no sample matrix.
3. **Lay out.** That pattern becomes one CSR image of ``L + U``, and the
   ``(row, col) -> slot`` map it defines becomes the
   [`CompressedJacobian`][menodax._sparsity.CompressedJacobian] the AD sweeps
   write into. The buffer is exactly ``nnz(L + U)`` elements: the
   symbolic pass *is* the memory footprint.
4. **Compile.** The factorisation and the two triangular solves become
   ``cuda.jit(device=True)`` functions, in the same
   ``factorize_local(lu, ipiv)`` / ``solve_local(lu, ipiv, rhs)`` shape
   [`menodax.rodas5P.dense_lu_solver`][] has, so the kernel calls one or the
   other and has no branch. Where the structure is small enough they are emitted
   as straight-line code with every slot a literal; above that they fall back to
   loops over index tables in constant memory.

One trajectory per thread, as everywhere else in this kernel. Every thread runs
the same pattern, so there is no divergence to pay for whichever form is
emitted.

**Why the straight-line form matters.** Table-driven, a sparse routine spends a
broadcast load on the index of every value it is about to read, and the load of
the value cannot issue until that index arrives. With a big enough ensemble the
other trajectories cover that latency; DISCO-EB's single-cosmology case is 128
trajectories, four warps on a 46-SM device, and there is nothing to cover it
with. Spelling the indices out removes the dependency entirely, and it costs
nothing anywhere else: neither the matrix nor the right-hand side can leave
local memory, because the kernel indexes both with loop variables of its own, so
this trades index loads for instruction count and not for registers. Measured on
DISCO-EB at N128: 528 ms table-driven, 419 ms with the solves unrolled, 398 ms
with the factorisation unrolled too, against 509 ms for the hand-written Schur
solver it replaced.

**No pivoting.** The pattern has to be fixed at compile time and the same in
every thread, so rows cannot be swapped on the numbers. Two things make that
sound here. The permutation is *symmetric*, so ``M``'s diagonal stays on the
diagonal and ``I/(h*gamma)`` guarantees every pivot is structurally there and
grows without bound as the step shrinks. And Rodas5P is a Rosenbrock-*W* method:
order 5 survives an approximate factorisation, so a badly conditioned pivot costs
step-size control rather than correctness -- and the controller is what notices.
A pivot that reaches exactly zero leaves an infinity in the factors, the error
norm goes to NaN, the step is rejected and the next attempt has a larger
``1/(h*gamma)`` on that diagonal. This is the same stance ``dense_lu_solver``
takes on a singular column, for the same reason.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, replace

import numpy as np
from numba_cuda_mlir import cuda

from menodax._codegen import compile_device_source
from menodax._sparsity import (
    CompressedJacobian,
    colour_sparsity,
    normalize_sparsity,
)

# AMD, or nothing at all. CHOLMOD's orderings were reachable here while
# scikit-sparse was a dependency, and measuring them is what retired it: over
# arrow, ring and bordered-block patterns "colamd", "nesdis" and "best" each
# returned AMD's own fill to the entry -- on a symmetric analysis CHOLMOD runs
# AMD for "colamd" anyway -- and "metis" was strictly worse wherever it
# differed, 660 nonzeros against 484 on the Einstein-Boltzmann-like case, losing
# the perfect elimination order AMD finds there. Nested dissection wins on
# meshes far larger than the tens-to-few-hundred variables this solver is for,
# and it was the only thing a package with no wheels was buying.
ORDERINGS = ("amd", "natural")


def fill_reducing_order(pattern: tuple[tuple[int, ...], ...], ordering: str = "amd"):
    """A permutation of the variables that keeps ``L + U`` small.

    AMD, on the symmetrised pattern ``S + S.T``.

    Why AMD rather than COLAMD, which is the other obvious candidate: COLAMD
    orders the *columns* so that fill stays bounded whatever row permutation
    partial pivoting later chooses. That is the right objective exactly when
    there will be pivoting -- and there will not be here, because the pattern is
    compiled into the kernel and cannot depend on the numbers. COLAMD's
    permutation is also one-sided, so it moves the diagonal off the diagonal,
    and this factorisation needs the diagonal precisely where ``I/(h*gamma)``
    puts it. AMD instead minimises (approximately) the fill of the Cholesky
    factor of ``S + S.T``, which is the standard bound on the fill of an
    unpivoted ``LU`` of ``S``, and it does so with a *symmetric* permutation
    ``P S P.T`` that leaves every diagonal entry on the diagonal. It is what
    UMFPACK and SuperLU use in their "symmetric mode" for the same reasons, and
    an iteration matrix ``I/(h*gamma) - J`` is about as close to structurally
    symmetric as an unsymmetric matrix gets.

    On DISCO-EB's Einstein-Boltzmann Jacobian this returns a *perfect*
    elimination order -- ``nnz(L + U) == nnz(J)``, not one entry of fill -- and
    the order it finds is the hand-written Schur solver's: peel each
    free-streaming multipole hierarchy from its truncated end inwards, then
    eliminate the densely coupled core last.

    SuiteSparse's AMD reaches this through cvxopt, which ships it in a
    manylinux wheel. It used to come through scikit-sparse's CHOLMOD bindings,
    which have no wheels and compile against SuiteSparse's headers -- so the
    default ordering worked only where someone had already run an `apt install`,
    and `pip install menodax` failed at that build. The two give the same fill
    on every pattern measured; where they differ it is in how they break ties
    between orders that are equally good.

    ``ordering`` is ``"amd"`` or ``"natural"``; see :data:`ORDERINGS` for what
    became of CHOLMOD's others.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"unknown ordering {ordering!r}; expected one of {ORDERINGS}")
    n = len(pattern)
    if ordering == "natural":
        return tuple(range(n))
    try:
        from cvxopt import amd, spmatrix
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the sparse direct solver orders its variables with SuiteSparse's "
            "AMD, which reaches it through cvxopt. Install it, or pass "
            "ordering='natural' to skip the ordering entirely."
        ) from exc

    # ``amd.order`` reads the *lower triangle* of a symmetric matrix and wants
    # the pattern alone, so the symmetrised pattern goes in with a unit on every
    # entry. Nothing has to be made positive definite for it, which CHOLMOD's
    # route did need: that reached the permutation through an actual
    # factorisation, and the numbers had to survive it.
    entries = {(r, c) for r, cs in enumerate(pattern) for c in cs}
    entries |= {(c, r) for r, c in entries}
    entries |= {(i, i) for i in range(n)}
    lower = sorted((r, c) for r, c in entries if r >= c)
    matrix = spmatrix(1.0, [r for r, _ in lower], [c for _, c in lower], (n, n))
    return tuple(int(i) for i in amd.order(matrix))



def fill_pattern(pattern: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """The exact ``L + U`` pattern of an unpivoted LU, as a boolean matrix.

    Gaussian elimination on the pattern alone: eliminating column ``k`` makes
    every row below it inherit row ``k``'s entries to the right of the diagonal.
    The result is exact -- it is what the numeric factorisation will touch, no
    more and no less -- which is why the footprint is taken this way rather than
    by factorising a sample matrix and counting. A sample cannot be exact: a
    coefficient that happens to vanish for those numbers, or an exact
    cancellation, drops an entry that another right-hand side needs, and the
    buffer is then one slot short in a kernel that has no way to say so. It is
    also cheaper, since it needs neither a plausible matrix nor a device.

    Bit-per-entry over the whole matrix, so the analysis is ``O(n ** 3 / 64)``
    time and ``O(n ** 2)`` bits. These are solvers for systems of tens to a few
    hundred variables (a 200-variable pattern analyses in single-digit
    milliseconds), so a sparse symbolic factorisation would buy nothing but code.
    """
    n = len(pattern)
    filled = np.zeros((n, n), dtype=bool)
    for row, cols in enumerate(pattern):
        filled[row, list(cols)] = True
    # I/(h*gamma) lands on every diagonal whether or not J has anything there,
    # so the diagonal is part of the pattern being factorised.
    np.fill_diagonal(filled, True)
    for k in range(n):
        below = np.flatnonzero(filled[k + 1 :, k]) + k + 1
        if below.size:
            filled[below, k + 1 :] |= filled[k, k + 1 :]
    return filled


@dataclass(frozen=True)
class SparseLULayout:
    """One CSR image of ``L + U``, in elimination order.

    CSR rather than CSC because every one of the three routines that reads this
    reads it *by rows*: the up-looking factorisation takes row ``i`` and
    subtracts multiples of the rows above it, the forward substitution is a dot
    product of row ``i`` of ``L`` with the solution so far, and the back
    substitution is the same over row ``i`` of ``U``. One row-major image serves
    all three; CSC would have to be transposed for two of them, and a
    column-oriented factorisation would still leave the solves wanting rows.

    ``L`` and ``U`` share the image -- ``L`` strictly left of the diagonal, ``U``
    from it rightwards -- because the factorisation is in place and a unit
    diagonal needs no storage. ``row_ptr``/``diag_ptr`` bracket the two halves of
    each row.

    Indices here are *elimination* indices: row ``i`` of this structure is
    variable ``order[i]`` of the caller's system. :attr:`row_origin` and
    :attr:`col_origin` carry the translation, so the device code never applies a
    permutation to a vector -- it visits the rows in elimination order and reads
    and writes the right-hand side where the caller left it.
    """

    n_vars: int
    order: tuple[int, ...]
    row_ptr: np.ndarray
    col_ind: np.ndarray
    diag_ptr: np.ndarray

    @property
    def nnz(self) -> int:
        """Elements in one trajectory's ``L + U``, i.e. the whole footprint."""
        return int(self.row_ptr[-1])

    @property
    def row_origin(self) -> np.ndarray:
        """Caller's index of each elimination row."""
        return np.asarray(self.order, dtype=np.int32)

    @property
    def col_origin(self) -> np.ndarray:
        """Caller's index of the column each slot belongs to."""
        return np.asarray(self.order, dtype=np.int32)[self.col_ind]

    def slot_table(self) -> np.ndarray:
        """``(n_vars, n_vars)`` of slots, ``-1`` where the structure has nothing."""
        table = np.full((self.n_vars, self.n_vars), -1, dtype=np.int64)
        origin = np.asarray(self.order, dtype=np.int64)
        for i in range(self.n_vars):
            start, stop = int(self.row_ptr[i]), int(self.row_ptr[i + 1])
            table[origin[i], origin[self.col_ind[start:stop]]] = np.arange(start, stop)
        return table

    def factorization_ops(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The rank-one updates, addressed off the multiplier that drives them.

        Up-looking LU visits row ``i`` and, for each entry ``(i, k)`` left of the
        diagonal in increasing ``k``, forms the multiplier ``L[i, k]`` and
        subtracts ``L[i, k] * U[k, k:]`` from row ``i``. With the pattern fixed,
        *which* slot each of those subtractions reads and writes is fixed too, so
        the whole inner merge -- the part a runtime sparse solver spends its time
        on, searching row ``i`` for the column it has to update -- is resolved
        here.

        The *sources* need no table at all: they are row ``k``'s upper entries,
        which CSR already holds contiguously, so the inner loop can simply run
        over ``pivot_of[p] + 1 .. upper_end[p]`` and read ``lu[s]`` straight.
        That leaves one index load per multiply-add rather than three, which is
        the whole cost of the inner loop besides the arithmetic.

        Returned as ``(dst_start, destination, upper_end)``, all indexed by the
        multiplier's own slot ``p``. The destinations for one multiplier are
        contiguous, and in the same order as the sources, so the inner loop walks
        ``destination`` from ``dst_start[p]`` alongside the sources.
        """
        n = self.n_vars
        dst_start = np.zeros(self.nnz, dtype=np.int64)
        upper_end = np.asarray(self.row_ptr[1:], dtype=np.int64)[
            np.asarray(self.col_ind, dtype=np.int64)
        ]
        dst: list[int] = []
        table = np.full((n, n), -1, dtype=np.int64)
        for i in range(n):
            start, stop = int(self.row_ptr[i]), int(self.row_ptr[i + 1])
            table[i, self.col_ind[start:stop]] = np.arange(start, stop)
        for i in range(n):
            for p in range(int(self.row_ptr[i]), int(self.diag_ptr[i])):
                k = int(self.col_ind[p])
                dst_start[p] = len(dst)
                for q in range(int(self.diag_ptr[k]) + 1, int(self.row_ptr[k + 1])):
                    dst.append(int(table[i, int(self.col_ind[q])]))
        return (
            dst_start.astype(np.int32),
            np.asarray(dst, dtype=np.int32),
            upper_end.astype(np.int32),
        )

    def flop_counts(self) -> tuple[int, int, int]:
        """``(multipliers, updates, substitutions)``.

        The first two are one factorisation; the third is one solve, both
        substitutions together. These decide whether the device code is emitted
        as straight-line or as loops, so they are counted from the structure
        rather than by building what they are counting.
        """
        lower = self.diag_ptr - self.row_ptr[:-1]
        upper = self.row_ptr[1:] - self.diag_ptr - 1
        multipliers = int(lower.sum())
        updates = int(upper[self.col_ind[_lower_mask(self)]].sum())
        return multipliers, updates, multipliers + int(upper.sum())


def _lower_mask(layout: SparseLULayout) -> np.ndarray:
    """True at the slots strictly left of their row's diagonal."""
    slot = np.arange(layout.nnz)
    row = np.repeat(np.arange(layout.n_vars), np.diff(layout.row_ptr))
    return slot < layout.diag_ptr[row]


def analyse(
    pattern: tuple[tuple[int, ...], ...], ordering: str = "amd"
) -> SparseLULayout:
    """Order, factorise symbolically, and lay out ``L + U`` in CSR.

    ``pattern`` is the normalised form, one tuple of column indices per row, as
    [`menodax._sparsity.normalize_sparsity`][] returns it.
    """
    n_vars = len(pattern)
    order = fill_reducing_order(pattern, ordering)
    # The elimination happens in the permuted matrix, so the pattern is
    # relabelled *before* the symbolic pass rather than after it: a permutation
    # of the fill is not the fill of the permutation, and the whole point of the
    # ordering is that it changes what fills in.
    forward = np.asarray(order, dtype=np.int64)
    inverse = np.empty(n_vars, dtype=np.int64)
    inverse[forward] = np.arange(n_vars)
    filled = fill_pattern(
        tuple(
            tuple(sorted(int(inverse[c]) for c in pattern[int(forward[i])]))
            for i in range(n_vars)
        )
    )
    row_ptr = np.zeros(n_vars + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(filled.sum(axis=1))
    col_ind = np.concatenate([np.flatnonzero(filled[i]) for i in range(n_vars)]).astype(
        np.int32
    )
    diag_ptr = np.array(
        [
            int(row_ptr[i])
            + int(np.searchsorted(col_ind[row_ptr[i] : row_ptr[i + 1]], i))
            for i in range(n_vars)
        ],
        dtype=np.int32,
    )
    return SparseLULayout(
        n_vars=n_vars,
        order=tuple(int(i) for i in order),
        row_ptr=row_ptr,
        col_ind=col_ind,
        diag_ptr=diag_ptr,
    )


def compressed_jacobian(
    pattern: tuple[tuple[int, ...], ...], layout: SparseLULayout
) -> CompressedJacobian:
    """The colour-compressed layout whose slots are ``layout``'s CSR slots.

    Colouring and storage are separate questions and this is where they meet.
    The colouring comes from the caller's pattern as it always does -- the sweeps
    cost ``n_colours + 1`` whatever happens to the storage -- but the slot each
    ``(row, colour)`` writes to is the factorisation's, so the AD deposits ``-J``
    directly into the buffer the factorisation will work in. Fill-in slots
    belong to no column of ``J`` and so are written by nobody, which is exactly
    what ``needs_clear`` in the kernel is for.

    A consequence worth knowing: the pattern may now be coloured as tightly as it
    really is. A structured solver that owned its own buffer had to declare its
    fill-in in the pattern too, since an entry with no slot had nowhere to go,
    and that cost colours -- DISCO-EB's 12 rather than 11. Here the fill has its
    own slots by construction, so only the true nonzeros need colouring.
    """
    compressed = colour_sparsity(pattern)
    table = layout.slot_table()
    packed = np.full(layout.n_vars * compressed.n_colours, -1, dtype=np.int64)
    for row, cols in enumerate(pattern):
        for col in cols:
            slot = int(table[row, col])
            if slot < 0:
                raise ValueError(
                    f"entry ({row}, {col}) is in the pattern but not in the "
                    "factorised structure, which cannot happen unless the two "
                    "were built from different patterns"
                )
            packed[row * compressed.n_colours + compressed.colour[col]] = slot
    diagonal = np.array([int(table[i, i]) for i in range(layout.n_vars)])
    return replace(
        compressed,
        packed=tuple(int(s) for s in packed),
        packed_diagonal=tuple(int(s) for s in diagonal),
        n_slots=layout.nnz,
    )


class SparseDirectSolver:
    """A Rodas5P linear solver compiled for one sparsity pattern.

    ``rodas5P.solve`` builds this itself from the ``sparsity`` it is given, so a
    caller never holds one; :attr:`compressed` is the layout it laid out and the
    layout the kernel then fills, which is what keeps the Enzyme sweeps and the
    factorisation from disagreeing about where an entry lives.

    It pivots nothing, so :attr:`ipiv_size` is 1 -- the kernel still allocates
    the array and the device functions still take it, because the dense solver
    needs it and the two have one shape.
    """

    ipiv_size = 1

    def __init__(self, layout: SparseLULayout, compressed: CompressedJacobian):
        self.layout = layout
        self.compressed = compressed
        # Resolved once and handed to both, since it is the expensive part of
        # the analysis and the factorisation is the only thing that needs it.
        ops = layout.factorization_ops()
        self.factorize_local = _make_factorize(layout, ops)
        self.solve_local = _make_solve(layout)

    @property
    def nnz(self) -> int:
        return self.layout.nnz

    def __repr__(self) -> str:
        multipliers, updates, substitutions = self.layout.flop_counts()
        return (
            f"<SparseDirectSolver n_vars={self.layout.n_vars} nnz={self.nnz} "
            f"factorise={multipliers + updates} solve={substitutions} "
            f"colours={self.compressed.n_colours}>"
        )


@functools.cache
def sparse_direct_solver_for(
    pattern: tuple[tuple[int, ...], ...], ordering: str = "amd"
) -> SparseDirectSolver:
    """Compile a direct sparse solver for a *normalised* pattern.

    This is what the kernel builder calls, since it has already normalised what
    the caller passed as ``sparsity``. Cached on the pattern, so asking twice
    gets the same compiled device functions and the kernel's own cache hits.
    """
    layout = analyse(pattern, ordering)
    return SparseDirectSolver(layout, compressed_jacobian(pattern, layout))


def sparse_direct_solver(sparsity, n_vars: int, *, ordering: str = "amd"):
    """Compile a direct sparse solver for ``sparsity``.

    ``sparsity`` takes the same spellings ``solve`` does: an ``(n_vars, n_vars)``
    mask, a scipy sparse matrix, or an ``(nnz, 2)`` index array. It must be a
    superset of the true nonzeros of ``J``, as for any pattern the kernel is
    given; it need *not* include the fill-in, which is what this computes.
    """
    return sparse_direct_solver_for(normalize_sparsity(sparsity, n_vars), ordering)


# --- the compiled device functions -------------------------------------------
# The index tables live in constant memory, where every thread of a warp reads
# the same address at the same time and the read broadcasts. Each device function
# binds *one* packed table: numba emits a separate copy of a closed-over array
# per reference site, and the 64 KiB constant window is shared with the colour
# seed table and everything the caller's right-hand side closes over.


# Above these many multiply-adds the routines stay table-driven. Straight-line
# code is faster wherever it fits -- it spends no broadcast load on an index the
# analysis already knows -- but it is instruction cache, and the solves are
# emitted eight times over, once per Rosenbrock stage. The factorisation appears
# once a step, so it may be the larger of the two.
MAX_UNROLLED_SUBSTITUTIONS = 2048
MAX_UNROLLED_UPDATES = 8192


def _make_factorize(layout: SparseLULayout, ops):
    """Up-looking LU of ``M``, in place, no pivoting.

    Row ``i`` is finished before row ``i + 1`` begins, so by the time the
    multiplier ``L[i, k]`` is formed, row ``k`` already holds its final ``U``.
    Within a row the entries left of the diagonal are visited in increasing
    column order, which is the order their own updates arrive in.

    The diagonal is left holding ``1 / U[i, i]``: it is divided by once here and
    multiplied by ``n_vars`` times per step in the back substitutions, eight of
    them.

    Small structures get this unrolled, every slot a literal, for the reasons
    ``_make_solve`` gives. Table-driven, the inner loop runs over row ``k``'s
    upper entries -- which CSR holds contiguously, so the *source* needs no index
    and only the destination costs a broadcast load per multiply-add.

    A zero pivot is left to produce an infinity rather than guarded against: the
    Rosenbrock-W property tolerates an approximate factorisation, the step
    controller rejects whatever comes out of one that is not, and the smaller
    step it then takes puts a larger ``1/(h*gamma)`` on this very diagonal.
    """
    n = layout.n_vars
    dst_start, destination, upper_end = ops
    # Slot of the pivot each multiplier divides by: diag_ptr[col_ind[p]],
    # resolved here so the factorisation never touches col_ind.
    pivot_of = layout.diag_ptr[layout.col_ind]

    if destination.size <= MAX_UNROLLED_UPDATES:
        return _unrolled_factorize(layout, dst_start, destination, pivot_of, upper_end)

    tables, (ROW, DIAG, PIVOT, START, END, DST) = _pack(
        layout.row_ptr, layout.diag_ptr, pivot_of, dst_start, upper_end, destination
    )

    @cuda.jit(device=True)
    def sparse_direct_factorize(lu, ipiv):
        t = cuda.const.array_like(tables)
        for i in range(n):
            pivot = t[DIAG + i]
            for p in range(t[ROW + i], pivot):
                above = t[PIVOT + p]
                factor = lu[p] * lu[above]
                lu[p] = factor
                q = DST + t[START + p]
                for s in range(above + 1, t[END + p]):
                    lu[t[q]] -= factor * lu[s]
                    q += 1
            lu[pivot] = 1.0 / lu[pivot]

    return sparse_direct_factorize


def _unrolled_factorize(layout, dst_start, destination, pivot_of, upper_end):
    """Generate the factorisation with every slot spelled out as a literal.

    Each multiplier gets its own name rather than reusing one, so the only
    dependencies left in the emitted code are the real ones and the compiler is
    free to schedule across them.
    """
    n = layout.n_vars
    row_ptr, diag_ptr = layout.row_ptr, layout.diag_ptr
    lines = ["def sparse_direct_factorize(lu, ipiv):"]
    for i in range(n):
        for p in range(int(row_ptr[i]), int(diag_ptr[i])):
            above = int(pivot_of[p])
            lines.append(f"    f{p} = lu[{p}] * lu[{above}]")
            lines.append(f"    lu[{p}] = f{p}")
            q = int(dst_start[p])
            for s in range(above + 1, int(upper_end[p])):
                lines.append(f"    lu[{int(destination[q])}] -= f{p} * lu[{s}]")
                q += 1
        pivot = int(diag_ptr[i])
        lines.append(f"    lu[{pivot}] = 1.0 / lu[{pivot}]")
    return compile_device_source("sparse_direct_factorize", lines)


def _make_solve(layout: SparseLULayout):
    """The two triangular solves: ``M x = rhs`` in place, sparse both ways.

    Each sweep touches only the nonzeros -- ``nnz(L) + nnz(U)`` multiply-adds
    against a dense solve's ``n_vars ** 2``. Rows are visited in elimination
    order and each reads and writes ``rhs`` at the *caller's* own index, so the
    permutation costs no gather, no scatter and no second vector. ``L`` has a
    unit diagonal and is not stored; ``U``'s diagonal was inverted by the
    factorisation, so the divide is a multiply.

    Small structures get the sweeps **unrolled**, every slot a literal. This is
    the eight-times-per-step routine and it is what the solver's cost is mostly
    made of, so the index loads are worth removing outright: table-driven, each
    multiply-add spends a broadcast load on the column it reads before it can
    issue the load of the value, and each row spends two more on its own bounds.
    That dependent pair is invisible on a full device, where other trajectories
    cover the latency, and is most of the cost when the ensemble is small enough
    to leave the SMs idle -- DISCO-EB's single-cosmology case is 128
    trajectories, four warps.

    Neither ``lu`` nor ``rhs`` can be promoted out of local memory by unrolling,
    so this trades index loads for instruction count and nothing else: the kernel
    itself indexes both with loop variables elsewhere.
    """
    n = layout.n_vars
    row_ptr, diag_ptr = layout.row_ptr, layout.diag_ptr
    origin, col_origin = layout.row_origin, layout.col_origin
    _, _, substitutions = layout.flop_counts()

    if substitutions <= MAX_UNROLLED_SUBSTITUTIONS:
        return _unrolled_solve(n, row_ptr, diag_ptr, origin, col_origin)

    tables, (ROW, DIAG, ROW_ORIG, COL_ORIG) = _pack(
        row_ptr, diag_ptr, origin, col_origin
    )

    @cuda.jit(device=True)
    def sparse_direct_solve(lu, ipiv, rhs):
        t = cuda.const.array_like(tables)
        for i in range(n):
            row = t[ROW_ORIG + i]
            acc = rhs[row]
            for p in range(t[ROW + i], t[DIAG + i]):
                acc -= lu[p] * rhs[t[COL_ORIG + p]]
            rhs[row] = acc
        for i in range(n - 1, -1, -1):
            row = t[ROW_ORIG + i]
            pivot = t[DIAG + i]
            acc = rhs[row]
            for p in range(pivot + 1, t[ROW + i + 1]):
                acc -= lu[p] * rhs[t[COL_ORIG + p]]
            rhs[row] = acc * lu[pivot]

    return sparse_direct_solve


def _unrolled_solve(n, row_ptr, diag_ptr, origin, col_origin):
    """Generate the two sweeps with every slot spelled out as a literal."""

    def terms(span):
        return "".join(
            f" - lu[{p}] * rhs[{col_origin[p]}]" for p in range(span.start, span.stop)
        )

    lines = ["def sparse_direct_solve(lu, ipiv, rhs):"]
    for i in range(n):
        row = int(origin[i])
        lower = range(int(row_ptr[i]), int(diag_ptr[i]))
        lines.append(f"    rhs[{row}] = rhs[{row}]{terms(lower)}")
    for i in range(n - 1, -1, -1):
        row, pivot = int(origin[i]), int(diag_ptr[i])
        upper = range(pivot + 1, int(row_ptr[i + 1]))
        lines.append(f"    rhs[{row}] = (rhs[{row}]{terms(upper)}) * lu[{pivot}]")
    return compile_device_source("sparse_direct_solve", lines)


def _pack(*arrays):
    """One int32 array of everything, plus each piece's offset into it.

    Every entry must be **non-negative**. A negative int32 read out of a
    constant array and then used in index arithmetic is promoted as though it
    were unsigned by numba-cuda-mlir (0.5.1): ``t[i] + 4`` with ``t[i] == -1``
    evaluates to ``2 ** 32 + 3``, so the address is wrong and nothing complains.
    Every table here is an offset into the buffer or into another table, so
    non-negativity is the natural form anyway; this is what keeps it that way.
    """
    offsets = np.cumsum([0] + [len(a) for a in arrays[:-1]])
    packed = np.concatenate([np.asarray(a, dtype=np.int32) for a in arrays])
    if packed.size and packed.min() < 0:
        raise ValueError("index tables must hold non-negative offsets only")
    return packed, tuple(int(o) for o in offsets)
