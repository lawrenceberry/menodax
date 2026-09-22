"""The compiled sparse direct solver: the analysis, and what the kernel does with it.

``menodax/_sparse_direct.py`` turns a sparsity pattern into a fill-reducing
order, an exact symbolic factorisation, a CSR layout and two ``cuda.jit`` device
functions. Most of that is host-side and testable without a GPU, so most of these
tests are: the ordering, the fill, the footprint, and a replay of the compiled
index tables in NumPy that checks they really do factorise the matrix. The GPU
tests then check the same thing end to end: the kernel builds this solver
whenever ``solve`` is given a ``sparsity``, so the answer must be the one the
dense LU gives without a pattern.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from menodax._sparse_direct import (
    ORDERINGS,
    SparseDirectSolver,
    analyse,
    compressed_jacobian,
    fill_pattern,
    sparse_direct_solver,
)
from menodax._sparsity import normalize_sparsity
from menodax.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")

try:  # the ordering needs SuiteSparse through scikit-sparse
    from sksparse.cholmod import cho_factor as _cho_factor  # noqa: F401

    HAVE_SKSPARSE = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_SKSPARSE = False

requires_sksparse = pytest.mark.skipif(
    not HAVE_SKSPARSE, reason="scikit-sparse (SuiteSparse CHOLMOD) required"
)


# --- the patterns these tests are about ---------------------------------------
def _arrow(n):
    """Dense first row and column: the worst case for the natural order.

    Eliminating variable 0 first couples every other variable to every other
    variable, so ``L + U`` is full. Eliminating it *last* costs nothing, which is
    what a minimum-degree ordering finds.
    """
    mask = np.eye(n, dtype=bool)
    mask[0, :] = mask[:, 0] = True
    return mask


def _bordered_block(core, tails, length):
    """A miniature Einstein-Boltzmann Jacobian: dense core, tridiagonal borders.

    Each tail is tridiagonal and meets the core at one variable, in both
    directions -- the structure of a free-streaming multipole hierarchy hanging
    off the quadrupole it streams out of.
    """
    n = core + tails * length
    mask = np.zeros((n, n), dtype=bool)
    mask[:core, :core] = True
    for t in range(tails):
        base = core + t * length
        parent = t % core
        for i in range(length):
            mask[base + i, base + i] = True
            if i:
                mask[base + i, base + i - 1] = True
            if i + 1 < length:
                mask[base + i, base + i + 1] = True
        mask[base, parent] = mask[parent, base] = True
    return mask


def _ring(n):
    """Periodic nearest-neighbour coupling; the wrap-around makes it fill in."""
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        mask[i, i] = mask[i, (i + 1) % n] = mask[i, (i - 1) % n] = True
    return mask


def _matrix(mask, seed=0, shift=60.0):
    """A diagonally dominant matrix with exactly that pattern."""
    rng = np.random.default_rng(seed)
    n = len(mask)
    dense = np.where(mask, rng.normal(size=(n, n)), 0.0)
    dense[np.diag_indices(n)] += shift
    return dense


def _replay(layout, dense):
    """Run the compiled index tables in NumPy, exactly as the device does."""
    n = layout.n_vars
    row_ptr, col_ind, diag_ptr = layout.row_ptr, layout.col_ind, layout.diag_ptr
    dst_start, destination, upper_end = layout.factorization_ops()
    pivot_of = diag_ptr[col_ind]
    origin, col_origin = layout.row_origin, layout.col_origin

    lu = np.zeros(layout.nnz)
    table = layout.slot_table()
    rows, cols = np.nonzero(table >= 0)
    lu[table[rows, cols]] = dense[rows, cols]

    for i in range(n):
        pivot = diag_ptr[i]
        for p in range(row_ptr[i], pivot):
            above = pivot_of[p]
            factor = lu[p] * lu[above]
            lu[p] = factor
            q = dst_start[p]
            for s in range(above + 1, upper_end[p]):
                lu[destination[q]] -= factor * lu[s]
                q += 1
        lu[pivot] = 1.0 / lu[pivot]

    def solve(b):
        x = np.asarray(b, dtype=np.float64).copy()
        for i in range(n):
            row = origin[i]
            acc = x[row]
            for p in range(row_ptr[i], diag_ptr[i]):
                acc -= lu[p] * x[col_origin[p]]
            x[row] = acc
        for i in range(n - 1, -1, -1):
            row, pivot = origin[i], diag_ptr[i]
            acc = x[row]
            for p in range(pivot + 1, row_ptr[i + 1]):
                acc -= lu[p] * x[col_origin[p]]
            x[row] = acc * lu[pivot]
        return x

    return solve


# --- the symbolic factorisation ----------------------------------------------
def test_fill_pattern_is_exactly_what_an_unpivoted_lu_touches():
    """Nothing the numbers reach is missing, and nothing unreachable is reserved."""
    mask = _ring(9)
    predicted = fill_pattern(normalize_sparsity(mask, 9))
    dense = _matrix(mask)

    # Unpivoted Gaussian elimination, keeping the factors in place.
    work = dense.copy()
    for k in range(9):
        work[k + 1 :, k] /= work[k, k]
        work[k + 1 :, k + 1 :] -= np.outer(work[k + 1 :, k], work[k, k + 1 :])
    touched = work != 0.0

    assert (touched <= predicted).all(), "the factorisation wrote outside the pattern"
    # A random matrix makes no accidental zeros, so the two must coincide
    # exactly -- which is the argument for taking the footprint symbolically
    # rather than from a sample: a sample that *did* make one would come out short.
    assert (touched == predicted).all()


@requires_sksparse
def test_amd_undoes_the_worst_case_for_the_natural_order():
    mask = _arrow(12)
    pattern = normalize_sparsity(mask, 12)
    natural = analyse(pattern, "natural")
    amd = analyse(pattern, "amd")

    assert natural.nnz == 12 * 12  # eliminating the hub first fills everything
    assert amd.nnz == int(mask.sum())  # eliminating it last fills nothing
    assert amd.order[-1] == 0


@requires_sksparse
def test_a_bordered_block_system_needs_no_fill_at_all():
    """The structure DISCO-EB has: minimum degree rediscovers its Schur solver.

    Each tail is peeled from its far end, where every variable has degree two, so
    nothing fills; the densely coupled core is eliminated last, where it was
    already dense.
    """
    mask = _bordered_block(core=5, tails=3, length=7)
    pattern = normalize_sparsity(mask, len(mask))
    amd = analyse(pattern, "amd")

    assert amd.nnz == int(mask.sum())
    # and the densely coupled core is what is left until last
    assert set(amd.order[-5:]) == set(range(5))


def test_the_ordering_is_validated():
    with pytest.raises(ValueError, match="unknown ordering"):
        analyse(normalize_sparsity(_ring(4), 4), "minimum-degree-ish")
    assert "amd" in ORDERINGS


# --- the layout ---------------------------------------------------------------
@pytest.mark.parametrize(
    "mask",
    [_ring(9), _arrow(8), _bordered_block(4, 2, 5), np.ones((5, 5), dtype=bool)],
    ids=["ring", "arrow", "bordered", "dense"],
)
def test_layout_is_a_consistent_csr_image(mask):
    n = len(mask)
    layout = analyse(normalize_sparsity(mask, n), "natural")

    assert layout.row_ptr[0] == 0 and layout.row_ptr[-1] == layout.nnz
    assert (np.diff(layout.row_ptr) >= 1).all()  # the diagonal is always there
    for i in range(n):
        span = layout.col_ind[layout.row_ptr[i] : layout.row_ptr[i + 1]]
        assert (np.diff(span) > 0).all(), "columns must ascend within a row"
        assert layout.col_ind[layout.diag_ptr[i]] == i
    # every declared nonzero has a slot of its own, inside the buffer
    table = layout.slot_table()
    slots = [int(table[r, c]) for r, c in zip(*np.nonzero(mask))]
    assert min(slots) >= 0
    assert len(set(slots)) == len(slots)
    assert max(slots) < layout.nnz


@pytest.mark.parametrize("ordering", ["natural", "amd"])
@pytest.mark.parametrize(
    "mask",
    [_ring(9), _arrow(8), _bordered_block(4, 2, 5), np.ones((5, 5), dtype=bool)],
    ids=["ring", "arrow", "bordered", "dense"],
)
def test_the_compiled_tables_factorise_the_matrix(mask, ordering):
    """Replay the device algorithm in NumPy: same tables, same loops.

    This is where a mistake in the ordering, the fill, the slot map or the
    grouping of the rank-one updates shows up, without a GPU in the way.
    """
    if ordering == "amd" and not HAVE_SKSPARSE:
        pytest.skip("scikit-sparse required")
    n = len(mask)
    layout = analyse(normalize_sparsity(mask, n), ordering)
    dense = _matrix(mask)
    solve = _replay(layout, dense)

    rng = np.random.default_rng(1)
    for _ in range(3):
        b = rng.normal(size=n)
        np.testing.assert_allclose(solve(b), np.linalg.solve(dense, b), rtol=1e-10)


def test_the_footprint_is_the_factorisation_and_nothing_more():
    mask = _ring(9)
    pattern = normalize_sparsity(mask, 9)
    layout = analyse(pattern, "natural")
    compressed = compressed_jacobian(pattern, layout)

    assert compressed.size == layout.nnz
    # The sweeps write the pattern's entries; the fill-in belongs to no column of
    # J, so nothing writes it and the kernel has to clear the buffer.
    claimed = int((compressed.store_slots() >= 0).sum())
    assert claimed == int(mask.sum()) < compressed.size
    # every diagonal has a slot, including any the pattern left out
    assert len(set(compressed.diagonal)) == 9


def test_the_pattern_may_be_coloured_as_tightly_as_it_is():
    """Storage is the factorisation's, so the colouring need not reserve fill.

    A solver that owned its buffer had to declare its fill-in in the pattern, and
    paid colours for it. Here the fill has slots of its own by construction.
    """
    mask = _bordered_block(core=5, tails=3, length=7)
    pattern = normalize_sparsity(mask, len(mask))
    compressed = compressed_jacobian(pattern, analyse(pattern, "natural"))
    for row, cols in enumerate(pattern):
        assert len({compressed.colour[c] for c in cols}) == len(cols)


# --- the solver object --------------------------------------------------------
def test_the_solver_is_the_layout_and_the_two_device_functions():
    solver = sparse_direct_solver(_ring(6), 6, ordering="natural")
    assert isinstance(solver, SparseDirectSolver)
    assert callable(solver.factorize_local) and callable(solver.solve_local)
    assert solver.ipiv_size == 1  # it pivots nothing
    assert solver.nnz == solver.compressed.size


def test_it_is_cached_on_the_pattern():
    """Two calls must give the same object, or the kernel cache misses."""
    a = sparse_direct_solver(_ring(6), 6, ordering="natural")
    b = sparse_direct_solver(
        np.array(sorted(zip(*np.nonzero(_ring(6))))), 6, ordering="natural"
    )
    assert a is b


# --- what the kernel does with it ---------------------------------------------
ROBERTSON_TIMES = np.array((0.0, 1e-6, 1e-2, 1e2, 1e5), dtype=np.float64)
ROBERTSON_Y0 = np.array([[0.891, 0.1, 0.009]], dtype=np.float64)
ROBERTSON_PARAMS = np.array([[0.04, 1e4, 3e7]], dtype=np.float64)


def robertson(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


def ring_ode(y, t, p):
    """Six-point periodic diffusion with decay. Its LU fills in."""
    return (
        p[0] * (y[1] - 2.0 * y[0] + y[5]) - p[1] * y[0],
        p[0] * (y[2] - 2.0 * y[1] + y[0]) - p[1] * y[1],
        p[0] * (y[3] - 2.0 * y[2] + y[1]) - p[1] * y[2],
        p[0] * (y[4] - 2.0 * y[3] + y[2]) - p[1] * y[3],
        p[0] * (y[5] - 2.0 * y[4] + y[3]) - p[1] * y[4],
        p[0] * (y[0] - 2.0 * y[5] + y[4]) - p[1] * y[5],
    )


@requires_cuda
@requires_sksparse
def test_a_dense_pattern_is_the_uninformative_end_of_the_same_mechanism():
    """No structure to find, so it must simply agree with the dense LU."""
    kw = dict(rtol=1e-10, atol=1e-12, first_step=1e-8, lu_precision="fp64")
    builtin = np.asarray(
        rodas5P_solve(robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_PARAMS, **kw)
    )
    got = np.asarray(
        rodas5P_solve(
            robertson,
            ROBERTSON_Y0,
            ROBERTSON_TIMES,
            ROBERTSON_PARAMS,
            sparsity=np.ones((3, 3), dtype=bool),
            **kw,
        )
    )
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, builtin, rtol=1e-6, atol=1e-9)


@requires_cuda
@requires_sksparse
def test_a_pattern_with_fill_in_solves_the_same():
    """The fill slots are written by nobody, so the kernel must clear them."""
    solver = sparse_direct_solver(_ring(6), 6)
    assert solver.nnz > int(_ring(6).sum()), "this pattern is supposed to fill in"

    y0 = np.zeros((64, 6))
    y0[:, 0] = 1.0
    params = np.tile(np.array([50.0, 1.0]), (64, 1))
    t = np.array([0.0, 0.01, 0.2])
    kw = dict(rtol=1e-11, atol=1e-13, first_step=1e-6, lu_precision="fp64")

    builtin = np.asarray(rodas5P_solve(ring_ode, y0, t, params, **kw))
    got = np.asarray(rodas5P_solve(ring_ode, y0, t, params, sparsity=_ring(6), **kw))
    np.testing.assert_allclose(got, builtin, rtol=1e-8, atol=1e-11)


@requires_cuda
@requires_sksparse
def test_the_table_driven_form_solves_what_the_unrolled_one_does(monkeypatch):
    """Both emissions of the same analysis, so they must agree exactly.

    The unrolled form is what every structure small enough gets, which is every
    structure in this file; the thresholds are lowered here so the loops over the
    index tables are exercised at all.
    """
    import menodax._sparse_direct as module
    import menodax.rodas5P as rodas5P

    y0 = np.zeros((64, 6))
    y0[:, 0] = 1.0
    params = np.tile(np.array([50.0, 1.0]), (64, 1))
    t = np.array([0.0, 0.01, 0.2])
    kw = dict(rtol=1e-11, atol=1e-13, first_step=1e-6, lu_precision="fp64")

    unrolled = np.asarray(
        rodas5P_solve(ring_ode, y0, t, params, sparsity=_ring(6), **kw)
    )

    monkeypatch.setattr(module, "MAX_UNROLLED_SUBSTITUTIONS", 0)
    monkeypatch.setattr(module, "MAX_UNROLLED_UPDATES", 0)
    # Both caches key on the pattern, and the pattern has not changed.
    module.sparse_direct_solver_for.cache_clear()
    rodas5P._make_kernel.cache_clear()
    rodas5P._make_jax_launch.cache_clear()
    got = np.asarray(rodas5P_solve(ring_ode, y0, t, params, sparsity=_ring(6), **kw))

    # Same factorisation to the last bit, so the step sequences coincide too.
    np.testing.assert_array_equal(got, unrolled)


@requires_cuda
@requires_sksparse
def test_gradients_survive_the_sparse_solver():
    """The joint system only ever asks the solver for the n_vars block."""
    y0 = jnp.zeros((1, 6)).at[0, 0].set(1.0)
    t = jnp.asarray([0.0, 0.2])
    kw = dict(rtol=1e-11, atol=1e-13, first_step=1e-6, lu_precision="fp64")

    def loss(p, **extra):
        out = rodas5P_solve(ring_ode, y0, t, jnp.asarray([p]), **kw, **extra)
        return jnp.sum(out[0, -1] ** 2)

    p0 = jnp.asarray([50.0, 1.0])
    builtin = jax.grad(loss)(p0)
    got = jax.grad(loss)(p0, sparsity=_ring(6))
    np.testing.assert_allclose(np.asarray(got), np.asarray(builtin), rtol=1e-7)
