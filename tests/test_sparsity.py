"""Colour-compressed Jacobians: the colouring, and what the kernel does with it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from solvers._sparsity import (
    CompressedJacobian,
    colour_sparsity,
    dense_jacobian,
    normalize_sparsity,
    pack,
)
from solvers.rodas5P import solve

jax.config.update("jax_enable_x64", True)

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")


def _mask(n, pairs):
    m = np.zeros((n, n), dtype=bool)
    for r, c in pairs:
        m[r, c] = True
    return m


# --- the chain: dy_i/dt moves mass from i to i+1 ------------------------------
# Its Jacobian is lower bidiagonal, so two colours suffice for four columns.
N = 4
CHAIN = _mask(N, [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)])


def chain_ode(y, t, p):
    return (
        -p[0] * y[0],
        p[0] * y[0] - p[1] * y[1],
        p[1] * y[1] - p[2] * y[2],
        p[2] * y[2],
    )


def _chain_solver(compressed):
    """M is lower bidiagonal here, so forward substitution is the whole solve."""
    diag = tuple(compressed.slot(i, i) for i in range(N))
    sub = tuple(compressed.slot(i, i - 1) if i else 0 for i in range(N))

    class Bidiagonal:
        ipiv_size = 1

    @cuda.jit(device=True)
    def factorize_local(lu, ipiv):
        pass  # nothing to factorise

    @cuda.jit(device=True)
    def solve_local(lu, ipiv, rhs):
        rhs[0] = rhs[0] / lu[diag[0]]
        for i in range(1, N):
            rhs[i] = (rhs[i] - lu[sub[i]] * rhs[i - 1]) / lu[diag[i]]

    solver = Bidiagonal()
    solver.factorize_local = factorize_local
    solver.solve_local = solve_local
    return solver


def _chain_exact(t, a=0.7, b=0.4, c=0.2):
    y0 = np.exp(-a * t)
    y1 = a / (b - a) * (np.exp(-a * t) - np.exp(-b * t))
    y2 = (
        a
        * b
        * (
            np.exp(-a * t) / ((b - a) * (c - a))
            + np.exp(-b * t) / ((a - b) * (c - b))
            + np.exp(-c * t) / ((a - c) * (b - c))
        )
    )
    return np.array([y0, y1, y2, 1.0 - y0 - y1 - y2])


# --- the colouring itself -----------------------------------------------------
def test_tridiagonal_needs_three_colours():
    n = 6
    band = _mask(n, [(i, j) for i in range(n) for j in (i - 1, i, i + 1) if 0 <= j < n])
    compressed = colour_sparsity(normalize_sparsity(band, n))
    assert compressed.n_colours == 3
    assert compressed.size == n * 3
    assert not compressed.is_dense


def test_no_pattern_is_the_dense_matrix():
    """The dense path is the uninformative end of the same mechanism."""
    dense = dense_jacobian(4)
    assert dense.is_dense and dense.n_colours == 4
    # slot(r, c) must be exactly row-major.
    assert [dense.slot(r, c) for r in range(4) for c in range(4)] == list(range(16))
    assert dense.diagonal == (0, 5, 10, 15)


def test_colouring_is_checked_for_structural_orthogonality():
    """Two columns of one colour sharing a row would silently share a slot."""
    bad = CompressedJacobian(n_vars=2, n_colours=1, colour=(0, 0))
    from solvers._sparsity import _check_orthogonal

    with pytest.raises(ValueError, match="not structurally orthogonal"):
        _check_orthogonal(((0, 1), ()), bad)


def test_seed_table_covers_the_parameter_direction():
    """The zero row doubles as the null parameter seed, so it must be wide enough."""
    compressed = colour_sparsity(normalize_sparsity(CHAIN, N))
    seeds = compressed.seed_table(min_width=9)
    assert seeds.shape == (compressed.n_colours + 1, 9)
    assert not seeds[-1].any()
    for c, g in enumerate(compressed.colour):
        assert seeds[g, c] == 1.0


def test_sparsity_accepts_several_spellings():
    from scipy import sparse as sp

    expected = normalize_sparsity(CHAIN, N)
    pairs = np.array(sorted(zip(*np.nonzero(CHAIN))), dtype=np.int64)
    assert normalize_sparsity(pairs, N) == expected
    assert normalize_sparsity(sp.csr_array(CHAIN.astype(np.int8)), N) == expected
    with pytest.raises(ValueError, match="lies outside"):
        normalize_sparsity(np.array([[0, 99]]), N)


# --- what the kernel does with it --------------------------------------------
@requires_cuda
def test_compressed_solve_matches_dense_and_exact():
    compressed = colour_sparsity(normalize_sparsity(CHAIN, N))
    assert compressed.n_colours == 2  # four columns, two sweeps

    y0 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (64, 1))
    p = np.tile(np.array([0.7, 0.4, 0.2]), (64, 1))
    t = np.array([0.0, 0.5, 2.0])
    kw = dict(rtol=1e-11, atol=1e-13, first_step=1e-4, lu_precision="fp64")

    dense = np.asarray(solve(chain_ode, y0, t, p, **kw))
    sparse = np.asarray(
        solve(
            chain_ode,
            y0,
            t,
            p,
            sparsity=CHAIN,
            linear_solver=_chain_solver(compressed),
            **kw,
        )
    )
    np.testing.assert_allclose(sparse, dense, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(sparse[0, -1], _chain_exact(2.0), rtol=1e-9, atol=1e-11)


@requires_cuda
def test_sparsity_alone_still_solves():
    """A pattern buys the sweeps; a linear solver is a separate decision.

    With none given the compressed sweeps are scattered into a dense matrix and
    dense_lu_solver takes it from there, so the answer must match the run that
    never compressed anything.
    """
    y0 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (64, 1))
    p = np.tile(np.array([0.7, 0.4, 0.2]), (64, 1))
    t = np.array([0.0, 0.5, 2.0])
    kw = dict(rtol=1e-11, atol=1e-13, first_step=1e-4, lu_precision="fp64")

    dense = np.asarray(solve(chain_ode, y0, t, p, **kw))
    expanded = np.asarray(solve(chain_ode, y0, t, p, sparsity=CHAIN, **kw))
    np.testing.assert_allclose(expanded, dense, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        expanded[0, -1], _chain_exact(2.0), rtol=1e-9, atol=1e-11
    )


def test_dense_slots_place_every_entry():
    """Every declared nonzero lands where a row-major matrix wants it."""
    compressed = colour_sparsity(normalize_sparsity(CHAIN, N))
    table = compressed.dense_slots()

    assert table.shape == (N * compressed.n_colours,)
    nonzeros = {(int(r), int(c)) for r, c in zip(*np.nonzero(CHAIN))}
    for row, col in nonzeros:
        assert table[compressed.slot(row, col)] == row * N + col
    # and nothing the pattern did not declare is claimed
    assert {int(d) for d in table if d >= 0} == {r * N + c for r, c in nonzeros}


def test_dense_layout_needs_no_expansion():
    """A colouring that already is the dense matrix is left alone."""
    assert dense_jacobian(N).is_row_major_dense
    assert not colour_sparsity(normalize_sparsity(CHAIN, N)).is_row_major_dense
    # A permuted colouring uses n_vars colours but is not the dense matrix.
    permuted = CompressedJacobian(n_vars=3, n_colours=3, colour=(1, 2, 0))
    assert permuted.is_dense and not permuted.is_row_major_dense


@requires_cuda
def test_gradients_survive_compression_and_a_custom_solver():
    """The joint system only ever asks the solver for the n_vars block."""
    compressed = colour_sparsity(normalize_sparsity(CHAIN, N))
    y0 = jnp.asarray([[1.0, 0.0, 0.0, 0.0]])
    t = jnp.asarray([0.0, 2.0])
    kw = dict(rtol=1e-11, atol=1e-13, first_step=1e-4, lu_precision="fp64")

    def loss(p, **extra):
        out = solve(chain_ode, y0, t, jnp.asarray([p]), **kw, **extra)
        return jnp.sum(out[0, -1] ** 2)

    p0 = jnp.asarray([0.7, 0.4, 0.2])
    dense = jax.grad(loss)(p0)
    sparse = jax.grad(loss)(p0, sparsity=CHAIN, linear_solver=_chain_solver(compressed))
    np.testing.assert_allclose(np.asarray(sparse), np.asarray(dense), rtol=1e-9)


# --- packed: the same addressing, without the holes --------------------------
def test_pack_keeps_one_slot_per_entry():
    grid = colour_sparsity(normalize_sparsity(CHAIN, N))
    packed = pack(grid)

    assert grid.size == N * grid.n_colours == 8
    assert packed.size == int(CHAIN.sum()) + 1 == 7  # +1 for the (3,3) diagonal
    # every declared entry still has a distinct slot, and they are row-major
    slots = {(r, c): packed.slot(r, c) for r, c in zip(*np.nonzero(CHAIN))}
    assert sorted(slots.values()) == list(range(len(slots)))
    # row-major: a row's entries all precede the next row's
    by_row = [[s for (r, _), s in slots.items() if r == row] for row in range(N)]
    assert all(max(a) < min(b) for a, b in zip(by_row, by_row[1:]) if a and b)
    # and an entry the pattern never declared no longer aliases someone else's
    with pytest.raises(ValueError, match="not in the pattern"):
        packed.slot(0, 3)


def test_pack_finds_room_for_an_undeclared_diagonal():
    """I/(h*gamma) lands on diagonals J leaves structurally zero -- like (3,3)."""
    packed = pack(colour_sparsity(normalize_sparsity(CHAIN, N)))

    assert not CHAIN[3, 3]  # y3' = p2 y2, so J has nothing there
    assert packed.size == int(CHAIN.sum()) + 1 == 7
    assert len(set(packed.diagonal)) == N  # every diagonal has its own slot


@requires_cuda
def test_packed_layout_solves_the_same():
    """One slot per entry rather than one per (row, colour), same answer."""
    packed = pack(colour_sparsity(normalize_sparsity(CHAIN, N)))
    y0 = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (64, 1))
    p = np.tile(np.array([0.7, 0.4, 0.2]), (64, 1))
    t = np.array([0.0, 0.5, 2.0])
    kw = dict(rtol=1e-11, atol=1e-13, first_step=1e-4, lu_precision="fp64")

    dense = np.asarray(solve(chain_ode, y0, t, p, **kw))
    got = np.asarray(
        solve(
            chain_ode,
            y0,
            t,
            p,
            sparsity=packed,
            linear_solver=_chain_solver(packed),
            **kw,
        )
    )
    np.testing.assert_allclose(got, dense, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got[0, -1], _chain_exact(2.0), rtol=1e-9, atol=1e-11)
