"""The linear-solver protocol: the default satisfies it like any caller's would.

``rodas5P.solve(..., linear_solver=...)`` replaces :func:`dense_lu_solver` with a
factorisation that can exploit structure in ``M = I/(h*gamma) - J``. The solver
here exploits nothing -- it is a second dense LU with partial pivoting over the
default dense layout -- which is the point: the built-in reaches the kernel
through this same protocol, so the two have to agree to solver tolerance.

Compressed layouts, the colouring that produces them, and the gradients that
survive both, live in ``test_sparsity.py``.
"""

import jax
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from solvers.rodas5P import (
    _DEFAULT_TRAJECTORIES_PER_BLOCK,
    check_linear_solver,
    dense_lu_solver,
    trajectories_per_block_or_default,
)
from solvers.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")


class DenseThreadLocalSolver:
    """The smallest object satisfying the protocol.

    It owns neither the buffer nor the Jacobian: the kernel hands it the
    thread's own matrix, already assembled in the dense layout, and it
    factorises and solves in place.
    """

    def __init__(self, n_vars: int):
        self.n_vars = int(n_vars)
        self.factorize_local = self._make_factorize()
        self.solve_local = self._make_solve()

    def _make_factorize(self):
        n = self.n_vars

        @cuda.jit(device=True)
        def factorize(lu, ipiv):
            for i in range(n):
                max_val = 0.0
                pivot_row = i
                for k in range(i, n):
                    val = abs(lu[k * n + i])
                    if val > max_val:
                        max_val = val
                        pivot_row = k
                ipiv[i] = pivot_row
                if max_val == 0.0:
                    continue
                if pivot_row != i:
                    for j in range(n):
                        tmp = lu[i * n + j]
                        lu[i * n + j] = lu[pivot_row * n + j]
                        lu[pivot_row * n + j] = tmp
                inv_pivot = 1.0 / lu[i * n + i]
                for k in range(i + 1, n):
                    factor = lu[k * n + i] * inv_pivot
                    lu[k * n + i] = factor
                    for j in range(i + 1, n):
                        lu[k * n + j] -= factor * lu[i * n + j]

        return factorize

    def _make_solve(self):
        n = self.n_vars

        @cuda.jit(device=True)
        def solve_device(lu, ipiv, rhs):
            for i in range(n):
                pivot = ipiv[i]
                if pivot != i:
                    tmp = rhs[i]
                    rhs[i] = rhs[pivot]
                    rhs[pivot] = tmp
                acc = rhs[i]
                for j in range(i):
                    acc -= lu[i * n + j] * rhs[j]
                rhs[i] = acc
            for i in range(n - 1, -1, -1):
                acc = rhs[i]
                for j in range(i + 1, n):
                    acc -= lu[i * n + j] * rhs[j]
                rhs[i] = acc / lu[i * n + i]

        return solve_device


ROBERTSON_TIMES = np.array((0.0, 1e-6, 1e-2, 1e2, 1e5), dtype=np.float64)
ROBERTSON_Y0 = np.array([[0.891, 0.1, 0.009]], dtype=np.float64)
ROBERTSON_PARAMS = np.array([[0.04, 1e4, 3e7]], dtype=np.float64)


def robertson(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


# Non-autonomous: dy/dt = -lam*y + forcing*t, y(0) = 0. Without the
# dt*d_i*df/dt stage correction a Rosenbrock-W method drops below order 5 here,
# so this is what checks df/dt reaches the stages.
LAMBDA = 10.0
FORCING = 5.0


def forced_decay(y, t, p):
    return (-p[0] * y[0] + p[1] * t,)


def _exact_forced(t):
    return FORCING * (LAMBDA * t - 1.0 + np.exp(-LAMBDA * t)) / LAMBDA**2


def test_default_solver_satisfies_the_protocol():
    """The built-in path is not privileged: it goes through the same check."""
    check_linear_solver(dense_lu_solver(4))


def test_protocol_is_enforced():
    with pytest.raises(TypeError, match="not a Rodas5P linear solver"):
        check_linear_solver(object())


def test_trajectories_per_block_or_default():
    """A warp by default; nothing on chip bounds an explicit request."""
    assert trajectories_per_block_or_default() == _DEFAULT_TRAJECTORIES_PER_BLOCK
    assert trajectories_per_block_or_default(8) == 8
    assert trajectories_per_block_or_default(64) == 64
    with pytest.raises(ValueError, match="must be positive"):
        trajectories_per_block_or_default(0)


@requires_cuda
def test_custom_linear_solver_matches_builtin():
    """A caller's own dense LU must reproduce the built-in one on a stiff system."""
    kw = dict(rtol=1e-10, atol=1e-12, first_step=1e-8, lu_precision="fp64")
    builtin = np.asarray(
        rodas5P_solve(robertson, ROBERTSON_Y0, ROBERTSON_TIMES, ROBERTSON_PARAMS, **kw)
    )
    custom = np.asarray(
        rodas5P_solve(
            robertson,
            ROBERTSON_Y0,
            ROBERTSON_TIMES,
            ROBERTSON_PARAMS,
            linear_solver=DenseThreadLocalSolver(3),
            **kw,
        )
    )
    assert np.isfinite(custom).all()
    # Independent factorisations and step sequences, so agreement is to solver
    # tolerance rather than bitwise.
    assert np.allclose(custom, builtin, rtol=1e-6, atol=1e-9)


@requires_cuda
def test_custom_solver_keeps_the_time_derivative_term():
    t_span = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    y0 = np.zeros((1, 1), dtype=np.float64)
    params = np.array([[LAMBDA, FORCING]], dtype=np.float64)
    result = np.asarray(
        rodas5P_solve(
            forced_decay,
            y0,
            t_span,
            params,
            linear_solver=DenseThreadLocalSolver(1),
            rtol=1e-10,
            atol=1e-12,
            first_step=1e-4,
            lu_precision="fp64",
        )
    )[0, :, 0]
    assert np.allclose(result, _exact_forced(t_span), rtol=1e-8, atol=1e-10)


@requires_cuda
def test_tf_index_ends_each_trajectory_at_its_own_time():
    """Save times past a trajectory's own end time hold its final state."""
    t_span = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    # The third parameter column is the per-trajectory end time.
    params = np.array(
        [[LAMBDA, FORCING, 1.0], [LAMBDA, FORCING, 0.5]], dtype=np.float64
    )
    y0 = np.zeros((2, 1), dtype=np.float64)
    result = np.asarray(
        rodas5P_solve(
            forced_decay,
            y0,
            t_span,
            params,
            linear_solver=DenseThreadLocalSolver(1),
            tf_index=2,
            rtol=1e-10,
            atol=1e-12,
            first_step=1e-4,
            lu_precision="fp64",
        )
    )
    exact = _exact_forced(t_span)
    assert np.allclose(result[0, :, 0], exact, rtol=1e-8, atol=1e-10)
    stops_at = int(np.searchsorted(t_span, 0.5, side="right"))
    assert np.allclose(result[1, :stops_at, 0], exact[:stops_at], rtol=1e-8, atol=1e-10)
    held = result[1, stops_at:, 0]
    assert np.allclose(held, result[1, stops_at - 1, 0], rtol=0, atol=0)
