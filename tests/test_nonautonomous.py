"""Non-autonomous correctness tests for the Rodas5P solver.

For a Rosenbrock-Wanner method to retain its order on a non-autonomous ODE,
each stage must include the explicit ``dt * d_i * df/dt`` correction in
addition to evaluating ``f`` at the stage time.  These tests use a linear
non-autonomous problem with a closed-form solution to verify the
numba-cuda-mlir/CUDA Rodas5P implementation, and with it that the ``df/dt``
Enzyme derives alongside the Jacobian is right.

Reference ODE:  dy/dt = -lambda * y + forcing * t,   y(0) = 0
Closed form:    y(t) = forcing * (lambda * t - 1 + exp(-lambda t)) / lambda**2
"""

import jax
import numpy as np
import pytest
from numba_cuda_mlir import cuda

from modax.rodas5P import solve as rodas5Pnumba_solve

jax.config.update("jax_enable_x64", True)

LAMBDA = 10.0
FORCING = 5.0
T_FINAL = 1.0
N_SAVE = 11


def _exact(t, lam, forcing):
    return forcing * (lam * t - 1.0 + np.exp(-lam * t)) / (lam**2)


def ode_fn(y, t, p):
    return (-p[0] * y[0] + p[1] * t,)


def _setup():
    y0 = np.zeros((1, 1), dtype=np.float64)
    t_span = np.linspace(0.0, T_FINAL, N_SAVE, dtype=np.float64)
    params = np.array([[LAMBDA, FORCING]], dtype=np.float64)
    exact = _exact(t_span, LAMBDA, FORCING)
    return y0, t_span, params, exact


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")
def test_rodas5P_nonautonomous_matches_analytical():
    y0, t_span, params, exact = _setup()
    sol = rodas5Pnumba_solve(
        ode_fn,
        y0,
        t_span,
        params,
        rtol=1e-9,
        atol=1e-11,
        first_step=1e-3,
    )
    y_num = np.asarray(sol)[0, :, 0]
    err = np.max(np.abs(y_num - exact))
    # LU is float32 in rodas5P, so we cannot expect 1e-8 here.
    assert err < 1e-5, f"max error {err:.2e} too large (with dT correction)"


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")
def test_rodas5P_autonomous_unchanged():
    """An autonomous problem must still solve correctly: Enzyme derives
    ``df/dt = 0``, so every ``D_i * dT`` term contributes zero.
    """
    # Re-use the same RHS with zero forcing, so f = -lambda*y.
    y0 = np.ones((1, 1), dtype=np.float64)
    t_span = np.linspace(0.0, 1.0, 6, dtype=np.float64)
    params = np.array([[LAMBDA, 0.0]], dtype=np.float64)
    exact = np.exp(-LAMBDA * t_span)

    sol = rodas5Pnumba_solve(
        ode_fn,
        y0,
        t_span,
        params,
        rtol=1e-7,
        atol=1e-9,
        first_step=1e-3,
    )
    y_num = np.asarray(sol)[0, :, 0]
    err = np.max(np.abs(y_num - exact))
    assert err < 1e-5, f"max error {err:.2e} too large (autonomous default)"
