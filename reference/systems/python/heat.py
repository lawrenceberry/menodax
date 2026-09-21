"""Semi-discretized one-dimensional heat equation systems.

Heat equation — stiffness from grid refinement

M is tridiagonal with uniform entries -2 / dx^2 on the diagonal and 1 / dx^2 on
the off-diagonals. Eigenvalues range from about -pi^2 to -4 / dx^2, so the
stiffness ratio grows like O(N^2) with grid refinement.

PDE:
    du/dt = alpha d^2u/dx^2,  x in (0, 1),  u(0, t) = u(1, t) = 0

Grid:
    N interior points, dx = 1 / (N + 1), x_i = (i + 1) * dx

The initial condition y0_i = sin(pi x_i) is the first discrete eigenmode of M,
so the exact solution is y_i(t) = y0_i * exp(lambda_1 * alpha * t), where
lambda_1 = (2 / dx^2) * (cos(pi * dx) - 1). This lets tests validate against
the analytic answer rather than a second numerical solver.

The matrix-vector product has a cancellation hazard in fp32: each row computes a
second difference whose individual terms are much larger than the result for a
smooth sine mode. The benchmarks therefore keep matrix-vector evaluation in
fp64 and vary only LU precision.
"""

import jax.numpy as jnp
import numpy as np

from reference.systems.python._tuple_codegen import (
    linear_combination,
    make_tuple_callback,
)

TIMES = jnp.array((0.0, 0.025, 0.05, 0.075, 0.1), dtype=jnp.float64)


def make_system(n_vars):
    """Construct a semi-discretized 1D heat equation with n_vars interior points."""
    if n_vars < 3:
        raise ValueError(f"n_vars must be at least 3, got {n_vars}")

    dx = 1.0 / (n_vars + 1)
    inv_dx2 = 1.0 / dx**2
    M_np = (
        np.diag(-2.0 * inv_dx2 * np.ones(n_vars))
        + np.diag(inv_dx2 * np.ones(n_vars - 1), 1)
        + np.diag(inv_dx2 * np.ones(n_vars - 1), -1)
    )
    x = np.arange(1, n_vars + 1) * dx
    y0 = jnp.array(np.sin(np.pi * x), dtype=jnp.float64)

    ode_fn = make_tuple_callback(
        "ode_fn",
        [
            f"p[0] * ({linear_combination(M_np[i], range(n_vars))})"
            for i in range(n_vars)
        ],
    )

    return {
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "y0": y0,
    }


_DEFAULT = make_system(4)
ode_fn = _DEFAULT["ode_fn"]


def make_params(size, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def exact_solution(n_vars, times, params_batch):
    """Exact discrete heat-equation solution for the first sine eigenmode."""
    dx = 1.0 / (n_vars + 1)
    x = np.arange(1, n_vars + 1) * dx
    y0 = np.sin(np.pi * x)
    lambda_1 = (2.0 / dx**2) * (np.cos(np.pi * dx) - 1.0)
    alpha = np.asarray(params_batch)[:, 0]
    times_np = np.asarray(times)
    decay = np.exp(lambda_1 * alpha[:, None] * times_np[None, :])
    return y0[None, None, :] * decay[:, :, None]
