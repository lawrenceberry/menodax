"""The Enzyme-derived Jacobian must match JAX's own, system by system.

``rodas5P`` takes no Jacobian: it forward-differentiates the right-hand side
with Enzyme instead. The reference systems no longer carry hand-written
Jacobians either, so the check on that derivation is JAX's forward-mode AD of
the very same ``ode_fn`` -- an independent differentiation pipeline, even
though it reads the same equations. It also covers the ``df/dt`` the solver
takes from the same sweeps, which is zero for every reference system, all of
them being autonomous.

The derivative is built here exactly as ``rodas5P._make_kernel`` builds it --
one directional derivative, seeded with a unit column at a time rather than
with a colour group -- so this pins the call shape the kernel depends on as
well as the values.

"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.benchmark_helpers import parametrize_system_cases

cuda = pytest.importorskip("numba_cuda_mlir.cuda")

pytestmark = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")


def evaluate_derivatives(ode_fn, y, t, params):
    """Return ``(jacobian, time_jacobian)`` from the solver's Enzyme derivative.

    ``y`` is ``(n, n_vars)`` and ``params`` is ``(n, n_params)``; the results
    are ``(n, n_vars, n_vars)`` and ``(n, n_vars)``.
    """
    from numba_cuda_mlir import types

    from modax._sensitivity import make_tangent

    y = np.ascontiguousarray(y, dtype=np.float64)
    params = np.ascontiguousarray(params, dtype=np.float64)
    n, n_vars = y.shape
    n_params = params.shape[1]
    tangent_of = make_tangent(ode_fn, n_vars, n_params)

    # The unit and zero directions the sweeps are seeded with, laid out the way
    # the kernel lays them out: row ``c`` is ``e_c`` and the last row is zero.
    length = max(n_vars, n_params)
    seeds = np.zeros((n_vars + 1, length), dtype=np.float64)
    for col in range(n_vars):
        seeds[col, col] = 1.0

    @cuda.jit
    def kernel(y, t, p, seed, jacobian, time_jacobian):
        i = cuda.grid(1)
        if i < y.shape[0]:
            column = cuda.local.array(n_vars, types.float64)
            zero = n_vars  # the seed table's trailing all-zero row
            for col in range(n_vars):
                # J . e_col, one sweep for one column of df/dy.
                tangent_of(
                    column,
                    y[i],
                    t,
                    p[i],
                    seed[col, 0:n_vars],
                    0.0,
                    seed[zero, 0:n_params],
                )
                for row in range(n_vars):
                    jacobian[i, row, col] = column[row]
            # Seeding time rather than the state gives df/dt whole.
            tangent_of(
                column,
                y[i],
                t,
                p[i],
                seed[zero, 0:n_vars],
                1.0,
                seed[zero, 0:n_params],
            )
            for row in range(n_vars):
                time_jacobian[i, row] = column[row]

    d_jacobian = cuda.device_array((n, n_vars, n_vars), dtype=np.float64)
    d_time_jacobian = cuda.device_array((n, n_vars), dtype=np.float64)
    threads = 64
    kernel[(n + threads - 1) // threads, threads](
        cuda.to_device(y),
        float(t),
        cuda.to_device(params),
        cuda.to_device(seeds),
        d_jacobian,
        d_time_jacobian,
    )
    return d_jacobian.copy_to_host(), d_time_jacobian.copy_to_host()


def jax_jacobian(ode_fn, y, t, params):
    """The reference ``df/dy``, from JAX's forward-mode AD of the same ``ode_fn``."""

    def rhs(y_i, p_i):
        return jnp.stack(jnp.broadcast_arrays(*ode_fn(y_i, t, p_i)))

    return np.asarray(jax.vmap(jax.jacfwd(rhs))(jnp.asarray(y), jnp.asarray(params)))


@parametrize_system_cases
def test_enzyme_jacobian_matches_jax(case):
    y = np.asarray(case.y0, dtype=np.float64)
    params = np.asarray(case.params, dtype=np.float64)
    t = float(case.t_span[0])

    jacobian, time_jacobian = evaluate_derivatives(case.ode_fn, y, t, params)
    expected = jax_jacobian(case.ode_fn, y, t, params)

    np.testing.assert_allclose(jacobian, expected, rtol=1e-12, atol=1e-12)
    # Every reference system is autonomous, so df/dt must come back exactly 0.
    np.testing.assert_array_equal(time_jacobian, np.zeros_like(time_jacobian))


def test_enzyme_jacobian_recovers_a_time_derivative():
    """A non-autonomous right-hand side, where df/dt is the whole point."""

    def ode_fn(y, t, p):
        return (-p[0] * y[0] + p[1] * t, y[0] * t)

    y = np.array([[2.0, -1.0], [0.5, 3.0]])
    params = np.array([[10.0, 5.0], [3.0, -2.0]])
    t = 0.75

    jacobian, time_jacobian = evaluate_derivatives(ode_fn, y, t, params)

    expected = np.asarray([[[-params[i, 0], 0.0], [t, 0.0]] for i in range(y.shape[0])])
    np.testing.assert_allclose(jacobian, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        time_jacobian, np.stack([params[:, 1], y[:, 0]], axis=1), rtol=1e-14
    )
