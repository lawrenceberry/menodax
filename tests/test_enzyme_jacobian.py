"""The Enzyme-derived Jacobian must match the analytic one, system by system.

``rodas5P`` no longer takes a ``jac_fn``: it differentiates the right-hand side
with Enzyme instead (see :mod:`solvers._enzyme_jacobian`). The reference systems
still carry hand-written Jacobians, which makes them the natural check on that
derivation — and on the ``df/dt`` the solver takes from the same sweeps, which
no reference system supplies but which is zero for all of them, every one being
autonomous.

"""

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

    from solvers._enzyme_jacobian import make_jacobian_column

    y = np.ascontiguousarray(y, dtype=np.float64)
    params = np.ascontiguousarray(params, dtype=np.float64)
    n, n_vars = y.shape
    n_params = params.shape[1]
    jacobian_column = make_jacobian_column(ode_fn, n_vars, n_params)

    @cuda.jit
    def kernel(y, t, p, jacobian, time_jacobian):
        i = cuda.grid(1)
        if i < y.shape[0]:
            column = cuda.local.array(n_vars, types.float64)
            for col in range(n_vars + 1):
                jacobian_column(column, y[i], t, p[i], col)
                if col == n_vars:
                    for row in range(n_vars):
                        time_jacobian[i, row] = column[row]
                else:
                    for row in range(n_vars):
                        jacobian[i, row, col] = column[row]

    d_jacobian = cuda.device_array((n, n_vars, n_vars), dtype=np.float64)
    d_time_jacobian = cuda.device_array((n, n_vars), dtype=np.float64)
    threads = 64
    kernel[(n + threads - 1) // threads, threads](
        cuda.to_device(y),
        float(t),
        cuda.to_device(params),
        d_jacobian,
        d_time_jacobian,
    )
    return d_jacobian.copy_to_host(), d_time_jacobian.copy_to_host()


@parametrize_system_cases
def test_enzyme_jacobian_matches_analytic(case):
    y = np.asarray(case.y0, dtype=np.float64)
    params = np.asarray(case.params, dtype=np.float64)
    t = float(case.t_span[0])

    jacobian, time_jacobian = evaluate_derivatives(case.ode_fn, y, t, params)
    expected = np.asarray(
        [np.asarray(case.jac_fn(y[i], t, params[i])) for i in range(y.shape[0])]
    )

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
