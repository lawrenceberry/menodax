import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


def _have_cuda() -> bool:
    try:
        from numba_cuda_mlir import cuda
    except ImportError:
        return False
    return bool(cuda.is_available())


def _build_numba_callbacks():
    from numba_cuda_mlir import cuda

    @cuda.jit(device=True)
    def decay_device(y, t, p):
        return (-p[0] * y[0],)

    @cuda.jit(device=True)
    def decay_jac_device(y, t, p):
        return ((-p[0],),)

    return decay_device, decay_jac_device


def _plain_numba_decay(y, t, p):
    return (-p[0] * y[0],)


def _plain_numba_decay_jac(y, t, p):
    return ((-p[0],),)


def _solver_cases():
    if not _have_cuda():
        return []

    from solvers.rodas5P import solve as rodas5Pnumba_solve
    from solvers.tsit5 import solve as tsit5numba_solve

    decay, decay_jac = _build_numba_callbacks()
    return [
        ("tsit5", tsit5numba_solve, (decay,), {}),
        ("rodas5P", rodas5Pnumba_solve, (decay, decay_jac), {}),
    ]


_SOLVER_CASES = _solver_cases()


@pytest.mark.skipif(not _have_cuda(), reason="numba_cuda_mlir unavailable")
@pytest.mark.parametrize(
    ("name", "solve_fn", "args", "kwargs"),
    _SOLVER_CASES,
)
def test_solver_vmap_over_params_matches_native_ensemble(name, solve_fn, args, kwargs):
    del name
    y0 = jnp.array([1.0])
    t_span = jnp.array([0.0, 0.5, 1.0])
    params = jnp.array([[0.5], [1.0], [2.0], [4.0]])
    solve_kwargs = {
        "rtol": 1e-5,
        "atol": 1e-7,
        "first_step": 0.1,
        "max_steps": 256,
        **kwargs,
    }

    direct = solve_fn(*args, y0, t_span, params, **solve_kwargs)
    vmapped = jax.vmap(lambda p: solve_fn(*args, y0, t_span, p, **solve_kwargs))(params)

    np.testing.assert_allclose(vmapped[:, 0], direct, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not _have_cuda(), reason="numba_cuda_mlir unavailable")
@pytest.mark.parametrize(
    ("name", "solve_fn", "args", "kwargs"),
    _SOLVER_CASES,
)
def test_solver_vmap_over_y0_and_params_matches_native_ensemble(
    name, solve_fn, args, kwargs
):
    del name
    y0s = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    t_span = jnp.array([0.0, 0.5, 1.0])
    params = jnp.array([[0.5], [1.0], [2.0], [4.0]])
    solve_kwargs = {
        "rtol": 1e-5,
        "atol": 1e-7,
        "first_step": 0.1,
        "max_steps": 256,
        **kwargs,
    }

    direct = solve_fn(*args, y0s, t_span, params, **solve_kwargs)
    vmapped = jax.vmap(lambda y0, p: solve_fn(*args, y0, t_span, p, **solve_kwargs))(
        y0s, params
    )

    np.testing.assert_allclose(vmapped[:, 0], direct, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not _have_cuda(), reason="numba_cuda_mlir unavailable")
def test_solvers_auto_jit_plain_python_callbacks():
    from solvers.rodas5P import solve as rodas5Pnumba_solve
    from solvers.tsit5 import solve as tsit5numba_solve

    y0 = np.array([1.0], dtype=np.float64)
    t_span = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    params = np.array([[0.5], [1.0], [2.0]], dtype=np.float64)
    solve_kwargs = {
        "rtol": 1e-5,
        "atol": 1e-7,
        "first_step": 0.1,
        "max_steps": 256,
    }
    expected = y0[0] * np.exp(-params[:, 0, None] * t_span[None, :])

    tsit5_sol = tsit5numba_solve(_plain_numba_decay, y0, t_span, params, **solve_kwargs)
    rodas5P_sol = rodas5Pnumba_solve(
        _plain_numba_decay,
        _plain_numba_decay_jac,
        y0,
        t_span,
        params,
        **solve_kwargs,
    )
    np.testing.assert_allclose(tsit5_sol[:, :, 0], expected, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(rodas5P_sol[:, :, 0], expected, rtol=2e-5, atol=2e-7)


@pytest.mark.skipif(not _have_cuda(), reason="numba_cuda_mlir unavailable")
def test_solver_vmap_return_stats_shapes():
    from solvers.rodas5P import solve as rodas5Pnumba_solve

    decay, decay_jac = _build_numba_callbacks()

    y0 = jnp.array([1.0])
    t_span = jnp.array([0.0, 0.5, 1.0])
    params = jnp.array([[0.5], [1.0], [2.0], [4.0]])

    _, stats = jax.vmap(
        lambda p: rodas5Pnumba_solve(
            decay,
            decay_jac,
            y0,
            t_span,
            p,
            rtol=1e-5,
            atol=1e-7,
            first_step=0.1,
            max_steps=256,
            return_stats=True,
        )
    )(params)

    assert stats["accepted_steps"].shape == (params.shape[0], 1)
    assert stats["rejected_steps"].shape == (params.shape[0], 1)
    assert stats["loop_steps"].shape == (params.shape[0], 1)
    assert stats["batch_loop_iterations"].shape == (params.shape[0], 1)
    assert stats["valid_lanes"].shape == (params.shape[0], 1)
    assert bool(jnp.all(stats["valid_lanes"] == 1))
