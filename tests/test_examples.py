"""Checks on the CUDA-device callbacks the examples hand to the numba solvers.

Each example carries two forms of the same equations: a ``jnp`` one traced by
the Diffrax/scipy reference backends, and a ``math``/tuple one that
``numba_cuda_mlir`` compiles for the modax kernel solver.  Nothing forces the
two to agree, so these tests pin it down:

* every device callback compiles to PTX -- ``cuda.compile_ptx`` runs the full
  numba typing and lowering pipeline without needing a GPU, so this catches the
  common failures (calling a plain Python helper, returning an array instead of
  a tuple) on any machine;
* the device RHS matches the ``jnp`` RHS;
* for the implicit examples, the Jacobian and time derivative Enzyme takes off
  the device RHS match ``jax.jacobian`` of the ``jnp`` one.  That needs a GPU,
  unlike the rest of this module, so those two tests skip without one;
* the Mukhanov-Sasaki background tables, which numba lowers into CUDA constant
  memory, stay inside the 64 KiB budget.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
types = pytest.importorskip("numba_cuda_mlir.types")

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# ``(y_row, t, p_row)`` -- the signature the solvers call these callbacks with.
_DEVICE_SIG = (types.float64[:], types.float64, types.float64[:])

# CUDA constant memory is 64 KiB per module.
_CONST_LIMIT_BYTES = 64 * 1024


def _load_example(name: str):
    """Import an ``examples/<name>/main.py`` that is not on the import path."""
    path = _EXAMPLES / name / "main.py"
    spec = importlib.util.spec_from_file_location(f"_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compile_device(fn):
    """Compile a flat tuple-returning device callback -> ``(ptx, const_bytes)``."""
    ptx, _ = cuda.compile_ptx(fn, _DEVICE_SIG, device=True, cc=(8, 0))
    const_bytes = sum(int(n) for n in re.findall(r"\.const .*?\[(\d+)\]", ptx))
    return ptx, const_bytes


requires_gpu = pytest.mark.skipif(
    not cuda.is_available(), reason="Enzyme derivatives are device callables"
)


def _max_rel_error(actual, desired):
    """Largest per-component relative error."""
    actual = np.asarray(actual, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    return float(np.max(np.abs(actual - desired) / (np.abs(desired) + 1e-300)))


def _max_scaled_error(actual, desired):
    """Largest error relative to the magnitude of the whole vector.

    The right measure when one component of a derivative vector can pass
    through zero while the others stay O(1): a per-component ratio then blows
    up on a difference that is negligible to the solver, whose error control
    also works off the state vector as a whole.
    """
    actual = np.asarray(actual, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    return float(np.max(np.abs(actual - desired)) / (np.max(np.abs(desired)) + 1e-300))


# ---------------------------------------------------------------------------
# BBN: rodas5P, four-species stiff network
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bbn():
    return _load_example("bbn_estimation")


def _bbn_samples(n=120):
    rng = np.random.default_rng(0)
    out = []
    while len(out) < n:
        x = float(np.exp(rng.uniform(np.log(0.14), np.log(129.0))))
        # g_star steps at T = m_e, where df/dx genuinely does not exist.
        if abs(x - 1.293 / 0.511) < 0.05:
            continue
        y = np.array(
            [
                rng.uniform(0.1, 0.6),
                rng.uniform(0.4, 0.9),
                10 ** rng.uniform(-12, -4),
                10 ** rng.uniform(-12, -2),
            ]
        )
        p = np.array([rng.uniform(0.5, 1.0), rng.uniform(2.0, 4.0)])
        out.append((y, x, p))
    return out


def test_bbn_device_callbacks_compile(bbn):
    _compile_device(bbn.bbn_ode_device)


def test_bbn_device_rhs_matches_jnp(bbn):
    for y, x, p in _bbn_samples():
        ref = np.asarray(bbn.bbn_ode(jnp.asarray(y), x, jnp.asarray(p)))
        assert _max_rel_error(bbn.bbn_ode_device(y, x, p), ref) < 1e-12


@requires_gpu
def test_bbn_enzyme_derivatives_match_autodiff(bbn):
    from tests.test_enzyme_jacobian import evaluate_derivatives

    for y, x, p in _bbn_samples(n=12):
        jacobian, time_jacobian = evaluate_derivatives(
            bbn.bbn_ode_device, y[None, :], x, p[None, :]
        )
        ref_jac = jax.jacobian(lambda yy, x=x, p=p: bbn.bbn_ode(yy, x, jnp.asarray(p)))(
            jnp.asarray(y)
        )
        ref_dt = jax.jacobian(
            lambda xx, y=y, p=p: bbn.bbn_ode(jnp.asarray(y), xx, jnp.asarray(p))
        )(x)
        assert _max_rel_error(jacobian[0], ref_jac) < 1e-10
        assert _max_rel_error(time_jacobian[0], ref_dt) < 1e-10


# ---------------------------------------------------------------------------
# 21-cm IGM: rodas5P, three-component thermal/ionization history
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def igm():
    return _load_example("21cm_igm_evolution")


def _igm_samples(igm, n=120):
    rng = np.random.default_rng(0)
    u_max = float(igm.u_from_redshift(igm.Z_FINAL))
    out = []
    while len(out) < n:
        u = rng.uniform(0.0, u_max)
        log_Tk = rng.uniform(np.log(0.6), np.log(4.0e4))
        # Stay off the clip and maximum kinks, where the derivative jumps.
        if abs(np.exp(log_Tk) - 10.0) < 0.5:
            continue
        y = np.array([log_Tk, rng.uniform(-25, -1), rng.uniform(-25, -1)])
        x_e = 1.0 / (1.0 + np.exp(-y[1]))
        q = 1.0 / (1.0 + np.exp(-y[2]))
        if (
            min(abs(x_e * (1 - x_e) - igm.LOGIT_EPS), abs(q * (1 - q) - igm.LOGIT_EPS))
            < 1e-9
        ):
            continue
        p = np.array([rng.uniform(-3, -0.5), rng.uniform(-2, 2), rng.uniform(3.5, 5.5)])
        out.append((y, u, p))
    return out


def test_igm_device_callbacks_compile(igm):
    _compile_device(igm.igm_ode_device)


def test_igm_device_rhs_matches_jnp(igm):
    for y, u, p in _igm_samples(igm):
        ref = np.asarray(igm.igm_ode(jnp.asarray(y), u, jnp.asarray(p)))
        assert _max_rel_error(igm.igm_ode_device(y, u, p), ref) < 1e-12


@requires_gpu
def test_igm_enzyme_derivatives_match_autodiff(igm):
    from tests.test_enzyme_jacobian import evaluate_derivatives

    for y, u, p in _igm_samples(igm, n=12):
        jacobian, time_jacobian = evaluate_derivatives(
            igm.igm_ode_device, y[None, :], u, p[None, :]
        )
        ref_jac = jax.jacobian(lambda yy, u=u, p=p: igm.igm_ode(yy, u, jnp.asarray(p)))(
            jnp.asarray(y)
        )
        ref_dt = jax.jacobian(
            lambda uu, y=y, p=p: igm.igm_ode(jnp.asarray(y), uu, jnp.asarray(p))
        )(u)
        # The dTdt/dT_k entry is a near-exact cancellation of two terms that
        # agree to ~9 digits, so its relative error floor is well above 1e-12
        # while the absolute error stays at the double-precision limit.
        assert _max_rel_error(jacobian[0], ref_jac) < 1e-5
        assert _max_rel_error(time_jacobian[0], ref_dt) < 1e-6


# ---------------------------------------------------------------------------
# Mukhanov-Sasaki: tsit5, interpolated background tables
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mukhanov():
    return _load_example("mukhanov_sasaki")


@pytest.fixture(scope="module")
def mukhanov_tables(mukhanov):
    """Background tables built on the CPU.

    ``solve_background`` now runs on the kernel solver and so needs a GPU;
    integrating the same ``background_ode`` with scipy keeps this test
    hardware-independent, and the tables only have to be representative.
    """
    from scipy.integrate import solve_ivp

    times = np.linspace(0.0, mukhanov.N_BACKGROUND_MAX, mukhanov.N_BACKGROUND_SAMPLES)
    params = np.array([mukhanov.MASS])
    sol = solve_ivp(
        lambda t, y: np.asarray(
            mukhanov.background_ode(jnp.asarray(y), t, jnp.asarray(params))
        ),
        (times[0], times[-1]),
        np.array([mukhanov.PHI_INITIAL, mukhanov.D_PHI_DN_INITIAL]),
        t_eval=times,
        rtol=1e-11,
        atol=1e-13,
        method="DOP853",
    )
    # Inflation ends before the grid does, so the solve stops early.
    return mukhanov.build_background_tables(times[: sol.y.shape[1]], sol.y.T)


def test_mukhanov_background_device_rhs_matches_jnp(mukhanov):
    rng = np.random.default_rng(0)
    params = np.array([mukhanov.MASS])
    _compile_device(mukhanov.background_ode_device)
    for _ in range(200):
        y = np.array([rng.uniform(1.0, 18.0), rng.uniform(-1.0, 0.0)])
        ref = np.asarray(
            mukhanov.background_ode(jnp.asarray(y), 0.0, jnp.asarray(params))
        )
        got = mukhanov.background_ode_device(y, 0.0, params)
        assert _max_rel_error(got, ref) < 1e-12


def test_mukhanov_mode_rhs_fits_in_constant_memory(mukhanov, mukhanov_tables):
    """The closed-over background tables must fit CUDA constant memory.

    Indexing a closed-over array emits one constant copy per reference site, so
    the packed single-array layout in ``make_mode_ode_device`` is load-bearing:
    three separate tables read twice each came to ~135 KiB and would not load.
    """
    _, const_bytes = _compile_device(mukhanov.make_mode_ode_device(mukhanov_tables))
    assert 0 < const_bytes < _CONST_LIMIT_BYTES


def test_mukhanov_mode_device_rhs_matches_jnp(mukhanov, mukhanov_tables):
    jax_ode = mukhanov.make_mode_ode(mukhanov_tables)
    device_ode = mukhanov.make_mode_ode_device(mukhanov_tables)
    _, _, y0, params = mukhanov.prepare_mode_problem(mukhanov_tables)

    rng = np.random.default_rng(0)
    for i in range(y0.shape[0]):
        for s in np.linspace(0.0, 1.0, 8):
            y = y0[i] * rng.uniform(0.5, 2.0, size=4)
            ref = np.asarray(jax_ode(jnp.asarray(y), s, jnp.asarray(params[i])))
            got = device_ode(y, s, params[i])
            # The arithmetic index differs from jnp.interp's searchsorted by a
            # rounding step when n_now lands next to a knot, so compare against
            # the vector scale rather than per component.
            assert _max_scaled_error(got, ref) < 1e-12


def test_mukhanov_mode_device_rhs_clamps_outside_the_table(mukhanov, mukhanov_tables):
    """Outside the table both forms clamp to the end values, as np.interp does."""
    jax_ode = mukhanov.make_mode_ode(mukhanov_tables)
    device_ode = mukhanov.make_mode_ode_device(mukhanov_tables)
    _, _, y0, params = mukhanov.prepare_mode_problem(mukhanov_tables)

    y = y0[0]
    for s in (-0.5, 1.5):
        ref = np.asarray(jax_ode(jnp.asarray(y), s, jnp.asarray(params[0])))
        got = device_ode(y, s, params[0])
        assert _max_scaled_error(got, ref) < 1e-12
