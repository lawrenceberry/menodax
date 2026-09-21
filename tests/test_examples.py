"""Checks on the CUDA-device callbacks the examples hand to the numba solvers.

Each example needs its equations in two forms: a ``jnp`` one traced by the
Diffrax and scipy reference backends, and a ``math``/tuple one that
``numba_cuda_mlir`` compiles for the modax kernel solver.  Both are built from
one body through ``examples/dual_backend.py``, which leaves the backend shim
-- which ``math`` name stands in for which ``jnp`` one, and the branchless
selects -- as the part that can still be wrong.  These tests pin that down:

* every device callback compiles to PTX -- ``cuda.compile_ptx`` runs the full
  numba typing and lowering pipeline without needing a GPU, so this catches the
  common failures (calling a plain Python helper, returning an array instead of
  a tuple) on any machine;
* the device arithmetic matches the ``jnp`` form.  The device callback itself
  calls CUDA device functions and so only runs on a GPU, hence the ``.host``
  member: the same body over the same ``math`` ops, with those helpers left as
  plain Python;
* for the implicit examples, the Jacobian and time derivative Enzyme takes off
  the device RHS match ``jax.jacobian`` of the ``jnp`` one.  That needs a GPU,
  unlike the rest of this module, so that test skips without one;
* the ``jnp`` form pickles and comes back computing the same values, since the
  scipy reference backend sends it to worker processes;
* the Mukhanov-Sasaki background tables, which numba lowers into CUDA constant
  memory, stay inside the 64 KiB budget.
"""

from __future__ import annotations

import functools
import importlib.util
import pickle
import re
import sys
from pathlib import Path
from typing import Callable, NamedTuple

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


@functools.cache
def _load_example(name: str):
    """Import an ``examples/<name>/main.py`` that is not on the import path.

    Registered in ``sys.modules`` so that pickle, which carries the examples'
    ``_make_*`` factories by module and name, can find them again.
    """
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


def _pickle_round_trip(rhs, samples):
    """``rhs.jax`` must survive pickling and still agree with the original."""
    rebuilt = pickle.loads(pickle.dumps(rhs.jax))
    for y, t, p in samples:
        expected = np.asarray(rhs.jax(jnp.asarray(y), t, jnp.asarray(p)))
        got = np.asarray(rebuilt(jnp.asarray(y), t, jnp.asarray(p)))
        np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# The implicit examples: rodas5P over a stiff network, Enzyme-derived Jacobian
# ---------------------------------------------------------------------------


def _bbn_samples(bbn, n=120):
    """BBN: four-species stiff network, ``(y, x, params)`` off the g_star step."""
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


def _igm_samples(igm, n=120):
    """21-cm IGM: three-component thermal/ionization history, ``(y, u, params)``."""
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


class _Implicit(NamedTuple):
    """One implicit example: where its RHS lives, sample points, tolerances."""

    example: str
    rhs: str  # the module attribute holding the ``Forms``
    samples: Callable  # ``(module, n) -> [(y, t, p), ...]``
    jacobian_tol: float
    time_derivative_tol: float


_IMPLICIT = [
    _Implicit("bbn_estimation", "BBN_ODE", _bbn_samples, 1e-10, 1e-10),
    # The dTdt/dT_k entry is a near-exact cancellation of two terms that
    # agree to ~9 digits, so its relative error floor is well above 1e-12
    # while the absolute error stays at the double-precision limit.
    _Implicit("21cm_igm_evolution", "IGM_ODE", _igm_samples, 1e-5, 1e-6),
]
implicit_case = pytest.mark.parametrize("case", _IMPLICIT, ids=lambda c: c.example)


def _implicit(case):
    module = _load_example(case.example)
    return module, getattr(module, case.rhs)


@implicit_case
def test_device_callbacks_compile(case):
    _, rhs = _implicit(case)
    _compile_device(rhs.device)


@implicit_case
def test_device_rhs_matches_jnp(case):
    module, rhs = _implicit(case)
    for y, t, p in case.samples(module):
        ref = np.asarray(rhs.jax(jnp.asarray(y), t, jnp.asarray(p)))
        assert _max_rel_error(rhs.host(y, t, p), ref) < 1e-12


@implicit_case
def test_traced_rhs_pickles(case):
    module, rhs = _implicit(case)
    _pickle_round_trip(rhs, case.samples(module, n=4))


@requires_gpu
@implicit_case
def test_enzyme_derivatives_match_autodiff(case):
    from tests.test_enzyme_jacobian import evaluate_derivatives

    module, rhs = _implicit(case)
    for y, t, p in case.samples(module, n=12):
        jacobian, time_jacobian = evaluate_derivatives(
            rhs.device, y[None, :], t, p[None, :]
        )
        ref_jac = jax.jacobian(lambda yy, t=t, p=p: rhs.jax(yy, t, jnp.asarray(p)))(
            jnp.asarray(y)
        )
        ref_dt = jax.jacobian(
            lambda tt, y=y, p=p: rhs.jax(jnp.asarray(y), tt, jnp.asarray(p))
        )(t)
        assert _max_rel_error(jacobian[0], ref_jac) < case.jacobian_tol
        assert _max_rel_error(time_jacobian[0], ref_dt) < case.time_derivative_tol


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
            mukhanov.BACKGROUND_ODE.jax(jnp.asarray(y), t, jnp.asarray(params))
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
    _compile_device(mukhanov.BACKGROUND_ODE.device)
    for _ in range(200):
        y = np.array([rng.uniform(1.0, 18.0), rng.uniform(-1.0, 0.0)])
        ref = np.asarray(
            mukhanov.BACKGROUND_ODE.jax(jnp.asarray(y), 0.0, jnp.asarray(params))
        )
        got = mukhanov.BACKGROUND_ODE.host(y, 0.0, params)
        assert _max_rel_error(got, ref) < 1e-12


def test_mukhanov_mode_rhs_fits_in_constant_memory(mukhanov, mukhanov_tables):
    """The closed-over background tables must fit CUDA constant memory.

    Indexing a closed-over array emits one constant copy per reference site, so
    the packed single-array layout in ``make_mode_ode`` is load-bearing:
    three separate tables read twice each came to ~135 KiB and would not load.
    """
    _, const_bytes = _compile_device(mukhanov.make_mode_ode(mukhanov_tables).device)
    assert 0 < const_bytes < _CONST_LIMIT_BYTES


def test_mukhanov_mode_device_rhs_matches_jnp(mukhanov, mukhanov_tables):
    mode_ode = mukhanov.make_mode_ode(mukhanov_tables)
    _, _, y0, params = mukhanov.prepare_mode_problem(mukhanov_tables)

    rng = np.random.default_rng(0)
    for i in range(y0.shape[0]):
        for s in np.linspace(0.0, 1.0, 8):
            y = y0[i] * rng.uniform(0.5, 2.0, size=4)
            ref = np.asarray(mode_ode.jax(jnp.asarray(y), s, jnp.asarray(params[i])))
            got = mode_ode.host(y, s, params[i])
            # One bracketing index, one table: the two forms differ only in
            # how their maths functions round, so compare against the vector
            # scale rather than per component.
            assert _max_scaled_error(got, ref) < 1e-12


def test_mukhanov_mode_rhs_pickles(mukhanov, mukhanov_tables):
    """The table kwarg travels with the recipe, so a worker can rebuild it."""
    mode_ode = mukhanov.make_mode_ode(mukhanov_tables)
    _, _, y0, params = mukhanov.prepare_mode_problem(mukhanov_tables)
    _pickle_round_trip(
        mode_ode, [(y0[i], s, params[i]) for i, s in ((0, 0.3), (5, 0.9))]
    )


def test_mukhanov_mode_device_rhs_clamps_outside_the_table(mukhanov, mukhanov_tables):
    """Outside the table both forms clamp to the end values, as np.interp does."""
    mode_ode = mukhanov.make_mode_ode(mukhanov_tables)
    _, _, y0, params = mukhanov.prepare_mode_problem(mukhanov_tables)

    y = y0[0]
    for s in (-0.5, 1.5):
        ref = np.asarray(mode_ode.jax(jnp.asarray(y), s, jnp.asarray(params[0])))
        got = mode_ode.host(y, s, params[0])
        assert _max_scaled_error(got, ref) < 1e-12
