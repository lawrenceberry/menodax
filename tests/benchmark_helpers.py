from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np
import pytest

from reference.systems.python import (
    bateman,
    brusselator,
    heat,
    kaps,
    lorenz,
    robertson,
    vdp,
)

jax.config.update("jax_enable_x64", True)

ENSEMBLE_SIZE = 2


@dataclass(frozen=True)
class SystemCase:
    name: str
    y0: np.ndarray
    t_span: np.ndarray
    params: np.ndarray
    ode_fn: Callable
    # No solver consumes this any more -- rodas5P derives its Jacobian from
    # ode_fn with Enzyme. It stays as the reference that
    # tests/test_enzyme_jacobian.py checks that derivation against.
    jac_fn: Callable
    system_config: dict
    kwargs: dict
    linear_implicit: bool
    expected_shape: tuple[int, int, int]
    assert_solution: Callable[[np.ndarray, "SystemCase"], None]


def _bateman_case():
    system = bateman.make_system(4, 1e2)
    t_span = np.asarray(bateman.TIMES[:3], dtype=np.float64)
    params = np.asarray(bateman.make_params(ENSEMBLE_SIZE), dtype=np.float64)
    y0 = np.broadcast_to(np.asarray(system["y0"]), (ENSEMBLE_SIZE, 4)).copy()

    def assert_solution(out, case):
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=3e-5)
        exact = bateman.exact_solution(
            system["M_np"], np.asarray(system["y0"]), case.t_span, case.params
        )
        np.testing.assert_allclose(out, exact, rtol=2e-2, atol=3e-5)

    return SystemCase(
        name="bateman",
        y0=y0,
        t_span=t_span,
        params=params,
        ode_fn=system["ode_fn"],
        jac_fn=system["jac_fn"],
        system_config={"n_vars": 4, "stiffness": 1e2},
        kwargs={"first_step": 1e-4, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 4),
        assert_solution=assert_solution,
    )


def _brusselator_case():
    ode, _, jac = brusselator.make_system(4)
    y0, params = brusselator.make_scenario(4, ENSEMBLE_SIZE, divergence=0.0)
    t_span = np.asarray(brusselator.TIMES[:3], dtype=np.float64)
    return SystemCase(
        name="brusselator",
        y0=np.asarray(y0, dtype=np.float64),
        t_span=t_span,
        params=np.asarray(params, dtype=np.float64),
        ode_fn=ode,
        jac_fn=jac,
        system_config={"n_grid": 4},
        kwargs={"first_step": 1e-3, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 8),
        assert_solution=lambda out, case: None,
    )


def _heat_case():
    system = heat.make_system(4)
    t_span = np.asarray(heat.TIMES[:3], dtype=np.float64)
    params = np.asarray(heat.make_params(ENSEMBLE_SIZE), dtype=np.float64)
    y0 = np.broadcast_to(np.asarray(system["y0"]), (ENSEMBLE_SIZE, 4)).copy()

    def assert_solution(out, case):
        exact = heat.exact_solution(4, case.t_span, case.params)
        np.testing.assert_allclose(out, exact, rtol=3e-2, atol=3e-5)

    return SystemCase(
        name="heat",
        y0=y0,
        t_span=t_span,
        params=params,
        ode_fn=system["ode_fn"],
        jac_fn=system["jac_fn"],
        system_config={"n_vars": 4},
        kwargs={"first_step": 1e-4, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 4),
        assert_solution=assert_solution,
    )


def _kaps_case():
    system = kaps.make_system(2, 1e-2)
    t_span = np.asarray(kaps.TIMES[:3], dtype=np.float64)
    params = np.asarray(kaps.make_params(ENSEMBLE_SIZE), dtype=np.float64)
    y0 = np.broadcast_to(np.asarray(system["y0"]), (ENSEMBLE_SIZE, 4)).copy()

    def assert_solution(out, case):
        exact = kaps.exact_solution(case.t_span, case.params, 2)
        np.testing.assert_allclose(out, exact, rtol=3e-2, atol=3e-5)

    return SystemCase(
        name="kaps",
        y0=y0,
        t_span=t_span,
        params=params,
        ode_fn=system["ode_fn"],
        jac_fn=system["jac_fn"],
        system_config={"n_pairs": 2, "epsilon_min": 1e-2},
        kwargs={"first_step": 1e-3, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=False,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 4),
        assert_solution=assert_solution,
    )


def _lorenz_case():
    y0, params = lorenz.make_scenario(ENSEMBLE_SIZE, divergence=0.0)
    t_span = np.asarray((0.0, 0.01, 0.02), dtype=np.float64)
    return SystemCase(
        name="lorenz",
        y0=np.asarray(y0, dtype=np.float64),
        t_span=t_span,
        params=np.asarray(params, dtype=np.float64),
        ode_fn=lorenz.ode_fn,
        jac_fn=lorenz.jac_fn,
        system_config={},
        kwargs={"first_step": 1e-4, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 3),
        assert_solution=lambda out, case: None,
    )


def _robertson_case():
    y0, params = robertson.make_scenario(ENSEMBLE_SIZE, divergence=0.0)
    t_span = np.asarray((0.0, 1e-6, 1e-4), dtype=np.float64)
    return SystemCase(
        name="robertson",
        y0=np.asarray(y0, dtype=np.float64),
        t_span=t_span,
        params=np.asarray(params, dtype=np.float64),
        ode_fn=robertson.ode_fn,
        jac_fn=robertson.jac_fn,
        system_config={},
        kwargs={"first_step": 1e-8, "rtol": 1e-5, "atol": 1e-7, "max_steps": 10000},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 3),
        assert_solution=lambda out, case: np.testing.assert_allclose(
            out.sum(axis=-1), 1.0, atol=3e-5
        ),
    )


def _vdp_case():
    ode, _, jac = vdp.make_system(2, mu=1.0)
    y0, params = vdp.make_scenario(2, ENSEMBLE_SIZE, divergence=0.0)
    t_span = np.asarray(vdp.TIMES[:3], dtype=np.float64)
    return SystemCase(
        name="vdp",
        y0=np.asarray(y0, dtype=np.float64),
        t_span=t_span,
        params=np.asarray(params, dtype=np.float64),
        ode_fn=ode,
        jac_fn=jac,
        system_config={"n_osc": 2},
        kwargs={"first_step": 1e-3, "rtol": 1e-5, "atol": 1e-7},
        linear_implicit=True,
        expected_shape=(ENSEMBLE_SIZE, len(t_span), 4),
        assert_solution=lambda out, case: None,
    )


SYSTEM_CASES = (
    _bateman_case(),
    _brusselator_case(),
    _heat_case(),
    _kaps_case(),
    _lorenz_case(),
    _robertson_case(),
    _vdp_case(),
)
SYSTEM_IDS = tuple(case.name for case in SYSTEM_CASES)


def parametrize_system_cases(fn):
    return pytest.mark.parametrize("case", SYSTEM_CASES, ids=SYSTEM_IDS)(fn)


def block_until_ready(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return value


def benchmark_solve(benchmark, solve_once):
    return benchmark.pedantic(
        lambda: block_until_ready(solve_once()),
        warmup_rounds=1,
        rounds=1,
    )


def assert_case_output(output, case: SystemCase):
    out = np.asarray(output)
    assert out.shape == case.expected_shape
    assert np.all(np.isfinite(out))
    case.assert_solution(out, case)
