import jax.numpy as jnp
import numpy as np
import pytest

from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.diffrax_tsit5 import solve as diffrax_tsit5_solve
from reference.solvers.python.julia_common import (
    JULIA_ENSEMBLE_BACKENDS,
    benchmark_julia_solver,
    julia_backend_id,
)
from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from tests.benchmark_helpers import (
    assert_case_output,
    benchmark_solve,
    parametrize_system_cases,
)

_GPU_KERNEL_UNSUPPORTED_SYSTEMS = {"bateman", "heat", "kaps"}


@parametrize_system_cases
def test_diffrax_tsit5_reference_system(benchmark, case):
    result = benchmark_solve(
        benchmark,
        lambda: diffrax_tsit5_solve(
            case.ode_fn,
            jnp.asarray(case.y0, dtype=jnp.float64),
            jnp.asarray(case.t_span, dtype=jnp.float64),
            jnp.asarray(case.params, dtype=jnp.float64),
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)


@parametrize_system_cases
def test_diffrax_kvaerno5_reference_system(benchmark, case):
    result = benchmark_solve(
        benchmark,
        lambda: diffrax_kvaerno5_solve(
            case.ode_fn,
            jnp.asarray(case.y0, dtype=jnp.float64),
            jnp.asarray(case.t_span, dtype=jnp.float64),
            jnp.asarray(case.params, dtype=jnp.float64),
            **case.kwargs,
        ),
    )
    assert_case_output(result, case)


@parametrize_system_cases
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_tsit5_reference_system(benchmark, case, ensemble_backend):
    if (
        ensemble_backend == "EnsembleGPUKernel"
        and case.name in _GPU_KERNEL_UNSUPPORTED_SYSTEMS
    ):
        pytest.skip(f"{case.name} is not GPUKernel-compatible in the Julia runner")
    result = benchmark_julia_solver(
        benchmark,
        julia_tsit5_solve,
        case.name,
        y0=case.y0,
        t_span=case.t_span,
        params=case.params,
        system_config=case.system_config,
        ensemble_backend=ensemble_backend,
        **case.kwargs,
    )
    assert_case_output(np.asarray(result), case)


@parametrize_system_cases
@pytest.mark.parametrize(
    "ensemble_backend", JULIA_ENSEMBLE_BACKENDS, ids=julia_backend_id
)
def test_julia_rodas5P_reference_system(benchmark, case, ensemble_backend):
    if (
        ensemble_backend == "EnsembleGPUKernel"
        and case.name in _GPU_KERNEL_UNSUPPORTED_SYSTEMS
    ):
        pytest.skip(f"{case.name} is not GPUKernel-compatible in the Julia runner")
    result = benchmark_julia_solver(
        benchmark,
        julia_rodas5P_solve,
        case.name,
        y0=case.y0,
        t_span=case.t_span,
        params=case.params,
        system_config=case.system_config,
        ensemble_backend=ensemble_backend,
        **case.kwargs,
    )
    assert_case_output(np.asarray(result), case)
