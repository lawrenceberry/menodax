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
@pytest.mark.parametrize(
    "solve_fn",
    (diffrax_tsit5_solve, diffrax_kvaerno5_solve),
    ids=("tsit5", "kvaerno5"),
)
def test_diffrax_reference_system(benchmark, case, solve_fn):
    result = benchmark_solve(
        benchmark,
        lambda: solve_fn(
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
@pytest.mark.parametrize(
    "solve_fn",
    (julia_tsit5_solve, julia_rodas5P_solve),
    ids=("tsit5", "rodas5P"),
)
def test_julia_reference_system(benchmark, case, ensemble_backend, solve_fn):
    if (
        ensemble_backend == "EnsembleGPUKernel"
        and case.name in _GPU_KERNEL_UNSUPPORTED_SYSTEMS
    ):
        pytest.skip(f"{case.name} is not GPUKernel-compatible in the Julia runner")
    result = benchmark_julia_solver(
        benchmark,
        solve_fn,
        case.name,
        y0=case.y0,
        t_span=case.t_span,
        params=case.params,
        system_config=case.system_config,
        ensemble_backend=ensemble_backend,
        **case.kwargs,
    )
    assert_case_output(np.asarray(result), case)
