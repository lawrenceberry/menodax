"""Solver scaling benchmark on the Lorenz system.

Sweeps the size of an ensemble of identical trajectories from 3 to 100k on a
log scale and records solve time for modax Tsit5, Diffrax Tsit5, torchdiffeq
Dopri8 (one adaptive step shared by the whole ensemble) and Julia Tsit5 with
both DiffEqGPU ensemble backends. A point that does not compile and
solve within the case timeout is recorded as such and omitted from the plot.
Outputs a CSV and a log-log plot named after the GPU.

Usage:
    uv run python benchmarks/nonstiff_lorenz_scaling/main.py
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Sets JAX's GPU memory policy before the first import that touches the card:
# the `reference.systems.python` modules build device arrays at import time,
# which initialises the backend and fixes that policy for the whole process.
import benchmarks.benchmark_common  # noqa: E402,F401
from benchmarks._sweep import (
    ENSEMBLE_SIZE,
    IDENTICAL_DIVERGENCE,
    Problem,
    SweepBenchmark,
    SweepCase,
    main,
)
from modax.tsit5 import solve as tsit5_solve
from reference.solvers.python.diffrax_tsit5 import solve as diffrax_tsit5_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from reference.solvers.python.torchdiffeq_dopri8 import (
    solve as torchdiffeq_dopri8_solve,
)
from reference.systems.python import lorenz

jax.config.update("jax_enable_x64", True)

_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
_LOCAL_SOLVER_KWARGS = {**_SOLVER_KWARGS, "pcoeff": 0.0, "icoeff": 1.0, "dcoeff": 0.0}


def _problem(size: int) -> Problem:
    return Problem(
        lorenz.ode_fn, *lorenz.make_scenario(size, divergence=IDENTICAL_DIVERGENCE)
    )


BENCHMARK = SweepBenchmark(
    script_dir=Path(__file__).resolve().parent,
    title="Tsit5 scaling — Lorenz",
    axis=ENSEMBLE_SIZE,
    values=(3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000),
    t_span=lorenz.TIMES,
    make_problem=_problem,
    julia_solve=julia_tsit5_solve,
    julia_system="lorenz",
    cases=(
        SweepCase(
            key="modax tsit5",
            color="#f0a202",
            marker="P",
            solve_fn=tsit5_solve,
            kwargs=_LOCAL_SOLVER_KWARGS,
        ),
        SweepCase(
            key="diffrax tsit5",
            color="#2ba84a",
            marker="s",
            solve_fn=diffrax_tsit5_solve,
            kwargs=_LOCAL_SOLVER_KWARGS,
        ),
        SweepCase(
            key="torchdiffeq dopri8",
            color="#d62728",
            marker="D",
            solve_fn=torchdiffeq_dopri8_solve,
            kwargs=_SOLVER_KWARGS,
            jit=False,
        ),
        SweepCase(
            key="julia tsit5 EnsembleGPUArray",
            color="#9b59b6",
            marker="^",
            mode="julia",
            ensemble_backend="EnsembleGPUArray",
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="julia tsit5 EnsembleGPUKernel",
            color="#9b59b6",
            marker="v",
            linestyle="--",
            mode="julia",
            ensemble_backend="EnsembleGPUKernel",
            kwargs=_SOLVER_KWARGS,
        ),
    ),
)

if __name__ == "__main__":
    main(BENCHMARK)
