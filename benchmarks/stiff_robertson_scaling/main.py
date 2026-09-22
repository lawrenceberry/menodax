"""Solver scaling benchmark on the Robertson stiff system.

Sweeps ensemble size from 1 to 100k on a log scale and records solve time for
modax Rodas5P with fp32 and fp64 LU precision, Diffrax Kvaerno5, and Julia
Rodas5P with both DiffEqGPU ensemble backends. Outputs a CSV and a log-log plot
per scenario named after the GPU.

Usage:
    uv run python benchmarks/stiff_robertson_scaling/main.py
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Sets JAX's GPU memory policy before the first import that touches the card:
# the `reference.systems.python` modules build device arrays at import time,
# which initialises the backend and fixes that policy for the whole process.
import benchmarks.benchmark_common  # noqa: E402,F401
from benchmarks._sweep import ENSEMBLE_SIZE, Problem, SweepBenchmark, SweepCase, main
from modax.rodas5P import solve as rodas5P_solve
from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.systems.python import robertson

jax.config.update("jax_enable_x64", True)

_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}


def _problem(size: int, divergence: float) -> Problem:
    return Problem(
        robertson.ode_fn, *robertson.make_scenario(size, divergence=divergence)
    )


def _skip(case: SweepCase, size: int) -> str | None:
    if case.ensemble_backend == "EnsembleGPUArray" and size == 1:
        # DiffEqGPU branches `trajectories == 1` to EnsembleSerial before the
        # EnsembleGPUArray path runs. That value is therefore neither a GPU
        # ensemble timing nor directly comparable to the larger-size scaling
        # curve, so omit the point instead of recording a misleading fast path.
        return "scalar fast path is not comparable to EnsembleGPUArray scaling"
    return None


BENCHMARK = SweepBenchmark(
    script_dir=Path(__file__).resolve().parent,
    title="Rodas5P scaling — Robertson ({scenario})",
    axis=ENSEMBLE_SIZE,
    values=(1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000),
    t_span=robertson.TIMES,
    make_problem=_problem,
    julia_solve=julia_rodas5P_solve,
    julia_system="robertson",
    skip=_skip,
    legend_loc="upper left",
    cases=(
        SweepCase(
            key="modax rodas5P fp32",
            color="#8c564b",
            marker="P",
            linestyle="--",
            solve_fn=rodas5P_solve,
            kwargs={**_SOLVER_KWARGS, "lu_precision": "fp32"},
        ),
        SweepCase(
            key="modax rodas5P fp64",
            color="#8c564b",
            marker="X",
            solve_fn=rodas5P_solve,
            kwargs={**_SOLVER_KWARGS, "lu_precision": "fp64"},
        ),
        SweepCase(
            key="diffrax kvaerno5",
            color="#2ba84a",
            marker="s",
            solve_fn=diffrax_kvaerno5_solve,
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="julia rodas5P EnsembleGPUArray",
            color="#9b59b6",
            marker="^",
            mode="julia",
            ensemble_backend="EnsembleGPUArray",
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="julia rodas5P EnsembleGPUKernel",
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
