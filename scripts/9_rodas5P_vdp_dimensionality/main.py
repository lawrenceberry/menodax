"""Dimensionality scaling benchmark on the coupled VDP lattice system.

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 trajectories and records solve time for the modax
Rodas5P kernel with fp32 and fp64 LU precision, Diffrax Kvaerno5, and Julia
Rodas5P with both DiffEqGPU ensemble backends. EnsembleGPUKernel failures
(expected for large dimensions) are stored as null and omitted from the plot.
Runs both "identical" and "divergent" scenarios; outputs a CSV and log-log plot
per scenario, named after the GPU and scenario.

Usage:
    uv run python scripts/9_rodas5P_vdp_dimensionality/main.py
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.systems.python import vdp
from scripts._sweep import DIMENSION, Problem, SweepBenchmark, SweepCase, main
from solvers.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

_ENSEMBLE_SIZE = 1000
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}


def _problem(dim: int, divergence: float) -> Problem:
    n_osc = dim // 2
    ode_fn, _ = vdp.make_system(n_osc)
    y0, params = vdp.make_scenario(n_osc, _ENSEMBLE_SIZE, divergence=divergence)
    return Problem(ode_fn, y0, params, julia_system_config={"n_osc": n_osc})


BENCHMARK = SweepBenchmark(
    script_dir=Path(__file__).resolve().parent,
    title="Rodas5P dimensionality — {scenario} — coupled VDP lattice",
    axis=DIMENSION,
    values=(2, 4, 6, 8, 10, 12, 16, 32, 64, 96, 128),
    t_span=vdp.TIMES,
    make_problem=_problem,
    julia_solve=julia_rodas5P_solve,
    julia_system="vdp",
    cases=(
        SweepCase(
            key="modax rodas5P kernel fp32",
            color="#8c564b",
            marker="P",
            linestyle="--",
            solve_fn=rodas5P_solve,
            kwargs={**_SOLVER_KWARGS, "lu_precision": "fp32"},
        ),
        SweepCase(
            key="modax rodas5P kernel fp64",
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
            key="julia rodas5P array",
            color="#9b59b6",
            marker="^",
            mode="julia",
            ensemble_backend="EnsembleGPUArray",
            kwargs=_SOLVER_KWARGS,
        ),
        SweepCase(
            key="julia rodas5P kernel",
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
