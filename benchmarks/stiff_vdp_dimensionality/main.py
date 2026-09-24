"""Dimensionality scaling benchmark on the coupled VDP lattice system.

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 identical trajectories and records solve time for modax
Rodas5P with fp32 and fp64 LU precision, modax Rodas5P fp32 handed the
lattice's Jacobian sparsity pattern (one Enzyme sweep per colour and the
compiled sparse LU), Diffrax Kvaerno5, and Julia Rodas5P with both DiffEqGPU
ensemble backends. A point that fails, or does not compile
and solve within the case timeout, is recorded as such and omitted from the
plot. Outputs a CSV and a log-log plot named after the GPU.

Usage:
    uv run python benchmarks/stiff_vdp_dimensionality/main.py
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
    DIMENSION,
    IDENTICAL_DIVERGENCE,
    Problem,
    SweepBenchmark,
    SweepCase,
    main,
)
from modax.rodas5P import solve as rodas5P_solve
from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.systems.python import vdp

jax.config.update("jax_enable_x64", True)

_ENSEMBLE_SIZE = 1000
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}


def _problem(dim: int) -> Problem:
    n_osc = dim // 2
    ode_fn, _ = vdp.make_system(n_osc)
    y0, params = vdp.make_scenario(
        n_osc, _ENSEMBLE_SIZE, divergence=IDENTICAL_DIVERGENCE
    )
    return Problem(
        ode_fn,
        y0,
        params,
        julia_system_config={"n_osc": n_osc},
        sparsity=vdp.make_sparsity(n_osc),
    )


BENCHMARK = SweepBenchmark(
    script_dir=Path(__file__).resolve().parent,
    title="Rodas5P dimensionality — coupled VDP lattice",
    axis=DIMENSION,
    values=(2, 4, 6, 8, 10, 12, 16, 32, 64, 96, 128),
    t_span=vdp.TIMES,
    make_problem=_problem,
    julia_solve=julia_rodas5P_solve,
    julia_system="vdp",
    cases=(
        SweepCase(
            key="modax rodas5P fp32 (sparse)",
            color="#8c564b",
            marker="o",
            linestyle=":",
            solve_fn=rodas5P_solve,
            kwargs={**_SOLVER_KWARGS, "lu_precision": "fp32"},
            sparse=True,
        ),
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
