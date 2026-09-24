"""Dimensionality scaling benchmark on the non-stiff coupled VDP lattice (Tsit5).

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 identical trajectories and records solve time for modax
Tsit5, Diffrax Tsit5, torchdiffeq Dopri8 (one adaptive step shared by the
whole ensemble) and Julia Tsit5 with both DiffEqGPU ensemble backends. A
point that fails, or does not compile and solve within the case timeout, is
recorded as such and omitted from the plot. Outputs a CSV and a log-log plot
named after the GPU.

Uses the non-stiff coupled VDP variant (mu = 1.0) so that explicit Tsit5
remains an appropriate solver.

Usage:
    uv run python benchmarks/nonstiff_vdp_dimensionality/main.py
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
from modax.tsit5 import clear_caches as tsit5_clear_caches
from modax.tsit5 import solve as tsit5_solve
from reference.solvers.python.diffrax_tsit5 import solve as diffrax_tsit5_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from reference.solvers.python.torchdiffeq_dopri8 import (
    solve as torchdiffeq_dopri8_solve,
)
from reference.systems.python import vdp

jax.config.update("jax_enable_x64", True)

_MU_NONSTIFF = 1.0
_D = 10.0
_OMEGA = 1.0
_ENSEMBLE_SIZE = 1000

_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
_LOCAL_SOLVER_KWARGS = {**_SOLVER_KWARGS, "pcoeff": 0.0, "icoeff": 1.0, "dcoeff": 0.0}


def _problem(dim: int) -> Problem:
    n_osc = dim // 2
    ode_fn, _ = vdp.make_system(n_osc, mu=_MU_NONSTIFF)
    y0, params = vdp.make_scenario(
        n_osc, _ENSEMBLE_SIZE, divergence=IDENTICAL_DIVERGENCE
    )
    return Problem(
        ode_fn,
        y0,
        params,
        julia_system_config={
            "n_osc": n_osc,
            "mu": _MU_NONSTIFF,
            "d": _D,
            "omega": _OMEGA,
        },
    )


BENCHMARK = SweepBenchmark(
    script_dir=Path(__file__).resolve().parent,
    title=f"Tsit5 dimensionality — coupled VDP lattice (μ={_MU_NONSTIFF})",
    axis=DIMENSION,
    values=(2, 4, 6, 8, 10, 12, 16, 32, 64, 128),
    t_span=vdp.TIMES,
    make_problem=_problem,
    julia_solve=julia_tsit5_solve,
    julia_system="vdp",
    cases=(
        SweepCase(
            key="modax tsit5",
            color="#f0a202",
            marker="P",
            solve_fn=tsit5_solve,
            kwargs=_LOCAL_SOLVER_KWARGS,
            # Release per-dimension workspaces and compiled kernels before the
            # next dimension allocates its own.
            cleanup=tsit5_clear_caches,
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
