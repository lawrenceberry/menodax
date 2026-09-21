"""Robertson divergence-CV benchmark for Rodas5P solvers.

Runs the Robertson system with 30,000 trajectories while sweeping the
``make_scenario(..., divergence=...)`` knob. For each solver and divergence
value, the benchmark records solve time and the actual distribution of accepted
plus rejected Rodas5P steps.

Usage:
    uv run python scripts/7_rodas5P_robertson_divergence/main.py
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.systems.python import robertson
from scripts._divergence import DivergenceBenchmark, DivergenceCase, main
from solvers.rodas5P import solve as rodas5P_solve

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 30_000
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

BENCHMARK = DivergenceBenchmark(
    script_dir=Path(__file__).resolve().parent,
    system="Robertson",
    title="Robertson Rodas5P divergence",
    solve=rodas5P_solve,
    ode_fn=robertson.ode_fn,
    t_span=robertson.TIMES,
    dim=robertson.N_VARS,
    n_traj=_N_TRAJ,
    divergences=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
    solver_kwargs=_SOLVER_KWARGS,
    make_data=lambda divergence: robertson.make_scenario(
        _N_TRAJ, seed=42, divergence=divergence
    ),
    julia_solve=julia_rodas5P_solve,
    julia_system="robertson",
    cases=(
        DivergenceCase(key="modax rodas5P kernel fp32", color="#f0a202", marker="s"),
        DivergenceCase(
            key="modax rodas5P kernel fp32 (sorted)",
            color="#f0a202",
            marker="P",
            sort_by_steps=True,
        ),
        # DivergenceCase(
        #     key="diffrax kvaerno5",
        #     color="#2ba84a",
        #     marker="^",
        #     mode="timing",
        #     solve_fn=diffrax_kvaerno5_solve,
        #     kwargs={**_SOLVER_KWARGS, "max_steps": 1_000_000},
        #     max_divergence=1.5,
        # ),
        # DivergenceCase(
        #     key="julia rodas5P array",
        #     color="#9b59b6",
        #     marker="D",
        #     mode="julia",
        #     ensemble_backend="EnsembleGPUArray",
        # ),
        DivergenceCase(
            key="julia rodas5P kernel",
            color="#d35400",
            marker="v",
            mode="julia",
            ensemble_backend="EnsembleGPUKernel",
            max_divergence=2.5,
        ),
        DivergenceCase(
            key="julia rodas5P kernel (sorted)",
            color="#d35400",
            marker="X",
            mode="julia",
            ensemble_backend="EnsembleGPUKernel",
            sort_by_steps=True,
            max_divergence=2.5,
        ),
    ),
)

if __name__ == "__main__":
    main(BENCHMARK)
