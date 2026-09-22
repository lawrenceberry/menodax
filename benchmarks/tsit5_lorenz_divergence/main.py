"""Lorenz divergence-CV benchmark for Tsit5 solvers.

Runs the Lorenz system with 100,000 trajectories while sweeping the
``make_scenario(..., divergence=...)`` knob. For each solver and divergence
value, the benchmark records solve time and the actual distribution of accepted
plus rejected Tsit5 steps.

Usage:
    uv run python benchmarks/tsit5_lorenz_divergence/main.py
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Sets JAX's GPU memory policy before the first import that touches the card:
# the `reference.systems.python` modules build device arrays at import time,
# which initialises the backend and fixes that policy for the whole process.
import benchmarks.benchmark_common  # noqa: E402,F401
from benchmarks._divergence import DivergenceBenchmark, DivergenceCase, main
from modax.tsit5 import solve as tsit5_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from reference.systems.python import lorenz

jax.config.update("jax_enable_x64", True)

_N_TRAJ = 100_000

BENCHMARK = DivergenceBenchmark(
    script_dir=Path(__file__).resolve().parent,
    system="Lorenz",
    title="Lorenz Tsit5 divergence",
    solve=tsit5_solve,
    ode_fn=lorenz.ode_fn,
    t_span=lorenz.TIMES,
    dim=lorenz.N_VARS,
    n_traj=_N_TRAJ,
    divergences=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    solver_kwargs={"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8},
    make_data=lambda divergence: lorenz.make_scenario(
        _N_TRAJ, seed=42, divergence=divergence
    ),
    julia_solve=julia_tsit5_solve,
    julia_system="lorenz",
    legend_loc="upper left",
    cases=(
        DivergenceCase(key="modax tsit5 kernel", color="#f0a202", marker="s"),
        DivergenceCase(
            key="modax tsit5 kernel (sorted)",
            color="#f0a202",
            marker="P",
            sort_by_steps=True,
        ),
        DivergenceCase(
            key="julia tsit5 kernel",
            color="#d35400",
            marker="v",
            mode="julia",
            ensemble_backend="EnsembleGPUKernel",
        ),
        DivergenceCase(
            key="julia tsit5 kernel (sorted)",
            color="#d35400",
            marker="X",
            mode="julia",
            ensemble_backend="EnsembleGPUKernel",
            sort_by_steps=True,
        ),
    ),
)

if __name__ == "__main__":
    main(BENCHMARK)
