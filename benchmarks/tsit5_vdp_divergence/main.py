"""High-dimensional non-stiff VDP divergence benchmark for Tsit5 solvers.

Runs the 64D non-stiff coupled VDP lattice (n_osc = 32, mu = 1.0) with 1000
trajectories while sweeping the ``make_scenario(..., divergence=...)`` knob.
For each solver and divergence value, the benchmark records solve time and the
actual distribution of accepted plus rejected Tsit5 steps.

Usage:
    uv run python benchmarks/tsit5_vdp_divergence/main.py
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
from reference.systems.python import vdp

jax.config.update("jax_enable_x64", True)

_MU_NONSTIFF = 1.0
_D = 10.0
_OMEGA = 1.0

_N_OSC = 32
_DIM = 2 * _N_OSC
_ENSEMBLE_SIZE = 1000

_ODE_FN, _ = vdp.make_system(_N_OSC, mu=_MU_NONSTIFF)

BENCHMARK = DivergenceBenchmark(
    script_dir=Path(__file__).resolve().parent,
    system=f"non-stiff coupled VDP lattice (μ={_MU_NONSTIFF})",
    title=f"{_DIM}D coupled VDP non-stiff (μ={_MU_NONSTIFF}) Tsit5 divergence",
    solve=tsit5_solve,
    ode_fn=_ODE_FN,
    t_span=vdp.TIMES,
    dim=_DIM,
    n_traj=_ENSEMBLE_SIZE,
    divergences=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0),
    solver_kwargs={"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8},
    make_data=lambda divergence: vdp.make_scenario(
        _N_OSC, _ENSEMBLE_SIZE, seed=42, divergence=divergence
    ),
    julia_solve=julia_tsit5_solve,
    julia_system="vdp",
    julia_system_config={"n_osc": _N_OSC, "mu": _MU_NONSTIFF, "d": _D, "omega": _OMEGA},
    extra_fields={"n_osc": _N_OSC},
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
