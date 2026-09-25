"""High-dimensional VDP divergence benchmark for Rodas5P solvers.

Runs the 96D coupled VDP lattice (n_osc = 48) with 1000 trajectories while
sweeping the ``make_scenario(..., divergence=...)`` knob, timing the fp32
Rodas5P solve both as-is and with trajectories pre-sorted by attempted step
count. For each solver and divergence value, the benchmark records solve time
and the actual distribution of accepted plus rejected Rodas5P steps.

Usage:
    uv run python benchmarks/stiff_vdp_divergence/main.py
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
from benchmarks.benchmark_common import (  # noqa: E402
    MODAX_COLOR,
)
from modax.rodas5P import solve as rodas5P_solve
from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.systems.python import vdp

jax.config.update("jax_enable_x64", True)

_N_OSC = 48
_DIM = 2 * _N_OSC
_ENSEMBLE_SIZE = 1000

_ODE_FN, _ = vdp.make_system(_N_OSC)

BENCHMARK = DivergenceBenchmark(
    script_dir=Path(__file__).resolve().parent,
    system="coupled VDP lattice",
    title=f"{_DIM}D coupled VDP divergence",
    solve=rodas5P_solve,
    ode_fn=_ODE_FN,
    t_span=vdp.TIMES,
    dim=_DIM,
    n_traj=_ENSEMBLE_SIZE,
    divergences=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0),
    solver_kwargs={"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8},
    make_data=lambda divergence: vdp.make_scenario(
        _N_OSC, _ENSEMBLE_SIZE, seed=42, divergence=divergence
    ),
    julia_solve=julia_rodas5P_solve,
    julia_system="vdp",
    julia_system_config={"n_osc": _N_OSC},
    extra_fields={"n_osc": _N_OSC},
    cases=(
        DivergenceCase(key="modax rodas5P fp32", color=MODAX_COLOR, marker="s"),
        DivergenceCase(
            key="modax rodas5P fp32 (sorted)",
            color=MODAX_COLOR,
            marker="P",
            sort_by_steps=True,
        ),
    ),
)

if __name__ == "__main__":
    main(BENCHMARK)
