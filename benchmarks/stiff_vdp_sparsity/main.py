"""Jacobian-density benchmark for modax Rodas5P on the coupled VDP lattice.

Fixes the system at 64 dimensions (n_osc = 32) and an ensemble of 1000
identical trajectories, and sweeps the lattice's coupling range -- how many
oscillators along the ring each one couples to -- from nearest neighbours to
all-to-all. Each range gives the right-hand side a different Jacobian density,
which is the x axis. Two cases: modax Rodas5P handed the Jacobian's sparsity
pattern, which buys it one Enzyme sweep per colour of the pattern and the
compiled sparse LU, and the same solver without it, which sweeps every column
and factorises the dense matrix. The dense case does not depend on the
density; the question is how far the sparse case's advantage stretches as the
pattern fills in. A point that does not compile and solve within the case
timeout is recorded as such and omitted from the plot. Outputs a CSV and a plot
named after the GPU.

Usage:
    uv run python benchmarks/stiff_vdp_sparsity/main.py
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
    IDENTICAL_DIVERGENCE,
    Problem,
    SweepAxis,
    SweepBenchmark,
    SweepCase,
    main,
)
from modax.rodas5P import solve as rodas5P_solve
from reference.systems.python import vdp

jax.config.update("jax_enable_x64", True)

_N_OSC = 32
_ENSEMBLE_SIZE = 1000
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}

# A range of _N_OSC // 2 couples every oscillator to every other.
_COUPLING_RANGES = (1, 2, 3, 4, 6, 8, 12, 16)

DENSITY = SweepAxis(
    "coupling_range",
    "Jacobian density (fraction of structurally nonzero entries)",
    "k",
    position=lambda coupling_range: vdp.jacobian_density(_N_OSC, coupling_range),
    position_column="jacobian_density",
    log_x=False,
)


def _problem(coupling_range: int) -> Problem:
    ode_fn, _ = vdp.make_system(_N_OSC, coupling_range=coupling_range)
    y0, params = vdp.make_scenario(
        _N_OSC, _ENSEMBLE_SIZE, divergence=IDENTICAL_DIVERGENCE
    )
    return Problem(
        ode_fn, y0, params, sparsity=vdp.make_sparsity(_N_OSC, coupling_range)
    )


BENCHMARK = SweepBenchmark(
    script_dir=Path(__file__).resolve().parent,
    title=f"Rodas5P Jacobian density — {2 * _N_OSC}D coupled VDP lattice",
    axis=DENSITY,
    values=_COUPLING_RANGES,
    t_span=vdp.TIMES,
    make_problem=_problem,
    cases=(
        SweepCase(
            key="modax rodas5P fp32 (sparse)",
            color="#8c564b",
            marker="P",
            linestyle="--",
            solve_fn=rodas5P_solve,
            kwargs=_SOLVER_KWARGS,
            sparse=True,
        ),
        SweepCase(
            key="modax rodas5P fp32 (dense)",
            color="#8c564b",
            marker="X",
            solve_fn=rodas5P_solve,
            kwargs=_SOLVER_KWARGS,
        ),
    ),
)

if __name__ == "__main__":
    main(BENCHMARK)
