"""Dimensionality scaling benchmark for KenCarp5 on the Brusselator system.

Sweeps ODE dimension from 4 to 128 (n_grid = 2 to 64, dim = 2*n_grid) on a
log scale with a fixed ensemble of 1000 trajectories and records solve time
for the local KenCarp5 solver (with both ``lu_precision="fp64"`` and
``lu_precision="fp32"`` LU factorization), Diffrax KenCarp5, and Julia KenCarp5
on ``EnsembleGPUArray``. DiffEqGPU has no ``GPUKenCarp5`` so the kernel backend
is skipped. Failed solves (e.g. at extreme dims) are stored as null and omitted
from the plot. Runs both ``identical`` and ``divergent`` scenarios.

Usage:
    uv run python scripts/12_kencarp5_brusselator_dimensionality/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_kencarp5 import solve as diffrax_kencarp5_solve
from reference.solvers.python.julia_kencarp5 import solve as julia_kencarp5_solve
from reference.systems.python import brusselator
from scripts.benchmark_common import (
    BenchmarkCase,
    collect_timed_timing,
    configure_latex_plot_style,
    drop_none_rows,
    format_cached_timing,
    get_gpu_name,
    gpu_slug,
    julia_solve_time_ms,
    load_cache,
    print_plot_title,
    save_cache,
    time_blocked_ms,
    timing_value_or_none,
)
from solvers.kencarp5 import solve as kencarp5numba_solve
from solvers.rodas5P import solve as rodas5Pnumba_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = brusselator.TIMES
_ENSEMBLE_SIZE = 1000
_N_RUNS = 1

# Total ODE dimension (= 2 * n_grid). Matches the script 9 VDP sweep shape.
_DIMENSIONS = (4, 8, 12, 16, 24, 32, 48, 64, 128)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
_SCENARIOS = (
    ("identical", 0.0),
    ("divergent", 1.0),
)


@dataclass(frozen=True)
class Case(BenchmarkCase):
    solve_fn: Callable[..., Any] | None = None
    mode: str = "kencarp"
    t_span: Any = None
    kwargs: dict[str, Any] | None = None
    lu_precision: str | None = None
    system_name: str | None = None
    ensemble_backend: str | None = None

    @property
    def is_julia(self) -> bool:
        return self.system_name is not None


CASES: tuple[Case, ...] = (
    Case(
        key="modax kencarp5 kernel fp64",
        color="#f0a202",
        marker="P",
        solve_fn=kencarp5numba_solve,
        mode="custom",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        lu_precision="fp64",
    ),
    Case(
        key="modax kencarp5 kernel fp32",
        color="#f0a202",
        marker="X",
        linestyle="--",
        solve_fn=kencarp5numba_solve,
        mode="custom",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        lu_precision="fp32",
    ),
    Case(
        key="modax rodas5P kernel fp32",
        color="#8c564b",
        marker="*",
        linestyle="--",
        solve_fn=rodas5Pnumba_solve,
        mode="rodas_kernel",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        lu_precision="fp32",
    ),
    Case(
        key="diffrax kencarp5",
        color="#2ba84a",
        marker="s",
        solve_fn=diffrax_kencarp5_solve,
        mode="kencarp",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
    ),
    # DiffEqGPU 3.13 has no SplitODEProblem support on EnsembleGPUArray, so this
    # row is fully implicit rather than IMEX.
    Case(
        key="julia kencarp5 array",
        color="#9b59b6",
        marker="^",
        solve_fn=julia_kencarp5_solve,
        mode="julia",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        system_name="brusselator",
        ensemble_backend="EnsembleGPUArray",
    ),
)


def time_case(case: Case, dim: int, *, divergence: float) -> float:
    n_grid = dim // 2
    y0_batch, params = brusselator.make_scenario(
        n_grid, _ENSEMBLE_SIZE, divergence=divergence
    )
    kwargs = case.kwargs or {}

    if case.is_julia:
        return julia_solve_time_ms(
            case.solve_fn,
            case.system_name,
            y0_batch,
            case.t_span,
            params,
            system_config={"n_grid": n_grid},
            ensemble_backend=case.ensemble_backend,
            **kwargs,
        )

    assert case.solve_fn is not None
    ex_fn, im_fn, ode_fn, _, im_jac_fn, jac_fn = brusselator.make_system(n_grid)

    def run():
        if case.mode == "custom":
            return case.solve_fn(
                ex_fn,
                im_fn,
                im_jac_fn,
                y0=y0_batch,
                t_span=case.t_span,
                params=params,
                lu_precision=case.lu_precision,
                **kwargs,
            )
        if case.mode == "rodas_kernel":
            return case.solve_fn(
                ode_fn,
                jac_fn,
                y0=y0_batch,
                t_span=case.t_span,
                params=params,
                lu_precision=case.lu_precision,
                **kwargs,
            )
        y0_j = jnp.asarray(y0_batch)
        p_j = jnp.asarray(params)
        return case.solve_fn(
            ex_fn,
            im_fn,
            y0_j,
            case.t_span,
            p_j,
            **({"lu_precision": case.lu_precision} if case.lu_precision else {}),
            **kwargs,
        )

    return time_blocked_ms(run, _N_RUNS)


def collect_timing(case: Case, dim: int, divergence: float):
    return collect_timed_timing(
        case.key,
        f"dim={dim:>4}",
        lambda: time_case(case, dim, divergence=divergence),
        label_width=28,
    )


_Row = tuple[str, str, int, float | None]


def run_benchmarks(
    cases: Sequence[Case], gpu_name: str, cache: dict, scenario: str, divergence: float
) -> list[_Row]:
    scenario_cache = cache.setdefault(gpu_name, {}).setdefault(scenario, {})
    rows: list[_Row] = []
    for case in cases:
        print(f"\n{case.key}:")
        solver_cache = scenario_cache.setdefault(case.key, {})
        for dim in _DIMENSIONS:
            dim_key = str(dim)
            if dim_key in solver_cache:
                ms = solver_cache[dim_key]
                ms_text = format_cached_timing(ms)
                print(f"  {case.key:<28} dim={dim:>4} ... (cached) {ms_text}")
            else:
                ms = collect_timing(case, dim, divergence)
                solver_cache[dim_key] = ms
                save_cache(_CACHE_PATH, cache)
            rows.append((case.key, case.key, dim, timing_value_or_none(ms)))
        print()
    return rows


def save_csv(rows: list[_Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_key", "solver", "dim", "solve_time_ms"])
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    rows: list[_Row],
    cases: Sequence[Case],
    gpu_name: str,
    output_path: Path,
    scenario: str,
) -> None:
    configure_latex_plot_style(plt)
    title = f"KenCarp5 dimensionality — {scenario} — Brusselator — {gpu_name}"
    print_plot_title(title)
    fig, ax = plt.subplots(figsize=(7, 5))
    for case in cases:
        dims, times_ms = drop_none_rows(rows, case.key)
        if not dims:
            continue
        ax.plot(
            dims,
            times_ms,
            marker=case.marker,
            color=case.color,
            linestyle=case.linestyle,
            label=case.key,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ODE dimension")
    ax.set_ylabel("Solve time (ms)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(_DIMENSIONS)
    ax.set_xticklabels([str(d) for d in _DIMENSIONS], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def _scenario_output_paths(gpu_name: str, scenario: str) -> tuple[Path, Path]:
    slug = gpu_slug(gpu_name)
    return (
        _SCRIPT_DIR / f"results-{slug}-{scenario}.csv",
        _SCRIPT_DIR / f"plot-{slug}-{scenario}.png",
    )


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(_CACHE_PATH)
    for scenario, divergence in _SCENARIOS:
        print(f"\n=== Scenario: {scenario} ===\n")
        rows = run_benchmarks(CASES, gpu_name, cache, scenario, divergence)
        csv_path, plot_path = _scenario_output_paths(gpu_name, scenario)
        save_csv(rows, csv_path)
        plot(rows, CASES, gpu_name, plot_path, scenario)


if __name__ == "__main__":
    main()
