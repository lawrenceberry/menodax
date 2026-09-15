"""Dimensionality scaling benchmark on the coupled VDP lattice system.

Sweeps ODE dimension from 2 to 128 (n_osc = 1 to 64) on a log scale with a
fixed ensemble of 1000 trajectories and records solve time for the local Rodas5P
solver with fp32/fp64 LU precision, Diffrax Kvaerno5, and Julia Rodas5P with
both DiffEqGPU ensemble backends. EnsembleGPUKernel failures (expected for
large dimensions) are stored as null and omitted from the plot. Runs both
"identical" and "divergent" scenarios; outputs a CSV and log-log plot per
scenario, named after the GPU and scenario.

Usage:
    uv run python scripts/9_rodas5P_vdp_dimensionality/main.py
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

from reference.solvers.python.diffrax_kvaerno5 import solve as diffrax_kvaerno5_solve
from reference.solvers.python.julia_rodas5P import solve as julia_rodas5P_solve
from reference.systems.python import vdp
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
from solvers.rodas5P import solve as rodas5Pnumba_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = vdp.TIMES
_ENSEMBLE_SIZE = 1000
_N_RUNS = 1

# Powers of 2 from 2 to 128 (n_osc = dim // 2: 1, 2, 4, 8, 16, 32, 64)
_DIMENSIONS = (2, 4, 6, 8, 10, 12, 16, 32, 64, 96, 128)

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
    mode: str = "diffrax"
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
        key="modax rodas5P kernel fp32",
        color="#8c564b",
        marker="P",
        linestyle="--",
        solve_fn=rodas5Pnumba_solve,
        mode="custom",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        lu_precision="fp32",
    ),
    Case(
        key="modax rodas5P kernel fp64",
        color="#8c564b",
        marker="X",
        solve_fn=rodas5Pnumba_solve,
        mode="custom",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        lu_precision="fp64",
    ),
    Case(
        key="diffrax kvaerno5",
        color="#2ba84a",
        marker="s",
        solve_fn=diffrax_kvaerno5_solve,
        mode="diffrax",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
    ),
    Case(
        key="julia rodas5P array",
        color="#9b59b6",
        marker="^",
        solve_fn=julia_rodas5P_solve,
        mode="julia",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        system_name="vdp",
        ensemble_backend="EnsembleGPUArray",
    ),
    Case(
        key="julia rodas5P kernel",
        color="#9b59b6",
        marker="v",
        linestyle="--",
        solve_fn=julia_rodas5P_solve,
        mode="julia",
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        system_name="vdp",
        ensemble_backend="EnsembleGPUKernel",
    ),
)


def time_case(case: Case, dim: int, *, divergence: float) -> float:
    n_osc = dim // 2
    y0_batch, params = vdp.make_scenario(n_osc, _ENSEMBLE_SIZE, divergence=divergence)
    kwargs = case.kwargs or {}

    if case.is_julia:
        return julia_solve_time_ms(
            case.solve_fn,
            case.system_name,
            y0_batch,
            case.t_span,
            params,
            system_config={"n_osc": n_osc},
            ensemble_backend=case.ensemble_backend,
            **kwargs,
        )

    if case.mode == "custom":
        ode_fn_nb, _, _ = vdp.make_system(n_osc)

        def run_custom():
            extra = (
                {"lu_precision": case.lu_precision}
                if case.lu_precision is not None
                else {}
            )
            return case.solve_fn(
                ode_fn_nb,
                y0=y0_batch,
                t_span=case.t_span,
                params=params,
                **extra,
                **kwargs,
            )

        return time_blocked_ms(run_custom, _N_RUNS)

    assert case.solve_fn is not None
    ode_fn, _, _ = vdp.make_system(n_osc)
    y0_j = jnp.asarray(y0_batch)
    p_j = jnp.asarray(params)

    def run():
        return case.solve_fn(
            ode_fn,
            y0=y0_j,
            t_span=case.t_span,
            params=p_j,
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


def _migrate_cache(cache: dict) -> dict:
    for gpu, gpu_data in cache.items():
        first_val = next(iter(gpu_data.values()), None)
        if isinstance(first_val, dict) and not any(
            k in {scenario for scenario, _ in _SCENARIOS} for k in gpu_data
        ):
            cache[gpu] = {"identical": gpu_data}
    return cache


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
    title = f"Rodas5P dimensionality — {scenario} — coupled VDP lattice — {gpu_name}"
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

    cache = _migrate_cache(load_cache(_CACHE_PATH))

    for scenario, divergence in _SCENARIOS:
        print(f"\n=== Scenario: {scenario} ===\n")
        rows = run_benchmarks(CASES, gpu_name, cache, scenario, divergence)
        csv_path, plot_path = _scenario_output_paths(gpu_name, scenario)
        save_csv(rows, csv_path)
        plot(rows, CASES, gpu_name, plot_path, scenario)


if __name__ == "__main__":
    main()
