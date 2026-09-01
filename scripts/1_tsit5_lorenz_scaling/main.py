"""Solver scaling benchmark on the Lorenz system.

Sweeps ensemble size from 1 to 100k on a log scale and records solve time for
the local Tsit5 solver, Diffrax Tsit5, and Julia Tsit5 with both
DiffEqGPU ensemble backends. Outputs a CSV and a log-log plot per scenario
named after the GPU.

Usage:
    uv run python scripts/1_tsit5_scaling/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

import jax
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.solvers.python.diffrax_tsit5 import solve as diffrax_tsit5_solve
from reference.solvers.python.julia_tsit5 import solve as julia_tsit5_solve
from reference.systems.python import lorenz
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
from solvers.tsit5 import solve as tsit5numba_solve

jax.config.update("jax_enable_x64", True)

_T_SPAN = (0.0, 5.0)
_ENSEMBLE_SIZES = (3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000)
_N_RUNS = 1

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
_LOCAL_SOLVER_KWARGS = {
    **_SOLVER_KWARGS,
    "pcoeff": 0.0,
    "icoeff": 1.0,
    "dcoeff": 0.0,
}
_SCENARIOS = (
    ("identical", 0.0),
    ("divergent", 1.0),
)


class Row(NamedTuple):
    key: str
    label: str
    size: int
    ms: float | None


@dataclass(frozen=True)
class Case(BenchmarkCase):
    solve_fn: Callable[..., Any]
    ode_fn: Callable[..., Any] | None
    y0: Any
    t_span: tuple[float, ...]
    kwargs: dict[str, Any]
    coerce_numpy: bool = False
    system_name: str | None = None
    ensemble_backend: str | None = None

    @property
    def is_julia(self) -> bool:
        return self.system_name is not None


CASES: tuple[Case, ...] = (
    Case(
        key="modax tsit5 kernel",
        color="#f0a202",
        marker="P",
        linestyle="-",
        solve_fn=tsit5numba_solve,
        ode_fn=lorenz.ode_fn,
        y0=None,
        t_span=_T_SPAN,
        kwargs=_LOCAL_SOLVER_KWARGS,
        coerce_numpy=True,
    ),
    Case(
        key="diffrax tsit5",
        color="#2ba84a",
        marker="s",
        linestyle="-",
        solve_fn=diffrax_tsit5_solve,
        ode_fn=lorenz.ode_fn,
        y0=lorenz.Y0,
        t_span=_T_SPAN,
        kwargs=_LOCAL_SOLVER_KWARGS,
    ),
    Case(
        key="julia tsit5 array",
        color="#9b59b6",
        marker="^",
        linestyle="-",
        solve_fn=julia_tsit5_solve,
        ode_fn=None,
        y0=lorenz.Y0,
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        coerce_numpy=True,
        system_name="lorenz",
        ensemble_backend="EnsembleGPUArray",
    ),
    Case(
        key="julia tsit5 kernel",
        color="#9b59b6",
        marker="v",
        linestyle="--",
        solve_fn=julia_tsit5_solve,
        ode_fn=None,
        y0=lorenz.Y0,
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        coerce_numpy=True,
        system_name="lorenz",
        ensemble_backend="EnsembleGPUKernel",
    ),
)


def time_case(case: Case, y0, params) -> float:
    solve_y0 = case.y0 if case.y0 is not None else y0
    if case.coerce_numpy:
        solve_y0 = np.asarray(solve_y0)
        params = np.asarray(params)

    if case.is_julia:
        # Julia reports its internal solve time, excluding subprocess overhead.
        return julia_solve_time_ms(
            case.solve_fn,
            case.system_name,
            solve_y0,
            case.t_span,
            params,
            ensemble_backend=case.ensemble_backend,
            **case.kwargs,
        )

    assert case.ode_fn is not None

    def run():
        return case.solve_fn(
            case.ode_fn,
            y0=solve_y0,
            t_span=case.t_span,
            params=params,
            **case.kwargs,
        )

    return time_blocked_ms(run, _N_RUNS)


def run_benchmarks(
    cases: Sequence[Case], gpu_name: str, cache: dict
) -> dict[str, list[Row]]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows_by_scenario: dict[str, list[Row]] = {}
    for scenario, divergence in _SCENARIOS:
        print(f"\n=== {scenario} ===\n")
        rows: list[Row] = []
        for case in cases:
            print(f"{case.key}:")
            solver_cache = gpu_cache.setdefault(f"{scenario}_{case.key}", {})
            for size in _ENSEMBLE_SIZES:
                size_key = str(size)
                if size_key in solver_cache:
                    ms = solver_cache[size_key]
                    ms_text = format_cached_timing(ms)
                    print(f"  {case.key:<20} n={size:>7} ... (cached) {ms_text}")
                else:
                    y0, params = lorenz.make_scenario(size, divergence=divergence)
                    ms = collect_timed_timing(
                        case.key,
                        f"n={size:>7}",
                        lambda: time_case(case, y0, params),
                        label_width=20,
                    )
                    solver_cache[size_key] = ms
                    save_cache(_CACHE_PATH, cache)
                rows.append(
                    Row(
                        key=case.key,
                        label=case.key,
                        size=size,
                        ms=timing_value_or_none(ms),
                    )
                )
            print()
        rows_by_scenario[scenario] = rows
    return rows_by_scenario


def save_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_key", "solver", "ensemble_size", "solve_time_ms"])
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    rows: list[Row],
    cases: Sequence[Case],
    gpu_name: str,
    scenario: str,
    output_path: Path,
) -> None:
    configure_latex_plot_style(plt)
    title = f"Tsit5 scaling — Lorenz ({scenario}) — {gpu_name}"
    print_plot_title(title)
    fig, ax = plt.subplots(figsize=(7, 5))
    for case in cases:
        sizes, times_ms = drop_none_rows(rows, case.key)
        if not sizes:
            continue
        ax.plot(
            sizes,
            times_ms,
            marker=case.marker,
            color=case.color,
            linestyle=case.linestyle,
            label=case.key,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Ensemble size")
    ax.set_ylabel("Solve time (ms)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(_ENSEMBLE_SIZES)
    ax.set_xticklabels([str(n) for n in _ENSEMBLE_SIZES], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main() -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(_CACHE_PATH)
    rows_by_scenario = run_benchmarks(CASES, gpu_name, cache)

    slug = gpu_slug(gpu_name)
    for scenario, rows in rows_by_scenario.items():
        csv_path = _SCRIPT_DIR / f"results-{slug}-{scenario}.csv"
        plot_path = _SCRIPT_DIR / f"plot-{slug}-{scenario}.png"
        save_csv(rows, csv_path)
        plot(rows, CASES, gpu_name, scenario, plot_path)


if __name__ == "__main__":
    main()
