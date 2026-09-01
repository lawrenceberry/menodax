"""KenCarp5 ensemble-size scaling benchmark on the Brusselator system.

Fixed grid size (n_grid=32, dim=64). Sweeps ensemble size from 3 to 100k on
a log scale and records solve time for the local KenCarp5 solver (with both
``lu_precision="fp64"`` and ``lu_precision="fp32"`` LU factorization), Diffrax
KenCarp5, and Julia KenCarp5 on ``EnsembleGPUArray``. Julia
provides no ``GPUKenCarp5`` so the kernel backend is skipped. Writes one CSV
and a log-log plot per scenario, named after the GPU.

Usage:
    uv run python scripts/11_kencarp5_brusselator_scaling/main.py
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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

_N_GRID = 32
_T_SPAN = brusselator.TIMES
_ENSEMBLE_SIZES = (3, 10, 30, 100, 300, 1000, 3000, 10000, 30000)
_N_RUNS = 1

_SCRIPT_DIR = Path(__file__).resolve().parent
_CACHE_PATH = _SCRIPT_DIR / "results.json"
_SOLVER_KWARGS = {"first_step": 1e-4, "rtol": 1e-6, "atol": 1e-8}
(
    _EXPLICIT_ODE_FN,
    _IMPLICIT_ODE_FN,
    _ODE_FN,
    _,
    _IMPLICIT_JAC_FN,
    _JAC_FN,
) = brusselator.make_system(_N_GRID)
_SCENARIOS = (
    ("identical", 0.0),
    ("divergent", 1.0),
)


@dataclass(frozen=True)
class Case(BenchmarkCase):
    solve_fn: Callable[..., Any] | None = None
    explicit_ode_fn: Callable[..., Any] | None = None
    implicit_ode_fn: Callable[..., Any] | None = None
    implicit_jac_fn: Callable[..., Any] | None = None
    ode_fn: Callable[..., Any] | None = None
    jac_fn: Callable[..., Any] | None = None
    t_span: Any = None
    kwargs: dict[str, Any] | None = None
    lu_precision: str | None = None
    coerce_jax: bool = False
    coerce_numpy: bool = False
    system_name: str | None = None
    system_config: dict[str, Any] | None = None
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
        explicit_ode_fn=_EXPLICIT_ODE_FN,
        implicit_ode_fn=_IMPLICIT_ODE_FN,
        implicit_jac_fn=_IMPLICIT_JAC_FN,
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        lu_precision="fp64",
        coerce_numpy=True,
    ),
    Case(
        key="modax kencarp5 kernel fp32",
        color="#f0a202",
        marker="X",
        linestyle="--",
        solve_fn=kencarp5numba_solve,
        explicit_ode_fn=_EXPLICIT_ODE_FN,
        implicit_ode_fn=_IMPLICIT_ODE_FN,
        implicit_jac_fn=_IMPLICIT_JAC_FN,
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        lu_precision="fp32",
        coerce_numpy=True,
    ),
    Case(
        key="modax rodas5P kernel fp32",
        color="#8c564b",
        marker="*",
        linestyle="--",
        solve_fn=rodas5Pnumba_solve,
        ode_fn=_ODE_FN,
        jac_fn=_JAC_FN,
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        lu_precision="fp32",
        coerce_numpy=True,
    ),
    Case(
        key="diffrax kencarp5",
        color="#2ba84a",
        marker="s",
        solve_fn=diffrax_kencarp5_solve,
        explicit_ode_fn=_EXPLICIT_ODE_FN,
        implicit_ode_fn=_IMPLICIT_ODE_FN,
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        coerce_jax=True,
    ),
    # DiffEqGPU 3.13 has no SplitODEProblem support on EnsembleGPUArray, so this
    # row is fully implicit rather than IMEX.
    Case(
        key="julia kencarp5 array",
        color="#9b59b6",
        marker="^",
        solve_fn=julia_kencarp5_solve,
        t_span=_T_SPAN,
        kwargs=_SOLVER_KWARGS,
        coerce_numpy=True,
        system_name="brusselator",
        system_config={"n_grid": _N_GRID},
        ensemble_backend="EnsembleGPUArray",
    ),
)


def time_case(case: Case, y0, params) -> float:
    solve_y0 = y0
    solve_params = params
    if case.coerce_jax:
        solve_y0 = jnp.asarray(solve_y0)
        solve_params = jnp.asarray(solve_params)
    if case.coerce_numpy:
        solve_y0 = np.asarray(solve_y0)
        solve_params = np.asarray(solve_params)

    kwargs = case.kwargs or {}
    if case.is_julia:
        return julia_solve_time_ms(
            case.solve_fn,
            case.system_name,
            solve_y0,
            case.t_span,
            solve_params,
            system_config=case.system_config,
            ensemble_backend=case.ensemble_backend,
            **kwargs,
        )

    assert case.solve_fn is not None

    def run():
        if case.implicit_jac_fn is not None:
            return case.solve_fn(
                case.explicit_ode_fn,
                case.implicit_ode_fn,
                case.implicit_jac_fn,
                y0=solve_y0,
                t_span=case.t_span,
                params=solve_params,
                lu_precision=case.lu_precision,
                **kwargs,
            )
        if case.ode_fn is not None and case.jac_fn is not None:
            return case.solve_fn(
                case.ode_fn,
                case.jac_fn,
                y0=solve_y0,
                t_span=case.t_span,
                params=solve_params,
                lu_precision=case.lu_precision,
                **kwargs,
            )
        if case.ode_fn is not None:
            return case.solve_fn(
                case.ode_fn,
                solve_y0,
                case.t_span,
                solve_params,
                lu_precision=case.lu_precision,
                **kwargs,
            )
        return case.solve_fn(
            case.explicit_ode_fn,
            case.implicit_ode_fn,
            solve_y0,
            case.t_span,
            solve_params,
            **({"lu_precision": case.lu_precision} if case.lu_precision else {}),
            **kwargs,
        )

    return time_blocked_ms(run, _N_RUNS)


def collect_timing(case: Case, size: int, y0, params):
    return collect_timed_timing(
        case.key,
        f"n={size:>7}",
        lambda: time_case(case, y0, params),
        label_width=28,
    )


_Row = tuple[str, str, int, float | None]


def run_benchmarks(
    cases: Sequence[Case], gpu_name: str, cache: dict
) -> dict[str, list[_Row]]:
    gpu_cache = cache.setdefault(gpu_name, {})
    rows_by_scenario: dict[str, list[_Row]] = {}
    for scenario, divergence in _SCENARIOS:
        print(f"\n=== {scenario} ===\n")
        rows: list[_Row] = []
        for case in cases:
            print(f"{case.key}:")
            solver_cache = gpu_cache.setdefault(f"{scenario}_{case.key}", {})
            for size in _ENSEMBLE_SIZES:
                size_key = str(size)
                if size_key in solver_cache:
                    ms = solver_cache[size_key]
                    ms_text = format_cached_timing(ms)
                    print(f"  {case.key:<28} n={size:>7} ... (cached) {ms_text}")
                else:
                    y0, params = brusselator.make_scenario(
                        _N_GRID, size, divergence=divergence
                    )
                    ms = collect_timing(case, size, y0, params)
                    solver_cache[size_key] = ms
                    save_cache(_CACHE_PATH, cache)
                rows.append((case.key, case.key, size, timing_value_or_none(ms)))
            print()
        rows_by_scenario[scenario] = rows
    return rows_by_scenario


def save_csv(rows: list[_Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_key", "solver", "ensemble_size", "solve_time_ms"])
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    rows: list[_Row],
    cases: Sequence[Case],
    gpu_name: str,
    scenario: str,
    output_path: Path,
) -> None:
    configure_latex_plot_style(plt)
    title = (
        f"KenCarp5 scaling — Brusselator (n_grid={_N_GRID}, {scenario}) — {gpu_name}"
    )
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
