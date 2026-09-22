"""Driver for the log-log sweep benchmarks: ensemble-size scaling and ODE dimensionality.

Both sweep one axis under two scenarios, an ``identical`` ensemble
(divergence 0) and a ``divergent`` one (divergence 1), record every case's
solve time at each point, and write a CSV and a log-log plot per scenario. The
two families differ only in the axis: the scaling sweep fixes the system and
varies the ensemble size, the dimensionality sweep fixes the ensemble and
rebuilds the system at each dimension. The axis also fixes the ``results.json``
layout, which is the one each family's existing cache already has.

A script is the residue: a :class:`SweepBenchmark` and ``main(BENCHMARK)``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.benchmark_common import (
    TIMEOUT_ERROR,
    BenchmarkCase,
    configure_latex_plot_style,
    format_cached_timing,
    get_gpu_name,
    jit_solve,
    julia_solve_time_ms,
    label_width,
    load_cache,
    output_paths,
    print_plot_title,
    save_cache,
    time_blocked,
    timeout_cache_entry,
    timing_value_or_none,
)

SCENARIOS = (
    ("identical", 0.0),
    ("divergent", 1.0),
)


@dataclass(frozen=True)
class SweepAxis:
    column: str
    """The CSV column and the name of the swept quantity."""
    label: str
    """The plot's x label."""
    short: str
    """Names the value in the progress line, ``{short}={value}``."""
    nested_cache: bool
    """``gpu -> scenario -> case -> value`` when true, ``gpu -> "scenario_case" -> value`` when false."""


ENSEMBLE_SIZE = SweepAxis("ensemble_size", "Ensemble size", "n", nested_cache=False)
DIMENSION = SweepAxis("dim", "ODE dimension", "dim", nested_cache=True)


class Problem(NamedTuple):
    """One point of the sweep: the callback and the ensemble to integrate."""

    ode_fn: Callable[..., Any]
    y0: np.ndarray
    params: np.ndarray
    julia_system_config: dict[str, Any] | None = None


@dataclass(frozen=True, kw_only=True)
class SweepCase(BenchmarkCase):
    """One curve of a sweep.

    ``mode`` is ``"jax"`` for any solver of the shape
    ``solve_fn(ode_fn, y0, t_span, params, **kwargs)`` -- a modax solver or a
    Diffrax reference -- which is jitted and timed to completion; or
    ``"julia"`` for a DiffEqGPU ``ensemble_backend``, which reports its own
    solve time. ``cleanup`` runs after each timing, for a solver that caches
    per-dimension state between points.
    """

    mode: str = "jax"
    solve_fn: Callable[..., Any] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ensemble_backend: str | None = None
    cleanup: Callable[[], None] | None = None


@dataclass(kw_only=True)
class SweepBenchmark:
    script_dir: Path
    title: str
    """Plot title with a ``{scenario}`` placeholder; ``— {gpu}`` is appended."""
    axis: SweepAxis
    values: tuple[int, ...]
    t_span: Any
    make_problem: Callable[[int, float], Problem]
    """``(value, divergence) -> Problem`` for one point of the sweep."""
    cases: tuple[SweepCase, ...]
    julia_solve: Any = None
    julia_system: str | None = None
    skip: Callable[[SweepCase, int], str | None] | None = None
    """Returns why a case is not run at a value, or ``None`` to run it."""
    legend_loc: str = "best"
    n_runs: int = 1

    @property
    def cache_path(self) -> Path:
        return self.script_dir / "results.json"


class Row(NamedTuple):
    key: str
    label: str
    value: int
    ms: float | None


def _time_case(
    bench: SweepBenchmark, case: SweepCase, value: int, divergence: float
) -> float:
    problem = bench.make_problem(value, divergence)
    if case.mode == "julia":
        return julia_solve_time_ms(
            bench.julia_solve,
            bench.julia_system,
            problem.y0,
            bench.t_span,
            problem.params,
            system_config=problem.julia_system_config,
            ensemble_backend=case.ensemble_backend,
            **case.kwargs,
        )
    if case.mode != "jax":
        raise ValueError(f"unknown sweep case mode: {case.mode!r}")
    assert case.solve_fn is not None
    run = jit_solve(case.solve_fn, problem.ode_fn, bench.t_span, **case.kwargs)
    try:
        ms, _ = time_blocked(lambda: run(problem.y0, problem.params), bench.n_runs)
        return ms
    finally:
        if case.cleanup is not None:
            case.cleanup()


def _collect_timing(prefix: str, run: Callable[[], float]):
    print(f"{prefix} ...", end=" ", flush=True)
    try:
        ms = run()
    except TimeoutError:
        print(TIMEOUT_ERROR)
        return timeout_cache_entry()
    except Exception as exc:
        print(f"FAILED ({exc})")
        return None
    print(f"{ms:.1f} ms")
    return ms


def _case_cache(
    bench: SweepBenchmark, gpu_cache: dict, scenario: str, case: SweepCase
) -> dict:
    if bench.axis.nested_cache:
        return gpu_cache.setdefault(scenario, {}).setdefault(case.key, {})
    return gpu_cache.setdefault(f"{scenario}_{case.key}", {})


def run_benchmarks(
    bench: SweepBenchmark, gpu_name: str, cache: dict, scenario: str, divergence: float
) -> list[Row]:
    gpu_cache = cache.setdefault(gpu_name, {})
    width = label_width(bench.cases)
    value_width = len(str(max(bench.values)))
    rows: list[Row] = []
    for case in bench.cases:
        print(f"\n{case.key}:")
        case_cache = _case_cache(bench, gpu_cache, scenario, case)
        for value in bench.values:
            value_key = str(value)
            prefix = f"  {case.key:<{width}} {bench.axis.short}={value:>{value_width}}"
            if bench.skip is not None and (reason := bench.skip(case, value)):
                print(f"{prefix} ... skipped ({reason})")
                if case_cache.pop(value_key, None) is not None:
                    save_cache(bench.cache_path, cache)
                continue
            if value_key in case_cache:
                ms = case_cache[value_key]
                print(f"{prefix} ... (cached) {format_cached_timing(ms)}")
            else:
                ms = _collect_timing(
                    prefix, lambda: _time_case(bench, case, value, divergence)
                )
                case_cache[value_key] = ms
                save_cache(bench.cache_path, cache)
            rows.append(Row(case.key, case.key, value, timing_value_or_none(ms)))
    return rows


def save_csv(bench: SweepBenchmark, rows: list[Row], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver_key", "solver", bench.axis.column, "solve_time_ms"])
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    bench: SweepBenchmark,
    rows: list[Row],
    gpu_name: str,
    scenario: str,
    output_path: Path,
) -> None:
    configure_latex_plot_style(plt)
    print_plot_title(f"{bench.title.format(scenario=scenario)} — {gpu_name}")
    fig, ax = plt.subplots(figsize=(7, 5))
    for case in bench.cases:
        points = [
            (row.value, row.ms)
            for row in rows
            if row.key == case.key and row.ms is not None
        ]
        if not points:
            continue
        values, times_ms = zip(*points)
        ax.plot(
            values,
            times_ms,
            marker=case.marker,
            color=case.color,
            linestyle=case.linestyle,
            label=case.key,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(bench.axis.label)
    ax.set_ylabel("Solve time (ms)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xticks(bench.values)
    ax.set_xticklabels([str(v) for v in bench.values], rotation=45, ha="right")
    ax.legend(loc=bench.legend_loc)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main(bench: SweepBenchmark) -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(bench.cache_path)
    for scenario, divergence in SCENARIOS:
        print(f"\n=== {scenario} ===")
        rows = run_benchmarks(bench, gpu_name, cache, scenario, divergence)
        print()
        csv_path, plot_path = output_paths(bench.script_dir, gpu_name, scenario)
        save_csv(bench, rows, csv_path)
        plot(bench, rows, gpu_name, scenario, plot_path)
