"""Driver for the sweep benchmarks: ensemble-size scaling, ODE dimensionality, Jacobian density.

Each sweeps one axis over an ensemble of *identical* trajectories (the
``make_scenario(..., divergence=0.0)`` ensemble), records every case's solve
time at each point, and writes one CSV and one plot named after the GPU. The
families differ only in the axis: the scaling sweep fixes the system and varies
the ensemble size, the dimensionality sweep fixes the ensemble and rebuilds the
system at each dimension, the density sweep fixes both and varies the
coupling. How a solver fares when the trajectories in a warp *diverge* is the
divergence benchmarks' subject (``_divergence.py``) and is not swept here.

Every point runs in a child process under
:data:`~benchmarks.benchmark_common.CASE_TIMEOUT_SECONDS`, compilation
included; a point that overruns is cached as a timeout and left off the plot.

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
    BenchmarkCase,
    configure_latex_plot_style,
    format_cached_timing,
    get_gpu_name,
    jit_solve,
    jit_value_and_grad,
    julia_solve_time_ms,
    label_width,
    load_cache,
    output_paths,
    print_plot_title,
    run_jobs,
    save_cache,
    time_blocked,
    timing_value_or_none,
)
from reference.solvers.python.julia_common import _julia_executable

IDENTICAL_DIVERGENCE = 0.0
"""The ``divergence`` every sweep passes to ``make_scenario``: identical trajectories."""


@dataclass(frozen=True)
class SweepAxis:
    column: str
    """The CSV column and the name of the swept quantity."""
    label: str
    """The plot's x label."""
    short: str
    """Names the value in the progress line, ``{short}={value}``."""
    position: Callable[[int], float] | None = None
    """Maps a swept value to its x coordinate; ``None`` plots the value itself."""
    position_column: str | None = None
    """An extra CSV column holding ``position(value)``, when there is one."""
    log_x: bool = True


ENSEMBLE_SIZE = SweepAxis("ensemble_size", "Ensemble size", "n")
DIMENSION = SweepAxis("dim", "ODE dimension", "dim")


class Problem(NamedTuple):
    """One point of the sweep: the callback and the ensemble to integrate.

    ``sparsity`` is the Jacobian pattern a ``sparse`` case passes to the
    solver; ``julia_y0`` replaces ``y0`` for the Julia cases when their system
    integrates a different state, such as one augmented with sensitivities.
    """

    ode_fn: Callable[..., Any]
    y0: np.ndarray
    params: np.ndarray
    julia_system_config: dict[str, Any] | None = None
    sparsity: Any = None
    julia_y0: np.ndarray | None = None


@dataclass(frozen=True, kw_only=True)
class SweepCase(BenchmarkCase):
    """One curve of a sweep.

    ``mode`` is ``"jax"`` for any solver of the shape
    ``solve_fn(ode_fn, y0, t_span, params, **kwargs)`` -- a modax solver or a
    Diffrax reference -- which is jitted and timed to completion;
    ``"jax_grad"`` for the same solver timed under ``jax.value_and_grad`` of
    the summed final state with respect to ``params``; or ``"julia"`` for a
    DiffEqGPU ``ensemble_backend``, which reports its own solve time.
    ``sparse`` hands the problem's Jacobian pattern to the solver as
    ``sparsity=``. ``cleanup`` runs after each timing, for a solver that caches
    per-dimension state between points.
    """

    mode: str = "jax"
    solve_fn: Callable[..., Any] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    ensemble_backend: str | None = None
    sparse: bool = False
    cleanup: Callable[[], None] | None = None


@dataclass(kw_only=True)
class SweepBenchmark:
    script_dir: Path
    title: str
    """Plot title; ``— {gpu}`` is appended."""
    axis: SweepAxis
    values: tuple[int, ...]
    t_span: Any
    make_problem: Callable[[int], Problem]
    """``value -> Problem`` for one point of the sweep."""
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

    @property
    def script_path(self) -> Path:
        return self.script_dir / "main.py"

    def case(self, key: str) -> SweepCase:
        return next(case for case in self.cases if case.key == key)


class Row(NamedTuple):
    key: str
    value: int
    position: float
    ms: float | None


def _position(bench: SweepBenchmark, value: int) -> float:
    if bench.axis.position is None:
        return float(value)
    return float(bench.axis.position(value))


def prepare(bench: SweepBenchmark, job: dict) -> None:
    """Before the clock starts on a Julia job, run the toolchain check it would pay."""
    if bench.case(job["case"]).mode == "julia":
        _julia_executable()


def measure(bench: SweepBenchmark, job: dict) -> float:
    """Solve time in ms of ``job["case"]`` at ``job["value"]``; runs in the worker."""
    case = bench.case(job["case"])
    value = int(job["value"])
    problem = bench.make_problem(value)
    if case.mode == "julia":
        y0 = problem.y0 if problem.julia_y0 is None else problem.julia_y0
        return julia_solve_time_ms(
            bench.julia_solve,
            bench.julia_system,
            y0,
            bench.t_span,
            problem.params,
            system_config=problem.julia_system_config,
            ensemble_backend=case.ensemble_backend,
            **case.kwargs,
        )
    if case.mode not in ("jax", "jax_grad"):
        raise ValueError(f"unknown sweep case mode: {case.mode!r}")
    assert case.solve_fn is not None
    kwargs = dict(case.kwargs)
    if case.sparse:
        if problem.sparsity is None:
            raise ValueError(f"{case.key} wants a sparsity pattern the problem lacks")
        kwargs["sparsity"] = problem.sparsity
    make_run = jit_value_and_grad if case.mode == "jax_grad" else jit_solve
    run = make_run(case.solve_fn, problem.ode_fn, bench.t_span, **kwargs)
    try:
        ms, _ = time_blocked(lambda: run(problem.y0, problem.params), bench.n_runs)
        return ms
    finally:
        if case.cleanup is not None:
            case.cleanup()


def run_benchmarks(bench: SweepBenchmark, gpu_name: str, cache: dict) -> list[Row]:
    # results.json layout: gpu -> case key -> str(value) -> ms | timeout | null.
    gpu_cache = cache.setdefault(gpu_name, {})
    width = label_width(bench.cases)
    value_width = len(str(max(bench.values)))
    rows: list[Row] = []

    def prefix(case: SweepCase, value: int) -> str:
        return f"  {case.key:<{width}} {bench.axis.short}={value:>{value_width}}"

    for case in bench.cases:
        print(f"\n{case.key}:")
        case_cache = gpu_cache.setdefault(case.key, {})
        jobs: list[dict] = []
        for value in bench.values:
            value_key = str(value)
            if bench.skip is not None and (reason := bench.skip(case, value)):
                print(f"{prefix(case, value)} ... skipped ({reason})")
                if case_cache.pop(value_key, None) is not None:
                    save_cache(bench.cache_path, cache)
            elif value_key in case_cache:
                cached = format_cached_timing(case_cache[value_key])
                print(f"{prefix(case, value)} ... (cached) {cached}")
            else:
                jobs.append({"case": case.key, "value": value})

        def on_started(job: dict) -> None:
            print(f"{prefix(case, job['value'])} ...", end=" ", flush=True)

        def on_result(job: dict, status: str, result: Any) -> None:
            if status == "ok":
                print(f"{result:.1f} ms", flush=True)
                case_cache[str(job["value"])] = result
            elif status == "timeout":
                print(format_cached_timing(result), flush=True)
                case_cache[str(job["value"])] = result
            else:
                print(f"FAILED ({result})", flush=True)
                case_cache[str(job["value"])] = None
            save_cache(bench.cache_path, cache)

        run_jobs(bench.script_path, jobs, on_started=on_started, on_result=on_result)

        for value in bench.values:
            if str(value) in case_cache:
                ms = timing_value_or_none(case_cache[str(value)])
                rows.append(Row(case.key, value, _position(bench, value), ms))
    return rows


def save_csv(bench: SweepBenchmark, rows: list[Row], path: Path) -> None:
    axis = bench.axis
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["solver_key", "solver", axis.column]
        if axis.position_column is not None:
            header.append(axis.position_column)
        writer.writerow(header + ["solve_time_ms"])
        for row in rows:
            record: list[Any] = [row.key, row.key, row.value]
            if axis.position_column is not None:
                record.append(row.position)
            writer.writerow(record + [row.ms])
    print(f"Results saved to {path}")


def plot(bench: SweepBenchmark, rows: list[Row], gpu_name: str, path: Path) -> None:
    configure_latex_plot_style(plt)
    print_plot_title(f"{bench.title} — {gpu_name}")
    fig, ax = plt.subplots(figsize=(7, 5))
    for case in bench.cases:
        points = [
            (row.position, row.ms)
            for row in rows
            if row.key == case.key and row.ms is not None
        ]
        if not points:
            continue
        positions, times_ms = zip(*points)
        ax.plot(
            positions,
            times_ms,
            marker=case.marker,
            color=case.color,
            linestyle=case.linestyle,
            label=case.key,
        )
    if bench.axis.log_x:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(bench.axis.label)
    ax.set_ylabel("Solve time (ms)")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    if bench.axis.position is None:
        ax.set_xticks(bench.values)
        ax.set_xticklabels([str(v) for v in bench.values], rotation=45, ha="right")
    ax.legend(loc=bench.legend_loc)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path}")


def main(bench: SweepBenchmark) -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}\n")

    cache = load_cache(bench.cache_path)
    rows = run_benchmarks(bench, gpu_name, cache)
    print()
    csv_path, plot_path = output_paths(bench.script_dir, gpu_name)
    save_csv(bench, rows, csv_path)
    plot(bench, rows, gpu_name, plot_path)
