"""Driver for the divergence benchmarks.

A divergence benchmark holds the ensemble size fixed and sweeps the
``make_scenario(..., divergence=...)`` knob. At each value it records every
case's solve time together with the distribution of attempted (accepted plus
rejected) steps across the ensemble, taken from the modax solver's
``return_stats=True`` output on the same data. The plot is solve time per mean
attempted step against the coefficient of variation of the attempted steps:
how much a solver's throughput degrades as the trajectories in a warp diverge.

Every point runs in a child process under
:data:`~benchmarks.benchmark_common.CASE_TIMEOUT_SECONDS`, compilation
included; a point that overruns is cached as a timeout and left off the plot.

A script is the residue: a :class:`DivergenceBenchmark` and ``main(BENCHMARK)``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.benchmark_common import (
    TIMEOUT_ERROR,
    BenchmarkCase,
    configure_latex_plot_style,
    get_gpu_name,
    is_timeout,
    jit_solve,
    julia_solve_time_ms,
    label_width,
    load_cache,
    output_paths,
    print_plot_title,
    run_jobs,
    save_cache,
    time_blocked,
)
from reference.solvers.python.julia_common import _julia_executable

_STATS_FIELDS = (
    "mean_steps",
    "step_std",
    "step_cv",
    "step_variance",
    "min_steps",
    "max_steps",
    "rejected_steps_mean",
)


@dataclass(frozen=True, kw_only=True)
class DivergenceCase(BenchmarkCase):
    """One series of a divergence benchmark.

    ``mode`` is ``"stats"`` for the benchmark's own modax solver, timed with
    ``return_stats=True`` so the step distribution comes from the timed run
    itself; ``"julia"`` for a DiffEqGPU ``ensemble_backend``; or ``"timing"``
    for another JAX solver passed as ``solve_fn`` (``kwargs`` overrides the
    benchmark's solver settings for it). The latter two report no step counts,
    so the modax solver supplies them, from its cached row for the same
    divergence when it has one and otherwise from a fresh solve.

    ``sort_by_steps`` pre-sorts the ensemble by attempted steps, most first, so
    that neighbouring threads do similar work. ``max_divergence`` skips the
    larger sweep values for a case that cannot finish them.
    """

    mode: str = "stats"
    ensemble_backend: str | None = None
    sort_by_steps: bool = False
    max_divergence: float | None = None
    solve_fn: Callable[..., Any] | None = None
    kwargs: dict[str, Any] | None = None


@dataclass(kw_only=True)
class DivergenceBenchmark:
    script_dir: Path
    system: str
    """Printed as ``System: {dim}D {system}, {n_traj} trajectories``."""
    title: str
    """Plot title; ``— {n_traj} trajectories — {gpu}`` is appended."""
    solve: Callable[..., Any]
    """The modax solver, ``modax.tsit5.solve`` or ``modax.rodas5P.solve``."""
    ode_fn: Callable[..., Any]
    t_span: Any
    dim: int
    n_traj: int
    divergences: tuple[float, ...]
    solver_kwargs: dict[str, Any]
    make_data: Callable[[float], tuple[np.ndarray, np.ndarray]]
    """``divergence -> (y0, params)`` for the whole ensemble."""
    cases: tuple[DivergenceCase, ...]
    julia_solve: Any = None
    julia_system: str | None = None
    julia_system_config: dict[str, Any] | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)
    """Constant columns written after ``dim`` in every row, e.g. ``n_osc``."""
    legend_loc: str = "best"
    n_runs: int = 1

    @property
    def cache_path(self) -> Path:
        return self.script_dir / "results.json"

    @property
    def script_path(self) -> Path:
        return self.script_dir / "main.py"

    def case(self, key: str) -> DivergenceCase:
        return next(case for case in self.cases if case.key == key)

    @property
    def csv_fields(self) -> tuple[str, ...]:
        return (
            "gpu",
            "solver_key",
            "solver",
            "divergence",
            "dim",
            *self.extra_fields,
            "ensemble_size",
            "solve_time_ms",
            "mean_steps",
            "step_std",
            "step_cv",
            "normalized_solve_time_ms_per_step",
            "step_variance",
            "min_steps",
            "max_steps",
            "rejected_steps_mean",
        )

    @cached_property
    def stats_solve(self) -> Callable[[Any, Any], tuple[Any, dict]]:
        """The modax solve with ``return_stats=True``, compiled once."""
        return jit_solve(
            self.solve,
            self.ode_fn,
            self.t_span,
            return_stats=True,
            **self.solver_kwargs,
        )

    @property
    def stats_donor_key(self) -> str | None:
        """The case whose cached rows lend step statistics to cases without any."""
        for case in self.cases:
            if case.mode == "stats" and not case.sort_by_steps:
                return case.key
        return None


def _attempted_steps(stats: dict) -> tuple[np.ndarray, np.ndarray]:
    accepted = np.asarray(jax.device_get(stats["accepted_steps"]))
    rejected = np.asarray(jax.device_get(stats["rejected_steps"]))
    return accepted + rejected, rejected


def _summarize_stats(stats: dict) -> dict[str, float | int]:
    attempted, rejected = _attempted_steps(stats)
    mean_steps = float(np.mean(attempted))
    ddof = 1 if attempted.size > 1 else 0
    step_variance = float(np.var(attempted, ddof=ddof))
    step_std = float(np.sqrt(step_variance))
    return {
        "mean_steps": mean_steps,
        "step_std": step_std,
        "step_cv": float(step_std / mean_steps) if mean_steps else 0.0,
        "step_variance": step_variance,
        "min_steps": int(np.min(attempted)),
        "max_steps": int(np.max(attempted)),
        "rejected_steps_mean": float(np.mean(rejected)),
    }


def _stats_from_row(row: dict) -> dict[str, float | int]:
    return {name: row[name] for name in _STATS_FIELDS}


def _sort_by_attempted_steps(
    bench: DivergenceBenchmark, y0: np.ndarray, params: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Reorder the ensemble by attempted steps, most first."""
    _, stats = bench.stats_solve(y0, params)
    attempted, _ = _attempted_steps(stats)
    order = np.argsort(-attempted, kind="stable")
    return y0[order], params[order], _summarize_stats(stats)


def _time_solve(
    bench: DivergenceBenchmark, case: DivergenceCase, y0: np.ndarray, params: np.ndarray
) -> tuple[float, dict | None]:
    """Solve time in ms and, when the case reports them, the per-trajectory stats."""
    if case.mode == "julia":
        # An identical ensemble is sent as one initial state.
        if y0.ndim == 2 and np.all(y0 == y0[0]):
            y0 = y0[0]
        ms = julia_solve_time_ms(
            bench.julia_solve,
            bench.julia_system,
            y0,
            bench.t_span,
            params,
            system_config=bench.julia_system_config,
            ensemble_backend=case.ensemble_backend,
            **bench.solver_kwargs,
        )
        return ms, None
    if case.mode == "timing":
        assert case.solve_fn is not None
        kwargs = bench.solver_kwargs if case.kwargs is None else case.kwargs
        run = jit_solve(case.solve_fn, bench.ode_fn, bench.t_span, **kwargs)
        y0_j, params_j = jnp.asarray(y0), jnp.asarray(params)
        ms, _ = time_blocked(lambda: run(y0_j, params_j), bench.n_runs)
        return ms, None
    if case.mode == "stats":
        ms, (_, stats) = time_blocked(
            lambda: bench.stats_solve(y0, params), bench.n_runs
        )
        return ms, stats
    raise ValueError(f"unknown divergence case mode: {case.mode!r}")


def _measure_row(
    bench: DivergenceBenchmark,
    gpu_name: str,
    case: DivergenceCase,
    divergence: float,
    summary: dict | None,
) -> dict:
    y0, params = bench.make_data(divergence)
    if case.sort_by_steps:
        y0, params, sorted_summary = _sort_by_attempted_steps(bench, y0, params)
        if summary is None:
            summary = sorted_summary
    ms, stats = _time_solve(bench, case, y0, params)
    if stats is not None:
        summary = _summarize_stats(stats)
    elif summary is None:
        _, stats = bench.stats_solve(y0, params)
        summary = _summarize_stats(stats)

    normalized = ms / summary["mean_steps"] if summary["mean_steps"] else 0.0
    return {
        "gpu": gpu_name,
        "solver_key": case.key,
        "solver": case.key,
        "divergence": float(divergence),
        "dim": bench.dim,
        **bench.extra_fields,
        "ensemble_size": bench.n_traj,
        "solve_time_ms": float(ms),
        **summary,
        "normalized_solve_time_ms_per_step": float(normalized),
    }


def _format_row(row: dict) -> str:
    return (
        f"{row['solve_time_ms']:.1f} ms, "
        f"mean_steps={row['mean_steps']:.1f}, "
        f"step_cv={row['step_cv']:.3f}, "
        f"norm={row['normalized_solve_time_ms_per_step']:.4f} ms/step"
    )


def _is_complete_row(bench: DivergenceBenchmark, value) -> bool:
    return isinstance(value, dict) and all(name in value for name in bench.csv_fields)


def prepare(bench: DivergenceBenchmark, job: dict) -> None:
    """Before the clock starts on a Julia job, run the toolchain check it would pay."""
    if bench.case(job["case"]).mode == "julia":
        _julia_executable()


def measure(bench: DivergenceBenchmark, job: dict) -> dict:
    """The row for ``job["case"]`` at ``job["divergence"]``; runs in the worker.

    ``job["summary"]`` is the step statistics lent by the donor case's cached
    row, or ``None`` when the case has to measure its own.
    """
    return _measure_row(
        bench,
        job["gpu"],
        bench.case(job["case"]),
        float(job["divergence"]),
        job["summary"],
    )


def run_benchmarks(
    bench: DivergenceBenchmark, gpu_name: str, cache: dict
) -> list[dict]:
    # results.json layout: gpu -> case key -> f"{divergence:.6g}" -> row.
    gpu_cache = cache.setdefault(gpu_name, {})
    width = label_width(bench.cases)
    donor_key = bench.stats_donor_key
    rows: list[dict] = []

    def prefix(case: DivergenceCase, divergence: float) -> str:
        return f"  {case.key:<{width}} divergence={divergence:>4.2f} ..."

    for case in bench.cases:
        print(f"\n{case.key}:")
        case_cache = gpu_cache.setdefault(case.key, {})
        jobs: list[dict] = []
        for divergence in bench.divergences:
            divergence_key = f"{divergence:.6g}"
            if case.max_divergence is not None and divergence > case.max_divergence:
                print(
                    f"{prefix(case, divergence)} SKIPPED "
                    f"(divergence > {case.max_divergence:g})",
                    flush=True,
                )
                case_cache.setdefault(divergence_key, None)
                continue
            cached = case_cache.get(divergence_key)
            if is_timeout(cached) or _is_complete_row(bench, cached):
                text = TIMEOUT_ERROR if is_timeout(cached) else _format_row(cached)
                print(f"{prefix(case, divergence)} (cached) {text}", flush=True)
                continue
            donor_row = None
            if donor_key is not None:
                donor_row = gpu_cache.get(donor_key, {}).get(divergence_key)
            summary = (
                _stats_from_row(donor_row)
                if _is_complete_row(bench, donor_row)
                else None
            )
            jobs.append(
                {
                    "case": case.key,
                    "divergence": divergence,
                    "summary": summary,
                    "gpu": gpu_name,
                }
            )

        def on_started(job: dict) -> None:
            print(prefix(case, job["divergence"]), end=" ", flush=True)

        def on_result(job: dict, status: str, result: Any) -> None:
            key = f"{job['divergence']:.6g}"
            if status == "ok":
                print(_format_row(result), flush=True)
                case_cache[key] = result
            elif status == "timeout":
                print(TIMEOUT_ERROR, flush=True)
                case_cache[key] = result
            else:
                print(f"FAILED ({result})", flush=True)
                case_cache[key] = None
            save_cache(bench.cache_path, cache)

        run_jobs(bench.script_path, jobs, on_started=on_started, on_result=on_result)

        for divergence in bench.divergences:
            row = case_cache.get(f"{divergence:.6g}")
            if _is_complete_row(bench, row):
                rows.append(row)
    return rows


def save_csv(bench: DivergenceBenchmark, rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bench.csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {path}")


def plot(
    bench: DivergenceBenchmark, rows: list[dict], gpu_name: str, output_path: Path
) -> None:
    configure_latex_plot_style(plt)
    print_plot_title(f"{bench.title} — {bench.n_traj} trajectories — {gpu_name}")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for case in bench.cases:
        case_rows = sorted(
            (row for row in rows if row["solver_key"] == case.key),
            key=lambda row: row["divergence"],
        )
        if not case_rows:
            continue
        ax.scatter(
            [row["step_cv"] for row in case_rows],
            [row["normalized_solve_time_ms_per_step"] for row in case_rows],
            color=case.color,
            marker=case.marker,
            s=42,
            label=case.key,
        )

    ax.set_xlabel("Normalized standard deviation of attempted steps")
    ax.set_ylabel("Solve time / mean attempted steps (ms)")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc=bench.legend_loc)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")


def main(bench: DivergenceBenchmark) -> None:
    gpu_name = get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"System: {bench.dim}D {bench.system}, {bench.n_traj} trajectories\n")

    cache = load_cache(bench.cache_path)
    rows = run_benchmarks(bench, gpu_name, cache)
    csv_path, plot_path = output_paths(bench.script_dir, gpu_name)
    save_csv(bench, rows, csv_path)
    plot(bench, rows, gpu_name, plot_path)
