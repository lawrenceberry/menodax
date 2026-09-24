"""Shared subprocess utilities for Julia GPU reference solvers.

Each public solver module (``julia_tsit5``, ``julia_rodas5P``) is built by
:func:`make_julia_solver`, which binds the Julia solver name and returns the
``(solve, solve_with_timing)`` pair the scripts and tests consume. A missing or
broken Julia toolchain raises :class:`JuliaUnavailableError`; only the
test-side :func:`benchmark_julia_solver` turns that into a pytest skip.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REFERENCE_SOLVERS_DIR = Path(__file__).resolve().parents[1]
_JULIA_DIR = _REFERENCE_SOLVERS_DIR / "julia"
_JULIA_RUNNER = _JULIA_DIR / "run_solver.jl"
_JULIA_PROJECT_FLAG = f"--project={_JULIA_DIR}"
JULIA_SOLVE_TIMEOUT_SECONDS = 30.0

JULIA_SOLVERS = ("tsit5", "rodas5P")
JULIA_ENSEMBLE_BACKENDS = ("EnsembleGPUArray", "EnsembleGPUKernel")


class JuliaUnavailableError(RuntimeError):
    """The Julia reference solver cannot run in this environment."""


@dataclass(frozen=True)
class JuliaSolveResult:
    ys: np.ndarray
    solve_time_s: float
    total_wall_time_s: float
    payload: dict


def julia_backend_id(ensemble_backend: str) -> str:
    """Return a stable pytest id for a Julia GPU backend."""
    if ensemble_backend == "EnsembleGPUArray":
        return "gpu-array"
    if ensemble_backend == "EnsembleGPUKernel":
        return "gpu-kernel"
    return ensemble_backend.lower()


def benchmark_julia_solver(benchmark, solve, *solve_args, **solve_kwargs):
    """Record Julia solve-only timing in pytest-benchmark and return NumPy output.

    This is the one test-facing entry point: a :class:`JuliaUnavailableError`
    becomes a pytest skip here and nowhere else.
    """
    import pytest  # noqa: PLC0415  (test-only dependency)

    solve_with_timing = getattr(solve, "_julia_solve_with_timing", None)
    if solve_with_timing is None:
        raise TypeError(
            "benchmark_julia_solver requires a Julia direct solve function with "
            "a _julia_solve_with_timing attribute"
        )

    if benchmark._mode:
        benchmark.has_error = True
        raise RuntimeError(f"Benchmark fixture already used in mode {benchmark._mode}")

    try:
        benchmark._mode = "benchmark.julia_solver(...)"
        if benchmark.disabled:
            return np.asarray(solve(*solve_args, **solve_kwargs))

        result = solve_with_timing(*solve_args, **solve_kwargs)
        stats = benchmark._make_stats(1)
        stats.update(result.solve_time_s)
        benchmark.extra_info["julia_solve_time_s"] = result.solve_time_s
        benchmark.extra_info["julia_total_wall_time_s"] = result.total_wall_time_s
        benchmark.extra_info["julia_solver_payload"] = result.payload
        return np.asarray(result.ys)
    except JuliaUnavailableError as exc:
        pytest.skip(str(exc))
    except Exception:
        benchmark.has_error = True
        raise


def make_julia_solver(solver_name: str):
    """Build the ``(solve, solve_with_timing)`` pair for one Julia solver.

    ``solve`` returns the ensemble solution; ``solve_with_timing`` returns a
    :class:`JuliaSolveResult` with the solve-only timing the benchmark scripts
    read through ``solve._julia_solve_with_timing``. ``timeout`` caps the
    Julia subprocess in seconds, startup and compilation included, and
    defaults to :data:`JULIA_SOLVE_TIMEOUT_SECONDS`; exceeding it raises
    :class:`TimeoutError`.
    """
    if solver_name not in JULIA_SOLVERS:
        raise ValueError(
            f"Unknown Julia reference solver '{solver_name}'; "
            f"expected one of {JULIA_SOLVERS}"
        )

    def solve_with_timing(
        system_name,
        y0,
        t_span,
        params,
        *,
        system_config=None,
        ensemble_backend="EnsembleGPUArray",
        rtol=1e-8,
        atol=1e-10,
        first_step=None,
        max_steps=100000,
        timeout=None,
    ):
        if ensemble_backend not in JULIA_ENSEMBLE_BACKENDS:
            raise JuliaUnavailableError(
                f"Julia reference solver '{solver_name}' does not support "
                f"{ensemble_backend}; expected one of {JULIA_ENSEMBLE_BACKENDS}"
            )
        return _run_julia_solver(
            solver_name,
            system_name,
            ensemble_backend,
            {} if system_config is None else dict(system_config),
            y0,
            t_span,
            params,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            timeout=timeout,
        )

    def solve(
        system_name,
        y0,
        t_span,
        params,
        *,
        system_config=None,
        ensemble_backend="EnsembleGPUArray",
        rtol=1e-8,
        atol=1e-10,
        first_step=None,
        max_steps=100000,
        timeout=None,
    ):
        return solve_with_timing(
            system_name,
            y0,
            t_span,
            params,
            system_config=system_config,
            ensemble_backend=ensemble_backend,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            timeout=timeout,
        ).ys

    solve.__doc__ = f"Solve an ensemble with Julia {solver_name}."
    solve_with_timing.__doc__ = (
        f"Solve an ensemble with Julia {solver_name} and return timing metadata."
    )
    solve._julia_solve_with_timing = solve_with_timing
    return solve, solve_with_timing


def _julia_subprocess_env():
    return os.environ.copy()


@functools.lru_cache(maxsize=1)
def _julia_executable() -> str:
    """Return the Julia executable, or raise :class:`JuliaUnavailableError`.

    The check is cached: it runs a Julia process that loads every package the
    runner needs, which is too slow to repeat per solve.
    """
    julia_exe = shutil.which("julia")
    if julia_exe is None:
        raise JuliaUnavailableError("Julia executable not found on PATH")
    if not _JULIA_RUNNER.exists():
        raise JuliaUnavailableError(
            f"Julia reference runner missing at {_JULIA_RUNNER}"
        )

    cmd = [
        julia_exe,
        _JULIA_PROJECT_FLAG,
        "-e",
        (
            "using JSON, CUDA, DiffEqGPU, OrdinaryDiffEq, SciMLBase, StaticArrays; "
            "CUDA.functional(true); "
            'println("ok")'
        ),
    ]
    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=_JULIA_DIR,
        env=_julia_subprocess_env(),
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        if "Pkg.instantiate" in message or "Package" in message:
            raise JuliaUnavailableError(
                "Julia reference solver environment is not ready. "
                f"Activate {_JULIA_DIR} and run `Pkg.instantiate()`. "
                f"Julia said: {message}"
            )
        raise JuliaUnavailableError(
            f"Julia reference solver environment check failed: {message}"
        )
    return julia_exe


def _run_julia_solver(
    solver_name,
    system_name,
    ensemble_backend,
    system_config,
    y0,
    t_span,
    params,
    *,
    rtol,
    atol,
    first_step,
    max_steps,
    timeout=None,
):
    julia_exe = _julia_executable()
    if timeout is None:
        timeout = JULIA_SOLVE_TIMEOUT_SECONDS

    y0_arr = np.ascontiguousarray(np.asarray(y0, dtype=np.float64))
    t_span_arr = np.ascontiguousarray(np.asarray(t_span, dtype=np.float64))
    params_arr = np.ascontiguousarray(np.asarray(params, dtype=np.float64))

    with tempfile.TemporaryDirectory(prefix="julia-ref-") as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        system_config_path = tmp_dir / "system_config.json"
        y0_bin = tmp_dir / "y0.bin"
        y0_meta = tmp_dir / "y0.json"
        params_bin = tmp_dir / "params.bin"
        params_meta = tmp_dir / "params.json"
        t_span_bin = tmp_dir / "t_span.bin"
        t_span_meta = tmp_dir / "t_span.json"
        ys_bin = tmp_dir / "ys.bin"
        ys_meta = tmp_dir / "ys.json"

        system_config_path.write_text(
            json.dumps(system_config, sort_keys=True),
            encoding="utf-8",
        )
        _write_c_order_array(y0_bin, y0_meta, y0_arr)
        _write_c_order_array(params_bin, params_meta, params_arr)
        _write_c_order_array(t_span_bin, t_span_meta, t_span_arr)

        cmd = [
            julia_exe,
            _JULIA_PROJECT_FLAG,
            str(_JULIA_RUNNER),
            solver_name,
            system_name,
            ensemble_backend,
            str(float(rtol)),
            str(float(atol)),
            "none" if first_step is None else str(float(first_step)),
            str(int(max_steps)),
            str(system_config_path),
            str(y0_bin),
            str(y0_meta),
            str(params_bin),
            str(params_meta),
            str(t_span_bin),
            str(t_span_meta),
            str(ys_bin),
            str(ys_meta),
        ]
        wall_start = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=_JULIA_DIR,
                env=_julia_subprocess_env(),
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"Julia reference solve exceeded {timeout:g}s timeout.\n"
                f"Command: {' '.join(cmd)}"
            ) from exc
        wall_end = time.perf_counter()
        if completed.returncode != 0:
            raise RuntimeError(
                "Julia reference solve failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        payload = json.loads(completed.stdout or "{}")
        if payload.get("status") != "ok":
            raise RuntimeError(
                "Julia reference solve returned a non-ok status: "
                f"{json.dumps(payload, indent=2, sort_keys=True)}"
            )
        return JuliaSolveResult(
            ys=_read_c_order_array(ys_bin, ys_meta),
            solve_time_s=float(payload["solve_time_s"]),
            total_wall_time_s=wall_end - wall_start,
            payload=payload,
        )


def _write_c_order_array(bin_path: Path, meta_path: Path, arr: np.ndarray) -> None:
    arr_c = np.ascontiguousarray(arr, dtype=np.float64)
    bin_path.write_bytes(arr_c.tobytes(order="C"))
    meta_path.write_text(
        json.dumps(
            {
                "dtype": "float64",
                "shape": list(arr_c.shape),
                "order": "C",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _read_c_order_array(bin_path: Path, meta_path: Path) -> np.ndarray:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    shape = tuple(int(dim) for dim in meta["shape"])
    arr = np.frombuffer(bin_path.read_bytes(), dtype=np.float64)
    return arr.reshape(shape, order="C")
