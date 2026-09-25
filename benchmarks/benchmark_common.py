"""Helpers shared by the two benchmark drivers, ``_divergence.py`` and ``_sweep.py``."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

# JAX preallocates 75% of the card the moment its GPU backend initialises, but
# these benchmarks share the card with two allocators it knows nothing about:
# the numba-cuda kernels behind the modax solvers, and the Julia subprocesses
# the DiffEqGPU references run in. Starved of the rest, the wide Rodas5P
# kernels fail to launch at all -- `cuLaunchKernel failed with CUDA driver
# error 2` -- and the larger Julia ensembles run out of GPU memory. How much
# room the kernels need varies with the problem, so a fixed smaller fraction
# does not work either: at dim 96 the identical VDP ensemble launches with half
# the card, while the divergent one needs roughly 85% of it free. Allocating on
# demand instead lets JAX take what it uses and no more, which is what the
# kernels and the subprocesses then have. It also recovers points the fixed
# preallocation lost entirely: dim 128 in fp32 and dim 96 in fp64 both run.
# Set the variable before running to override.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402  -- must follow the memory setting above
import jax.numpy as jnp  # noqa: E402

T = TypeVar("T")

REPO_ROOT = Path(__file__).resolve().parents[1]

CASE_TIMEOUT_SECONDS = 120.0
"""The wall-clock cap on one case at one point of a sweep, compilation included.

A point that has not compiled and solved within it is recorded as a timeout
and left off the plot.
"""

TIMEOUT_ERROR = "exceeded timeout"
_TIMEOUT_STATUS = "timeout"


def configure_latex_plot_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        }
    )


def print_plot_title(title: str) -> None:
    print(f"Plot title: {title}")


# One colour per solver, the same in every plot; within a solver the marker and
# the line style tell its cases apart (LU precision, sparsity, sorting, backend).
MODAX_COLOR = "#f0a202"
JULIA_COLOR = "#9b59b6"
DIFFRAX_COLOR = "#1f77b4"
TORCHDIFFEQ_COLOR = "#d62728"


@dataclass(frozen=True, kw_only=True)
class BenchmarkCase:
    key: str
    color: str
    marker: str
    linestyle: str = "-"


def label_width(cases: Sequence[BenchmarkCase]) -> int:
    """Column width that fits every case key in the progress output."""
    return max(len(case.key) for case in cases)


def timeout_cache_entry() -> dict[str, str]:
    return {"status": _TIMEOUT_STATUS, "error": TIMEOUT_ERROR}


def is_timeout(value) -> bool:
    return (
        isinstance(value, dict)
        and value.get("status") == _TIMEOUT_STATUS
        and value.get("error") == TIMEOUT_ERROR
    )


def format_cached_timing(value) -> str:
    if is_timeout(value):
        return TIMEOUT_ERROR
    if value is None:
        return "FAILED"
    return f"{value:.1f} ms"


def timing_value_or_none(value) -> float | None:
    if is_timeout(value) or value is None:
        return None
    return float(value)


def get_gpu_name() -> str:
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
            )
            .strip()
            .splitlines()[0]
            .strip()
        )
        if out:
            return out
    except Exception:
        pass
    try:
        devices = jax.devices("gpu")
        if devices:
            return devices[0].device_kind
    except Exception:
        pass
    return "unknown_GPU"


def gpu_slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache, indent=2))


def output_paths(script_dir: Path, gpu_name: str) -> tuple[Path, Path]:
    """The CSV and plot paths for one GPU."""
    stem = gpu_slug(gpu_name)
    return script_dir / f"results-{stem}.csv", script_dir / f"plot-{stem}.png"


def time_blocked(run: Callable[[], T], n_runs: int) -> tuple[float, T]:
    """Mean wall time of ``run`` in ms over ``n_runs``, after one warm-up call.

    Blocks on the result each time, so an asynchronous JAX dispatch is timed to
    completion. Returns the last result too.
    """
    result = run()
    jax.block_until_ready(result)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        result = run()
        jax.block_until_ready(result)
    return (time.perf_counter() - t0) / n_runs * 1000, result


def jit_solve(
    solve_fn: Callable[..., T], ode_fn: Callable, t_span: Any, **kwargs: Any
) -> Callable[[Any, Any], T]:
    """``solve_fn(ode_fn, y0, t_span, params, **kwargs)`` under ``jax.jit``.

    Only ``y0`` and ``params`` are traced; the callback, the save times and the
    solver settings are closed over. The first call compiles, which is why the
    timers warm up before they measure.
    """

    @jax.jit
    def run(y0, params):
        return solve_fn(ode_fn, y0, t_span, params, **kwargs)

    return run


def jit_value_and_grad(
    solve_fn: Callable[..., Any], ode_fn: Callable, t_span: Any, **kwargs: Any
) -> Callable[[Any, Any], tuple[Any, Any]]:
    """``jax.value_and_grad`` of the summed final state with respect to ``params``.

    The loss is the sum of every trajectory's final state, so its gradient is
    the full parameter sensitivity of the ensemble -- one column per parameter
    per trajectory -- and nothing about the loss can make the derivative
    cheaper than the sensitivities themselves. Jitted as :func:`jit_solve` is.
    """

    def loss(params, y0):
        return jnp.sum(solve_fn(ode_fn, y0, t_span, params, **kwargs)[:, -1, :])

    value_and_grad = jax.value_and_grad(loss)

    @jax.jit
    def run(y0, params):
        return value_and_grad(params, y0)

    return run


def julia_solve_time_ms(
    solve: Any,
    system_name: str,
    y0: Any,
    t_span: Any,
    params: Any,
    **kwargs: Any,
) -> float:
    """Julia's own solve time, which excludes the subprocess and transfer overhead.

    The subprocess is capped at :data:`CASE_TIMEOUT_SECONDS` unless the caller
    passes its own ``timeout``.
    """
    kwargs.setdefault("timeout", CASE_TIMEOUT_SECONDS)
    result = solve._julia_solve_with_timing(
        system_name,
        y0,
        t_span,
        params,
        **kwargs,
    )
    return result.solve_time_s * 1000


# --- running a driver's jobs in a child process under a deadline ---------------
#
# Neither an XLA compile nor a CUDA synchronisation returns to Python until it
# is done, so nothing in-process -- not ``signal.alarm``, not a watchdog thread
# -- can interrupt a case that is taking too long. The measurements therefore
# run in a child process, one per case, which streams a JSON line per point
# back to the driver; the driver enforces the deadline and, when a point
# overruns, kills the child's whole process group (the Julia grandchild
# included) and starts a fresh child for the points that remain. The child is
# ``benchmarks/_worker.py``.

CHILD_FAILED = "child process exited"


def _drain(stream, sink: Callable[[str], None]) -> None:
    for line in stream:
        sink(line)
    sink(None)  # type: ignore[arg-type]


def run_jobs(
    script_path: Path,
    jobs: Sequence[dict],
    *,
    on_started: Callable[[dict], None],
    on_result: Callable[[dict, str, Any], None],
    timeout: float = CASE_TIMEOUT_SECONDS,
) -> None:
    """Measure ``jobs`` with the driver's ``measure`` in a child process.

    Each job is a JSON-serialisable dict the driver understands. ``on_started``
    is called when the child begins a job; ``on_result`` with the job, a status
    -- ``"ok"``, ``"timeout"`` or ``"failed"`` -- and the measurement, the
    timeout cache entry, or the error text. ``timeout`` is the cap in seconds
    on each job from the moment the child begins it: what the driver's
    ``prepare`` does for a job, such as checking the Julia toolchain, is not
    counted.
    """
    pending = list(jobs)
    while pending:
        proc = subprocess.Popen(
            [sys.executable, "-m", "benchmarks._worker", str(script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(pending))
        proc.stdin.close()

        lines: queue.Queue = queue.Queue()
        stderr_tail: list[str] = []

        def keep_tail(line, tail=stderr_tail):
            if line is not None:
                tail.append(line)
                del tail[:-40]

        threading.Thread(target=_drain, args=(proc.stdout, lines.put)).start()
        threading.Thread(target=_drain, args=(proc.stderr, keep_tail)).start()

        def kill() -> None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

        def failure() -> str:
            tail = "".join(stderr_tail).strip()
            return f"{CHILD_FAILED}\n{tail}" if tail else CHILD_FAILED

        respawn = False
        for index, job in enumerate(pending):
            # The child announces the job before it measures, so that the
            # deadline starts at the measurement and not at the child's start.
            try:
                line = lines.get(timeout=timeout)
            except queue.Empty:
                line = None
            if line is None or "started" not in json.loads(line):
                kill()
                on_started(job)
                on_result(job, "failed", failure())
                pending = pending[index + 1 :]
                respawn = True
                break
            on_started(job)
            try:
                line = lines.get(timeout=timeout)
            except queue.Empty:
                kill()
                on_result(job, "timeout", timeout_cache_entry())
                pending = pending[index + 1 :]
                respawn = True
                break
            if line is None:
                kill()
                on_result(job, "failed", failure())
                pending = pending[index + 1 :]
                respawn = True
                break
            message = json.loads(line)
            status = message["status"]
            if status == "ok":
                on_result(job, status, message["result"])
            elif status == "timeout":
                on_result(job, status, timeout_cache_entry())
            else:
                on_result(job, "failed", message["error"])
        if not respawn:
            pending = []
            proc.wait()
