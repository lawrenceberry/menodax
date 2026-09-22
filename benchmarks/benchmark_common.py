"""Helpers shared by the two benchmark drivers, ``_divergence.py`` and ``_sweep.py``."""

from __future__ import annotations

import json
import os
import subprocess
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

T = TypeVar("T")

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


def output_paths(
    script_dir: Path, gpu_name: str, scenario: str | None = None
) -> tuple[Path, Path]:
    """The CSV and plot paths for one GPU, and one scenario if the script has several."""
    stem = (
        gpu_slug(gpu_name) if scenario is None else f"{gpu_slug(gpu_name)}-{scenario}"
    )
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


def julia_solve_time_ms(
    solve: Any,
    system_name: str,
    y0: Any,
    t_span: Any,
    params: Any,
    **kwargs: Any,
) -> float:
    """Julia's own solve time, which excludes the subprocess and transfer overhead."""
    result = solve._julia_solve_with_timing(
        system_name,
        y0,
        t_span,
        params,
        **kwargs,
    )
    return result.solve_time_s * 1000
