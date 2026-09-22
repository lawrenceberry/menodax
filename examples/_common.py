"""What the worked examples share: the solver backends and the benchmark.

Each example's science runs on a menodax kernel solver. For a like-for-like
timing it also drives two reference backends over the same right-hand side --
Diffrax on the GPU under ``jax.vmap``, and ``scipy.solve_ivp`` on the CPU,
the no-GPU baseline serial codes such as ECHO21 use. What differs between the
three for one example is small enough to declare in a :class:`Backends`; this
module holds the rest: a uniform ``solve(ode_fn, y0, t_span, params)`` for a
backend name, the timing loop, the benchmark table and the command line.
"""

from __future__ import annotations

import argparse
import importlib
import time
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import numpy as np

from examples.dual_backend import Forms

BACKENDS = ("menodax", "diffrax", "scipy")


class Backends(NamedTuple):
    """How one example runs its ensemble on each backend.

    ``menodax_solve`` is the kernel solver (``menodax.tsit5.solve`` or
    ``menodax.rodas5P.solve``); ``diffrax_method`` names the Diffrax reference
    solver as its module suffix, ``"kvaerno5"`` or ``"tsit5"``. The three
    keyword dicts are what each solver is called with beyond
    ``(ode_fn, y0, t_span, params)``.
    """

    menodax_solve: Callable
    menodax_kwargs: dict[str, Any]
    diffrax_method: str
    diffrax_kwargs: dict[str, Any]
    scipy_kwargs: dict[str, Any]


def make_solver(backend: str, backends: Backends):
    """Return a uniform ``solve(ode_fn, y0, t_span, params)`` for a backend.

    The reference solvers are imported here rather than at module load, so an
    example that only ever runs on the kernel solver never imports Diffrax.
    """
    if backend == "menodax":
        return partial(backends.menodax_solve, **backends.menodax_kwargs)
    if backend == "diffrax":
        module = importlib.import_module(
            f"reference.solvers.python.diffrax_{backends.diffrax_method}"
        )
        return partial(module.solve, **backends.diffrax_kwargs)
    if backend == "scipy":
        from reference.solvers.python.scipy_solve_ivp import solve as scipy_solve

        return partial(scipy_solve, **backends.scipy_kwargs)
    raise ValueError(f"unknown backend: {backend}")


def rhs_for(backend: str, rhs: Forms):
    """The form of a right-hand side a backend takes.

    The kernel solver compiles the ``device`` form with numba-cuda; the
    reference backends trace the ``jax`` one built from the same body.
    """
    return rhs.device if backend == "menodax" else rhs.jax


def time_solve(fn, repeats):
    """Return (mean seconds excluding compile, result) over ``repeats`` runs."""
    result = fn()
    jax.block_until_ready(result)
    t0 = time.perf_counter()
    for _ in range(repeats):
        result = fn()
        jax.block_until_ready(result)
    return (time.perf_counter() - t0) / repeats, result


def run_benchmark(
    backends: Backends,
    rhs: Forms,
    y0,
    t_span,
    params,
    *,
    backend_names,
    repeats,
    title,
    column,
    metric,
):
    """Time one ensemble solve per backend and print a row for each.

    ``metric`` maps the ``(N, n_save, n_vars)`` solution, as a NumPy array, to
    the string in the last column -- a scalar the backends should agree on,
    as a sanity check on the timings meaning the same thing.
    """
    n = params.shape[0]
    print(f"{title}\n", flush=True)
    print(f"{'backend':>10}  {'wall (s)':>10}  {'per solve':>12}  {column}", flush=True)
    print("-" * 60, flush=True)
    for backend in backend_names:
        solve = make_solver(backend, backends)
        ode_fn = rhs_for(backend, rhs)
        try:
            secs, sol = time_solve(lambda: solve(ode_fn, y0, t_span, params), repeats)
        except Exception as exc:  # noqa: BLE001
            print(f"{backend:>10}  FAILED: {exc}", flush=True)
            continue
        value = metric(np.asarray(sol))
        print(
            f"{backend:>10}  {secs:10.3f}  {secs / n * 1e3:9.4f} ms  {value}",
            flush=True,
        )


def parse_args(description, *, n_default, n_help):
    """The examples' shared command line: run the science, or ``--benchmark``."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="time the batched ensemble solve across solver backends",
    )
    parser.add_argument(
        "--backends", nargs="+", default=list(BACKENDS), choices=BACKENDS
    )
    parser.add_argument("--n", type=int, default=n_default, help=n_help)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()
