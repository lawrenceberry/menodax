"""Reference CPU solver via :func:`scipy.integrate.solve_ivp`.

This mirrors the no-GPU baseline used by serial cosmology codes such as ECHO21,
which integrate each trajectory independently on the CPU with
``scipy.integrate.solve_ivp``.  It is the "without GPU" reference for the
worked cosmological examples.

To make the comparison cost-equivalent with a modern GPU, the wrapper
parallelises the ensemble across CPU cores with :mod:`multiprocessing`.  Each
worker process JIT-compiles the right-hand side once on the CPU backend in its
``initializer``, then runs ``scipy.solve_ivp`` serially over its share of
trajectories.  All :mod:`jax` imports are deliberately deferred so that
spawned children can be pinned to the CPU platform without inheriting the
parent's GPU state.
"""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing as mp
import os

import numpy as np
from scipy.integrate import solve_ivp

# Environment a CPU-only JAX process wants: the platform pin, and one BLAS /
# OpenMP thread per process so ``n_workers`` processes do not oversubscribe.
_CPU_PLATFORM_ENV = ("JAX_PLATFORMS", "cpu")
_SINGLE_THREAD_ENV = (
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
)


def _set_cpu_env_defaults() -> None:
    """Pin JAX to the CPU and BLAS to one thread unless already configured."""
    for key, value in (_CPU_PLATFORM_ENV, *_SINGLE_THREAD_ENV):
        os.environ.setdefault(key, value)


@contextlib.contextmanager
def _cpu_env_for_spawn():
    """Temporarily force the CPU environment while spawning worker processes.

    Children inherit the environment at spawn time; if the parent has GPU JAX,
    the child must be told to use CPU before any ``import jax`` runs (which
    happens during unpickling of the user's example module).  The platform pin
    is forced -- a parent pinned to ``gpu`` must not leak that -- while the
    thread counts only fill in defaults.  The parent's own environment is
    restored on exit.
    """
    keys = [_CPU_PLATFORM_ENV[0], *(key for key, _ in _SINGLE_THREAD_ENV)]
    snapshot = {key: os.environ.get(key) for key in keys}
    os.environ[_CPU_PLATFORM_ENV[0]] = _CPU_PLATFORM_ENV[1]
    _set_cpu_env_defaults()
    try:
        yield
    finally:
        for key, value in snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _solve_trajectory(
    rhs, t0, tf, save_times, method, rtol, atol, first_step, y0_i, p_i
):
    """Integrate one trajectory and return it as ``[n_save, n_vars]``.

    ``rhs`` is a JIT-compiled ``(y, t, p) -> dy/dt`` on the CPU backend.  Rows
    the integrator did not reach (an early failure) are left as NaN so the
    caller can see where a trajectory stopped rather than reading stale data.
    """
    import jax.numpy as jnp  # noqa: PLC0415  (jax is imported lazily, see module doc)

    n_vars = y0_i.shape[0]

    def fun(t, y):
        return np.asarray(rhs(jnp.asarray(y), t, p_i), dtype=np.float64)

    sol = solve_ivp(
        fun,
        (t0, tf),
        y0_i,
        method=method,
        t_eval=save_times,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
    )
    out = np.full((save_times.shape[0], n_vars), np.nan, dtype=np.float64)
    ys = np.asarray(sol.y, dtype=np.float64)
    if ys.ndim == 2 and ys.shape[0] == n_vars:
        ys = ys.T
        out[: ys.shape[0]] = ys
    return out


# ---------------------------------------------------------------------------
# Worker-side machinery (runs only inside spawned subprocesses)
# ---------------------------------------------------------------------------

_WORKER_RHS = None  # set by ``_worker_init``


def _worker_init(rhs) -> None:
    """Spawned-worker initialiser: force CPU JAX, then JIT the RHS once.

    ``rhs`` is delivered through multiprocessing's own pickler, which handles
    ``__main__`` functions across spawned processes (plain :mod:`pickle` does
    not).  Closures still require a picklable callable; the examples ensure
    this by exposing their right-hand sides as module-level functions or
    callable class instances.
    """
    _set_cpu_env_defaults()
    import jax  # noqa: PLC0415  (deferred so the parent's GPU JAX is not used)

    global _WORKER_RHS
    _WORKER_RHS = jax.jit(lambda y, t, p: rhs(y, t, p))


def _worker_solve_one(args):
    """Solve a single trajectory using the worker's pre-JIT'd RHS."""
    return _solve_trajectory(_WORKER_RHS, *args)


# ---------------------------------------------------------------------------
# Persistent pool cache (so the warm-up JIT in each worker is paid once)
# ---------------------------------------------------------------------------

_POOL_CACHE: dict = {}


def _close_pools() -> None:
    for pool in list(_POOL_CACHE.values()):
        try:
            pool.terminate()
            pool.join()
        except Exception:
            pass


atexit.register(_close_pools)


def _get_pool(ode_fn, n_workers: int):
    key = (id(ode_fn), n_workers)
    if key not in _POOL_CACHE:
        with _cpu_env_for_spawn():
            ctx = mp.get_context("spawn")
            _POOL_CACHE[key] = ctx.Pool(
                n_workers, initializer=_worker_init, initargs=(ode_fn,)
            )
    return _POOL_CACHE[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    method: str = "LSODA",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    first_step=None,
    max_steps: int | None = None,  # noqa: ARG001  (kept for API symmetry)
    n_processes: int | None = None,
    **_ignored,
):
    """Solve an ensemble with ``scipy.solve_ivp``, optionally in parallel.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side ``ode_fn(y, t, params) -> dy/dt`` (jax.numpy ok).
    y0 : array, shape ``[n_vars]`` or ``[N, n_vars]``
        Shared initial state (broadcast to all trajectories) or per-trajectory.
    t_span : array-like, shape ``[n_save]``
        Strictly increasing save times (``len >= 2``).
    params : array, shape ``[N, ...]``
        Per-trajectory parameters.
    method : str
        Any ``solve_ivp`` method.  ``"LSODA"`` (auto stiff/non-stiff) matches
        the choice made by ECHO21; use ``"RK45"`` for non-stiff problems.
    n_processes : int, optional
        Number of worker processes.  ``None`` (default) uses every logical
        core (``os.cpu_count()``).  ``1`` forces a serial in-process solve.

    Returns
    -------
    array, shape ``[N, n_save, n_vars]``
    """
    params_arr = np.asarray(params)
    n = params_arr.shape[0]
    y0_arr = np.asarray(y0, dtype=np.float64)
    if y0_arr.ndim == 1:
        y0_arr = np.broadcast_to(y0_arr, (n, y0_arr.shape[0]))
    save_times = np.asarray(t_span, dtype=np.float64)
    t0, tf = float(save_times[0]), float(save_times[-1])

    if n_processes is None:
        n_workers = os.cpu_count() or 1
    else:
        n_workers = max(1, int(n_processes))
    n_workers = min(n_workers, n)

    if n_workers <= 1:
        # Serial in-process path: JIT the RHS once on the CPU backend.
        import jax  # noqa: PLC0415

        cpu_rhs = jax.jit(lambda y, t, p: ode_fn(y, t, p), backend="cpu")
        out = np.empty((n, save_times.shape[0], y0_arr.shape[1]), dtype=np.float64)
        for i in range(n):
            out[i] = _solve_trajectory(
                cpu_rhs,
                t0,
                tf,
                save_times,
                method,
                rtol,
                atol,
                first_step,
                y0_arr[i],
                params_arr[i],
            )
        return out

    # Parallel multi-process path.
    pool = _get_pool(ode_fn, n_workers)
    args_list = [
        (
            t0,
            tf,
            save_times,
            method,
            rtol,
            atol,
            first_step,
            np.asarray(y0_arr[i]),
            np.asarray(params_arr[i]),
        )
        for i in range(n)
    ]
    chunksize = max(1, n // (n_workers * 4))
    results = pool.map(_worker_solve_one, args_list, chunksize=chunksize)
    return np.stack(results, axis=0)
