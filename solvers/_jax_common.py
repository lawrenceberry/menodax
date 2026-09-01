"""Shared scaffolding for exposing the numba-cuda ensemble solvers to JAX."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


def normalize_y0_params(y0, params):
    """Broadcast ``y0`` / ``params`` to a consistent ``(N, …)`` ensemble layout.

    Accepts either 1-D (``(n_vars,)`` / ``(n_params,)``) or 2-D
    (``(N, n_vars)`` / ``(N, n_params)``) inputs and returns 2-D arrays with a
    common leading axis, so every numba-cuda solver shares one calling
    convention.
    """
    y0_in = jnp.asarray(y0, dtype=jnp.float64)
    params_arr = jnp.asarray(params)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        n = 1
        n_vars = y0_in.shape[0]
        y0_arr = jnp.broadcast_to(y0_in, (n, n_vars))
        params_arr = jnp.broadcast_to(params_arr, (n, params_arr.shape[0]))
    elif y0_in.ndim == 1:
        n = params_arr.shape[0]
        n_vars = y0_in.shape[0]
        y0_arr = jnp.broadcast_to(y0_in, (n, n_vars))
    else:
        n = y0_in.shape[0]
        n_vars = y0_in.shape[1]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = jnp.broadcast_to(params_arr, (n, params_arr.shape[0]))
        elif params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )
    return y0_arr, params_arr, n, n_vars


def _broadcast_for_vmap(arg, is_batched: bool, axis_size: int, name: str):
    arr = jnp.asarray(arg)
    if is_batched:
        if arr.ndim != 2:
            raise NotImplementedError(
                f"vmap over an already-ensembled {name} is not supported; "
                "call the solver with batched y0/params directly instead."
            )
        if arr.shape[0] != axis_size:
            raise ValueError(
                f"batched {name} has leading axis {arr.shape[0]}, expected {axis_size}"
            )
        return arr
    if arr.ndim != 1:
        raise NotImplementedError(
            f"vmap with unbatched ensemble-shaped {name} is not supported; "
            "call the solver with batched y0/params directly instead."
        )
    return jnp.broadcast_to(arr, (axis_size,) + arr.shape)


def per_trajectory_stats_postprocess(stats, axis_size):
    """Stats reshape when every key already has shape ``(axis_size,)``.

    Used by the numba-cuda solvers whose kernels emit per-trajectory counters
    for every stats field.
    """
    del axis_size
    stats_out = jax.tree_util.tree_map(lambda x: x[:, None], stats)
    stats_batched = jax.tree_util.tree_map(lambda _: True, stats_out)
    return stats_out, stats_batched


def make_custom_vmap_solver(
    solve_impl: Callable,
    *,
    return_stats: bool,
    stats_postprocess: Callable | None = None,
):
    """Wrap a solver implementation so outer ``jax.vmap`` becomes one ensemble call.

    ``solve_impl`` must accept ``(y0, t_span, params)`` and return the normal
    public solver result for those arrays.  The custom batching rule supports
    vmapping scalar solves over ``y0`` and/or ``params`` and lowers that vmap to
    a single native ensemble solve with a leading trajectory axis.

    ``stats_postprocess`` reshapes the stats pytree after the ensemble solve;
    it defaults to :func:`per_trajectory_stats_postprocess`, matching the
    numba-cuda kernels that emit per-trajectory counters.
    """

    if stats_postprocess is None:
        stats_postprocess = per_trajectory_stats_postprocess

    @custom_vmap
    def _solve(y0, t_span, params):
        return solve_impl(y0, t_span, params)

    @_solve.def_vmap
    def _solve_vmap(axis_size, in_batched, y0, t_span, params):
        y0_batched, t_span_batched, params_batched = in_batched
        if t_span_batched:
            t_span_arr = jnp.asarray(t_span)
            if t_span_arr.ndim != 2:
                raise NotImplementedError(
                    "vmap over nested t_span values is not supported; use a shared "
                    "t_span and vmap over y0 and/or params, or call the solver directly."
                )
            # JAX can mark closed-over constant save times as batched inside
            # a larger vmapped function.  Treat that as a shared time grid.
            t_span = t_span_arr[0]

        y0_arr = _broadcast_for_vmap(y0, y0_batched, axis_size, "y0")
        params_arr = _broadcast_for_vmap(params, params_batched, axis_size, "params")
        result = solve_impl(y0_arr, t_span, params_arr)

        if not return_stats:
            return result[:, None, :, :], True

        sol, stats = result
        stats_out, stats_batched = stats_postprocess(stats, axis_size)
        return (sol[:, None, :, :], stats_out), (True, stats_batched)

    return _solve
