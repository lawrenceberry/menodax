"""Shared scaffolding for exposing the numba-cuda ensemble solvers to JAX."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap


def normalize_y0_params(y0, params, xp=jnp):
    """Broadcast ``y0`` / ``params`` to a consistent ``(N, …)`` ensemble layout.

    Accepts either 1-D (``(n_vars,)`` / ``(n_params,)``) or 2-D
    (``(N, n_vars)`` / ``(N, n_params)``) inputs and returns 2-D arrays with a
    common leading axis, so every numba-cuda solver shares one calling
    convention.  ``xp`` picks the array module: ``jnp`` for the JAX entry
    points, ``numpy`` for the host-side ``prepare_solve`` path.
    """
    y0_arr = xp.asarray(y0, dtype=xp.float64)
    params_arr = xp.asarray(params, dtype=xp.float64)

    if y0_arr.ndim not in (1, 2) or params_arr.ndim not in (1, 2):
        raise ValueError(
            "y0 must have shape (n_vars,) or (N, n_vars) and params shape "
            f"(n_params,) or (N, n_params); got y0.shape={y0_arr.shape} and "
            f"params.shape={params_arr.shape}"
        )
    if y0_arr.ndim == 2:
        n = y0_arr.shape[0]
        if params_arr.ndim == 2 and params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_arr.shape} and "
                f"params.shape={params_arr.shape}"
            )
    elif params_arr.ndim == 2:
        n = params_arr.shape[0]
    else:
        n = 1

    if y0_arr.ndim == 1:
        y0_arr = xp.broadcast_to(y0_arr, (n, y0_arr.shape[0]))
    if params_arr.ndim == 1:
        params_arr = xp.broadcast_to(params_arr, (n, params_arr.shape[0]))
    return y0_arr, params_arr, n, y0_arr.shape[1]


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


def make_custom_vmap_solver(solve_impl: Callable, *, return_stats: bool):
    """Wrap a solver implementation so outer ``jax.vmap`` becomes one ensemble call.

    ``solve_impl`` must accept ``(y0, t_span, params)`` and return the normal
    public solver result for those arrays.  The custom batching rule supports
    vmapping scalar solves over ``y0`` and/or ``params`` and lowers that vmap to
    a single native ensemble solve with a leading trajectory axis.  Every stats
    field the kernels emit is a per-trajectory counter, so the stats pytree
    only needs a trailing solve axis added.
    """

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
        stats_out = jax.tree_util.tree_map(lambda x: x[:, None], stats)
        stats_batched = jax.tree_util.tree_map(lambda _: True, stats_out)
        return (sol[:, None, :, :], stats_out), (True, stats_batched)

    return _solve
