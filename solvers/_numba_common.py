"""Shared host-side helpers for numba-cuda custom-kernel solvers."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.descriptor import MLIRDispatcher


@dataclass
class NumbaWorkspace:
    y0_dev: Any
    times_dev: Any
    params_dev: Any
    hist_dev: Any
    accepted_dev: Any
    rejected_dev: Any
    loop_dev: Any


@dataclass(frozen=True)
class PreparedNumbaSolve:
    kernel: Any
    workspace: Any
    dt0: np.float64
    rtol: np.float64
    atol: np.float64
    max_steps: np.int32
    blocks: int
    threads: Any


def normalize_inputs(y0, t_span, params, first_step, *, solver_name: str):
    y0_in = np.asarray(y0, dtype=np.float64)
    params_arr = np.asarray(params, dtype=np.float64)
    times = np.asarray(t_span, dtype=np.float64)

    if y0_in.ndim == 1 and params_arr.ndim == 1:
        n = 1
        y0_arr = np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
        params_arr = np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
    elif y0_in.ndim == 1:
        n = params_arr.shape[0]
        y0_arr = np.broadcast_to(y0_in, (n, y0_in.shape[0])).copy()
    else:
        n = y0_in.shape[0]
        y0_arr = y0_in
        if params_arr.ndim == 1:
            params_arr = np.broadcast_to(params_arr, (n, params_arr.shape[0])).copy()
        elif params_arr.shape[0] != n:
            raise ValueError(
                "params must have shape (n_params,) or (N, n_params) when y0 has "
                f"shape (N, n_vars); got y0.shape={y0_in.shape} and "
                f"params.shape={params_arr.shape}"
            )

    if y0_arr.ndim != 2:
        raise ValueError(f"custom-kernel {solver_name} expects y0 shape (N, n_vars)")
    if params_arr.ndim != 2:
        raise ValueError(
            f"custom-kernel {solver_name} expects params shape (N, n_params)"
        )
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("t_span must be a 1-D array with at least two save times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_span must be strictly increasing")

    return y0_arr, times, params_arr, initial_step(times, first_step)


def build_error_weights(error_weights, n: int, n_vars: int) -> np.ndarray:
    """Broadcast a user ``error_weights`` argument to a ``(n, n_vars)`` array.

    ``None`` yields all-ones (every component weighted equally); a 1-D array of
    length ``n_vars`` is broadcast across trajectories; a 2-D ``(n, n_vars)``
    array is used as-is. This array is copied to the device and read per
    component as the ``weight`` argument of the kernel's error-contribution
    device function.
    """
    if error_weights is None:
        return np.ones((n, n_vars), dtype=np.float64)
    weights = np.asarray(error_weights, dtype=np.float64)
    if weights.ndim == 1:
        weights = np.broadcast_to(weights, (n, n_vars))
    return np.ascontiguousarray(weights, dtype=np.float64)


def initial_step(times, first_step):
    return (
        np.float64(first_step)
        if first_step is not None
        else np.float64((times[-1] - times[0]) * 1e-6)
    )


def copy_workspace_inputs(workspace, y0_arr, times, params_arr):
    workspace.y0_dev.copy_to_device(y0_arr)
    workspace.times_dev.copy_to_device(times)
    workspace.params_dev.copy_to_device(params_arr)


def numpy_stats(accepted_steps, rejected_steps, loop_steps):
    return {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": np.ones_like(loop_steps, dtype=np.int32),
    }


def jax_stats(accepted, rejected, loop_steps):
    return {
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "loop_steps": loop_steps,
        "batch_loop_iterations": loop_steps,
        "valid_lanes": jnp.ones_like(loop_steps, dtype=jnp.int32),
    }


def as_launch_block_dim(block_dim):
    if isinstance(block_dim, int):
        return block_dim
    if isinstance(block_dim, (tuple, list)):
        return tuple(int(x) for x in block_dim)
    x = getattr(block_dim, "x", None)
    y = getattr(block_dim, "y", 1)
    z = getattr(block_dim, "z", 1)
    if x is not None:
        return (int(x), int(y), int(z))
    return 128


def block_threads_x(block_dim) -> int:
    launch = as_launch_block_dim(block_dim)
    if isinstance(launch, int):
        return launch
    return int(launch[0])


@functools.cache
def as_cuda_device(fn):
    if isinstance(fn, MLIRDispatcher):
        return fn
    return cuda.jit(device=True)(fn)


@functools.cache
def make_cuda_vector_writer(fn, n_vars: int):
    fn_device = as_cuda_device(fn)

    @cuda.jit(device=True)
    def write_vector(y, t, p, out, i):
        values = fn_device(y[i], t, p[i])
        for j in range(n_vars):
            out[i, j] = values[j]

    return write_vector


@functools.cache
def make_cuda_transposed_vector_writer(fn, n_vars: int):
    """Like :func:`make_cuda_vector_writer`, but for transposed (SoA) state.

    State/work arrays are laid out ``(n_vars, n)`` so that for a fixed component
    the trajectory axis is contiguous. The strided column ``y[:, s]`` passed to
    the callback is coalesced across the warp (all lanes read the same component
    at consecutive ``s``), so no per-lane gather is needed. ``prow`` is the
    trajectory's parameter row and ``s`` indexes the column of both the input
    state and the output array; this lets the same writer drive a global
    workspace (``s`` = global trajectory index) or a per-block shared workspace
    (``s`` = thread-within-block index).
    """
    fn_device = as_cuda_device(fn)

    @cuda.jit(device=True)
    def write_vector(y, t, prow, out, s):
        values = fn_device(y[:, s], t, prow)
        for j in range(n_vars):
            out[j, s] = values[j]

    return write_vector


@functools.cache
def make_cuda_striped_vector_writer(fn, n_vars: int):
    """Like :func:`make_cuda_vector_writer`, but each lane writes a disjoint
    output stripe ``j = lane, lane + stride, ...`` so a batch's lanes share the
    n_vars-element write. Every lane evaluates the full callback (cheap and
    wall-clock-free under SIMT lockstep); only the global write is split."""
    fn_device = as_cuda_device(fn)

    @cuda.jit(device=True)
    def write_vector(y, t, p, out, i, lane, stride):
        values = fn_device(y[i], t, p[i])
        for j in range(lane, n_vars, stride):
            out[i, j] = values[j]

    return write_vector


@functools.cache
def make_cuda_matrix_writer(fn, n_vars: int):
    fn_device = as_cuda_device(fn)

    @cuda.jit(device=True)
    def write_matrix(y, t, p, out, i):
        values = fn_device(y[i], t, p[i])
        for row in range(n_vars):
            for col in range(n_vars):
                out[i, row, col] = values[row][col]

    return write_matrix


@functools.cache
def make_cuda_zero_vector_writer(n_vars: int):
    @cuda.jit(device=True)
    def write_zero_vector(y, t, p, out, i):
        for j in range(n_vars):
            out[i, j] = 0.0

    return write_zero_vector
