"""Shared host-side helpers for numba-cuda custom-kernel solvers."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import MLIRDispatcher

from solvers._jax_common import normalize_y0_params
from solvers._jax_numba_custom_call import (
    ABI_ARRAY,
    ABI_SCALAR_F64,
    ABI_SCALAR_I32,
    ffi_abi_call,
)

# Both kernels open with the same parameters -- the ensemble inputs, the scalar
# step-control settings, and the history and counter outputs -- and differ only
# in the scratch arrays that follow, so the leading argument types and ABI kinds
# are declared once here.
_F64_1D = types.float64[::1]
_F64_2D = types.float64[:, ::1]
_I32_1D = types.int32[::1]
SOLVER_ARGTYPES = (
    _F64_2D,  # y0
    _F64_1D,  # times
    _F64_2D,  # params
    types.float64,  # dt0
    types.float64,  # rtol
    types.float64,  # atol
    types.int32,  # max_steps
    _F64_2D,  # error weights
    types.float64[:, :, ::1],  # hist
    _I32_1D,  # accepted steps
    _I32_1D,  # rejected steps
    _I32_1D,  # loop steps
)
SCRATCH_ARGTYPE = _F64_2D
SOLVER_INPUT_KINDS = (
    ABI_ARRAY,
    ABI_ARRAY,
    ABI_ARRAY,
    ABI_SCALAR_F64,
    ABI_SCALAR_F64,
    ABI_SCALAR_F64,
    ABI_SCALAR_I32,
    ABI_ARRAY,
)


@dataclass
class NumbaWorkspace:
    y0_dev: Any
    times_dev: Any
    params_dev: Any
    weights_dev: Any
    hist_dev: Any
    accepted_dev: Any
    rejected_dev: Any
    loop_dev: Any
    work: list[Any]


@dataclass(frozen=True)
class PreparedNumbaSolve:
    kernel: Any
    workspace: NumbaWorkspace
    dt0: np.float64
    rtol: np.float64
    atol: np.float64
    max_steps: np.int32
    blocks: int
    threads: Any


def normalize_inputs(y0, t_span, params, first_step):
    y0_arr, params_arr, _, _ = normalize_y0_params(y0, params, xp=np)
    times = np.asarray(t_span, dtype=np.float64)

    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("t_span must be a 1-D array with at least two save times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t_span must be strictly increasing")

    # Broadcasting leaves zero-stride views behind; the device copies need real
    # buffers.
    return (
        np.ascontiguousarray(y0_arr),
        times,
        np.ascontiguousarray(params_arr),
        initial_step(first_step),
    )


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


def initial_step(first_step):
    """The ``dt0`` scalar the kernels take, from a user ``first_step``.

    ``dt0`` is a launch-time scalar, so it cannot be derived here from the save
    times: under ``jit`` (and under the ``custom_vmap`` rule, which traces
    unconditionally) those are tracers. A caller that pins no first step passes
    the non-positive sentinel instead, and each kernel takes 1e-6 of its own
    integration window off the ``times`` array it already reads.
    """
    return np.float64(0.0 if first_step is None else first_step)


def get_workspace(
    cache: dict,
    n: int,
    n_vars: int,
    n_save: int,
    n_params: int,
    *,
    transposed: bool,
    n_work: int,
) -> NumbaWorkspace:
    """Fetch (or allocate and cache) the device workspace for one solve shape.

    ``transposed`` selects the state layout: ``(n_vars, n)`` for kernels whose
    warps read one component across trajectories, ``(n, n_vars)`` for kernels
    that keep a trajectory's row together. ``hist`` always keeps the
    ``(n, n_save, n_vars)`` output layout, since it is written only at save
    points.
    """
    key = (n, n_vars, n_save, n_params)
    workspace = cache.get(key)
    if workspace is not None:
        return workspace

    state_shape = (n_vars, n) if transposed else (n, n_vars)
    workspace = NumbaWorkspace(
        y0_dev=cuda.device_array(state_shape, dtype=np.float64),
        times_dev=cuda.device_array(n_save, dtype=np.float64),
        params_dev=cuda.device_array((n, n_params), dtype=np.float64),
        weights_dev=cuda.device_array(state_shape, dtype=np.float64),
        hist_dev=cuda.device_array((n, n_save, n_vars), dtype=np.float64),
        accepted_dev=cuda.device_array(n, dtype=np.int32),
        rejected_dev=cuda.device_array(n, dtype=np.int32),
        loop_dev=cuda.device_array(n, dtype=np.int32),
        work=[cuda.device_array(state_shape, dtype=np.float64) for _ in range(n_work)],
    )
    cache[key] = workspace
    return workspace


def copy_workspace_inputs(workspace, y0_arr, times, params_arr, weights_arr):
    workspace.y0_dev.copy_to_device(y0_arr)
    workspace.times_dev.copy_to_device(times)
    workspace.params_dev.copy_to_device(params_arr)
    workspace.weights_dev.copy_to_device(weights_arr)


def run_kernel(prepared, scratch, *, return_stats: bool, copy_solution: bool):
    """Launch a prepared solve directly through numba and collect its outputs."""
    workspace = prepared.workspace
    prepared.kernel[prepared.blocks, prepared.threads](
        workspace.y0_dev,
        workspace.times_dev,
        workspace.params_dev,
        prepared.dt0,
        prepared.rtol,
        prepared.atol,
        prepared.max_steps,
        workspace.weights_dev,
        workspace.hist_dev,
        workspace.accepted_dev,
        workspace.rejected_dev,
        workspace.loop_dev,
        *scratch,
    )
    cuda.synchronize()

    solution = (
        workspace.hist_dev.copy_to_host() if copy_solution else workspace.hist_dev
    )
    if not return_stats:
        return solution
    return solution, solver_stats(
        workspace.accepted_dev.copy_to_host(),
        workspace.rejected_dev.copy_to_host(),
        workspace.loop_dev.copy_to_host(),
    )


def ensemble_ffi_call(
    launch,
    arrays,
    scratch_specs,
    *,
    n: int,
    n_vars: int,
    n_save: int,
    dt0,
    rtol,
    atol,
    max_steps,
):
    """Launch a solver kernel from JAX and return its four solution outputs.

    ``arrays`` are the array inputs (y0, times, params, error weights) in kernel
    order and ``scratch_specs`` describes the kernel's scratch arrays, which XLA
    allocates as extra outputs. Returns ``(hist, accepted, rejected, loop)``.
    """
    int_spec = jax.ShapeDtypeStruct((n,), jnp.int32)
    output_specs = (
        jax.ShapeDtypeStruct((n, n_save, n_vars), jnp.float64),
        int_spec,
        int_spec,
        int_spec,
    ) + tuple(scratch_specs)
    result = ffi_abi_call(
        launch,
        arrays,
        output_specs,
        input_kinds=SOLVER_INPUT_KINDS,
        scalar_f64_values=(dt0, rtol, atol),
        scalar_i32_values=(max_steps,),
    )
    return result[:4]


def solver_stats(accepted, rejected, loop_steps):
    return {
        "accepted_steps": accepted,
        "rejected_steps": rejected,
        "loop_steps": loop_steps,
    }


@functools.cache
def as_cuda_device(fn):
    if isinstance(fn, MLIRDispatcher):
        return fn
    return cuda.jit(device=True)(fn)


@functools.cache
def make_cuda_transposed_vector_writer(fn, n_vars: int):
    """A vector writer for transposed (SoA) state.

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
    """A vector writer in which each lane writes a disjoint
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
