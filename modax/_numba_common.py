"""Shared host-side helpers for numba-cuda custom-kernel solvers."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.descriptor import MLIRDispatcher

from modax._jax_numba_custom_call import (
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
HOOK_OUT_ARGTYPE = _F64_2D
"""Per-trajectory rows a save hook accumulates into (``(n, hook_size)``)."""
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
    n_save_hist: int | None = None,
    hook_size: int | None = None,
):
    """Launch a solver kernel from JAX and return its solution outputs.

    ``arrays`` are the array inputs (y0, times, params, error weights) in kernel
    order and ``scratch_specs`` describes the kernel's scratch arrays, which XLA
    allocates as extra outputs. Returns ``(hist, accepted, rejected, loop)``.

    ``n_save_hist`` is the history's time extent when it differs from the number
    of save times (a kernel that keeps only the final state passes 1), and
    ``hook_size`` the width of the save hook's per-trajectory accumulator rows,
    which the kernel takes as one more output right after the counters; the
    call then returns ``(hist, accepted, rejected, loop, hook_out)``.
    """
    int_spec = jax.ShapeDtypeStruct((n,), jnp.int32)
    n_hist = n_save if n_save_hist is None else n_save_hist
    output_specs = (
        jax.ShapeDtypeStruct((n, n_hist, n_vars), jnp.float64),
        int_spec,
        int_spec,
        int_spec,
    )
    if hook_size is not None:
        output_specs += (jax.ShapeDtypeStruct((n, hook_size), jnp.float64),)
    output_specs += tuple(scratch_specs)
    result = ffi_abi_call(
        launch,
        arrays,
        output_specs,
        input_kinds=SOLVER_INPUT_KINDS,
        scalar_f64_values=(dt0, rtol, atol),
        scalar_i32_values=(max_steps,),
    )
    return result[:4] if hook_size is None else result[:5]


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


def make_cuda_local_vector_writer(fn, n_vars: int):
    """A vector writer over one trajectory's own thread-local arrays.

    Both kernels run one trajectory per thread, so there is no stripe to share:
    the thread owning the trajectory writes the whole vector, and both
    ``y_row`` and ``out`` are its own local memory rather than rows of a global
    scratch array.
    """
    fn_device = as_cuda_device(fn)

    @cuda.jit(device=True)
    def write_vector(y_row, t, p_row, out):
        values = fn_device(y_row, t, p_row)
        for j in range(n_vars):
            out[j] = values[j]

    return write_vector
