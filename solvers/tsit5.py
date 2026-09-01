"""Generic Tsit5 custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import gc
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numba import cuda, types

from solvers._jax_common import (
    make_custom_vmap_solver,
    normalize_y0_params,
    per_trajectory_stats_postprocess,
)
from solvers._jax_numba_custom_call import (
    ABI_ARRAY,
    ABI_SCALAR_F64,
    ABI_SCALAR_I32,
    ffi_abi_call,
    make_launch,
)
from solvers._numba_common import (
    NumbaWorkspace,
    PreparedNumbaSolve,
    build_error_weights,
    initial_step,
    jax_stats,
    make_cuda_transposed_vector_writer,
    numpy_stats,
)
from solvers._numba_common import (
    normalize_inputs as _normalize_inputs,
)

# fmt: off
C2 = 161.0 / 1000.0
C3 = 327.0 / 1000.0
C4 = 9.0 / 10.0
C5 = 0.9800255409045097
C6 = 1.0
C7 = 1.0

A21 = 161.0 / 1000.0

A31 = -0.008480655492356989
A32 = 0.335480655492357

A41 = 2.8971530571054935
A42 = -6.359448489975075
A43 = 4.3622954328695815

A51 = 5.325864828439257
A52 = -11.748883564062828
A53 = 7.4955393428898365
A54 = -0.09249506636175525

A61 = 5.86145544294642
A62 = -12.92096931784711
A63 = 8.159367898576159
A64 = -0.071584973281401
A65 = -0.028269050394068383

A71 = 0.09646076681806523
A72 = 0.01
A73 = 0.4798896504144996
A74 = 1.379008574103742
A75 = -3.2900695154360807
A76 = 2.324710524099774

B1 = A71
B2 = A72
B3 = A73
B4 = A74
B5 = A75
B6 = A76

E1 = 0.0017800110522257773
E2 = 0.0008164344596567463
E3 = -0.007880878010261994
E4 = 0.1447110071732629
E5 = -0.5823571654525552
E6 = 0.45808210592918686
E7 = -1.0 / 66.0
# fmt: on


SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 10.0
# Elementary I-controller exponent (-1/k, k the error order). The PID terms in
# the kernel are expressed relative to this.
EXPONENT = -1.0 / 5.0

_WORKSPACE_CACHE: dict[tuple[int, int, int, int], object] = {}

# Hybrid backend. The shared-memory kernel keeps all stage vectors in on-chip
# shared memory (one thread per trajectory, BLOCK = _SHARED_BLOCK). It wins in
# the latency-bound regime -- small ensembles and/or low dimension, where the
# device is under-occupied and shared memory's low latency dominates -- but
# needs ~9 * _SHARED_BLOCK * n_vars * 8 bytes of static shared memory per block
# (capping n_vars) and its low occupancy loses to the transposed-global kernel
# once the ensemble is large enough to saturate the GPU. The thresholds bound
# the regime where shared was measured to win on an RTX 4070 SUPER; outside it
# the global kernel is used. Retune per GPU or override via ``backend``.
_SHARED_BLOCK = 32
_SHARED_MAX_NVARS = 16
_SHARED_MAX_ENSEMBLE = 16384


def _use_shared_backend(n: int, n_vars: int, backend: str) -> bool:
    if backend == "global":
        return False
    if backend == "shared":
        if n_vars > _SHARED_MAX_NVARS:
            raise ValueError(
                "shared backend requires n_vars <= "
                f"{_SHARED_MAX_NVARS}; got n_vars={n_vars}"
            )
        return True
    if backend != "auto":
        raise ValueError(
            f"backend must be 'auto', 'shared', or 'global'; got {backend!r}"
        )
    return n_vars <= _SHARED_MAX_NVARS and n <= _SHARED_MAX_ENSEMBLE


def clear_caches() -> None:
    """Drop the cached device workspaces and compiled kernels.

    Useful when sweeping problem sizes in a single process: each unique
    ``n_vars`` allocates a fresh device workspace and compiles a separate
    kernel, and neither is released by GC because they're held by the
    module-level caches.
    """
    _WORKSPACE_CACHE.clear()
    _make_body.cache_clear()
    _make_kernel.cache_clear()
    _make_shared_kernel.cache_clear()
    _make_jax_launch.cache_clear()
    _make_shared_jax_launch.cache_clear()
    gc.collect()


@dataclass
class Workspace(NumbaWorkspace):
    work: list[Any]
    weights_dev: Any


@dataclass(frozen=True)
class PreparedSolve(PreparedNumbaSolve):
    workspace: Workspace
    uses_shared: bool = False


def get_workspace(
    cache: dict, n: int, n_vars: int, n_save: int, n_params: int
) -> Workspace:
    key = (n, n_vars, n_save, n_params)
    workspace = cache.get(key)
    if workspace is not None:
        return workspace

    # State and stage vectors are stored transposed (n_vars, n) so that for a
    # fixed component the trajectory axis is contiguous, giving coalesced warp
    # accesses independent of n_vars. hist keeps the (n, n_save, n_vars) output
    # layout (written only at save points).
    workspace = Workspace(
        y0_dev=cuda.device_array((n_vars, n), dtype=np.float64),
        times_dev=cuda.device_array(n_save, dtype=np.float64),
        params_dev=cuda.device_array((n, n_params), dtype=np.float64),
        hist_dev=cuda.device_array((n, n_save, n_vars), dtype=np.float64),
        accepted_dev=cuda.device_array(n, dtype=np.int32),
        rejected_dev=cuda.device_array(n, dtype=np.int32),
        loop_dev=cuda.device_array(n, dtype=np.int32),
        work=[cuda.device_array((n_vars, n), dtype=np.float64) for _ in range(9)],
        weights_dev=cuda.device_array((n_vars, n), dtype=np.float64),
    )
    cache[key] = workspace
    return workspace


@functools.cache
def _make_body(
    ode_fn,
    n_vars: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
):
    """Build the per-trajectory Tsit5 integration loop as a CUDA device fn.

    The body is storage-agnostic: ``i`` indexes the per-trajectory global arrays
    (``y0``/``weights``/``hist``/stats) while ``s`` indexes the stage workspace
    columns. The global kernel passes ``s = i`` (stage vectors are global
    ``(n_vars, n)`` arrays); the shared kernel passes ``s = threadIdx.x`` (stage
    vectors are per-block shared ``(n_vars, BLOCK)`` arrays). This keeps both
    backends sharing a single integrator implementation.
    """
    # PID step-control exponents (Soderlind). Defaults (0, 1, 0) give E1=EXPONENT
    # and E2=E3=0, recovering the elementary I-controller exactly.
    e1 = EXPONENT * (icoeff + pcoeff + dcoeff)
    e2 = -EXPONENT * (pcoeff + 2.0 * dcoeff)
    e3 = EXPONENT * dcoeff
    ode_write = make_cuda_transposed_vector_writer(ode_fn, n_vars)

    @cuda.jit(device=True)
    def body(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        weights,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
        y,
        u,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        i,
        s,
    ):
        prow = params[i]
        for j in range(n_vars):
            y[j, s] = y0[j, i]
            hist[i, 0, j] = y0[j, i]
            k7[j, s] = 0.0

        n_save = times.shape[0]
        t = times[0]
        tf = times[n_save - 1]
        dt = dt0
        save_idx = 1
        n_steps = 0
        accepted_steps = 0
        rejected_steps = 0
        has_fsal = False
        err_prev = 1.0
        err_prev2 = 1.0

        while save_idx < n_save and t < tf and n_steps < max_steps:
            dt_use = dt
            if dt_use > tf - t:
                dt_use = tf - t
            if dt_use < 1e-30:
                dt_use = 1e-30

            if has_fsal:
                for j in range(n_vars):
                    k1[j, s] = k7[j, s]
            else:
                ode_write(y, t, prow, k1, s)

            for j in range(n_vars):
                u[j, s] = y[j, s] + dt_use * (A21 * k1[j, s])
            ode_write(u, t + C2 * dt_use, prow, k2, s)

            for j in range(n_vars):
                u[j, s] = y[j, s] + dt_use * (A31 * k1[j, s] + A32 * k2[j, s])
            ode_write(u, t + C3 * dt_use, prow, k3, s)

            for j in range(n_vars):
                u[j, s] = y[j, s] + dt_use * (
                    A41 * k1[j, s] + A42 * k2[j, s] + A43 * k3[j, s]
                )
            ode_write(u, t + C4 * dt_use, prow, k4, s)

            for j in range(n_vars):
                u[j, s] = y[j, s] + dt_use * (
                    A51 * k1[j, s] + A52 * k2[j, s] + A53 * k3[j, s] + A54 * k4[j, s]
                )
            ode_write(u, t + C5 * dt_use, prow, k5, s)

            for j in range(n_vars):
                u[j, s] = y[j, s] + dt_use * (
                    A61 * k1[j, s]
                    + A62 * k2[j, s]
                    + A63 * k3[j, s]
                    + A64 * k4[j, s]
                    + A65 * k5[j, s]
                )
            ode_write(u, t + C6 * dt_use, prow, k6, s)

            for j in range(n_vars):
                u[j, s] = y[j, s] + dt_use * (
                    B1 * k1[j, s]
                    + B2 * k2[j, s]
                    + B3 * k3[j, s]
                    + B4 * k4[j, s]
                    + B5 * k5[j, s]
                    + B6 * k6[j, s]
                )
            ode_write(u, t + C7 * dt_use, prow, k7, s)

            err_sum = 0.0
            for j in range(n_vars):
                err_est = dt_use * (
                    E1 * k1[j, s]
                    + E2 * k2[j, s]
                    + E3 * k3[j, s]
                    + E4 * k4[j, s]
                    + E5 * k5[j, s]
                    + E6 * k6[j, s]
                    + E7 * k7[j, s]
                )
                scale = atol + rtol * max(abs(y[j, s]), abs(u[j, s]))
                r = weights[j, i] * err_est / scale
                err_sum += r * r
            err_norm = math.sqrt(err_sum / n_vars)
            accept = err_norm <= 1.0 and not math.isnan(err_norm)

            t_new = t
            if accept:
                t_new = t + dt_use
                while save_idx < n_save and times[save_idx] <= t_new + 1e-12 * max(
                    1.0, abs(times[save_idx])
                ):
                    theta = (times[save_idx] - t) / dt_use
                    b1 = (
                        -1.0530884977290216
                        * theta
                        * (theta - 1.3299890189751412)
                        * (
                            theta * theta
                            - 1.4364028541716351 * theta
                            + 0.7139816917074209
                        )
                    )
                    b2 = (
                        0.1017
                        * theta
                        * theta
                        * (
                            theta * theta
                            - 2.1966568338249754 * theta
                            + 1.2949852507374631
                        )
                    )
                    b3 = (
                        2.490627285651252793
                        * theta
                        * theta
                        * (
                            theta * theta
                            - 2.38535645472061657 * theta
                            + 1.57803468208092486
                        )
                    )
                    b4 = (
                        -16.54810288924490272
                        * (theta - 1.21712927295533244)
                        * (theta - 0.61620406037800089)
                        * theta
                        * theta
                    )
                    b5 = (
                        47.37952196281928122
                        * (theta - 1.203071208372362603)
                        * (theta - 0.658047292653547382)
                        * theta
                        * theta
                    )
                    b6 = (
                        -34.87065786149660974
                        * (theta - 1.2)
                        * (theta - 0.666666666666666667)
                        * theta
                        * theta
                    )
                    b7 = 2.5 * (theta - 1.0) * (theta - 0.6) * theta * theta
                    for j in range(n_vars):
                        hist[i, save_idx, j] = y[j, s] + dt_use * (
                            b1 * k1[j, s]
                            + b2 * k2[j, s]
                            + b3 * k3[j, s]
                            + b4 * k4[j, s]
                            + b5 * k5[j, s]
                            + b6 * k6[j, s]
                            + b7 * k7[j, s]
                        )
                    save_idx += 1
                for j in range(n_vars):
                    y[j, s] = u[j, s]
                accepted_steps += 1
                has_fsal = True
            else:
                rejected_steps += 1
                for j in range(n_vars):
                    k7[j, s] = 0.0
                has_fsal = False

            if math.isnan(err_norm) or err_norm > 1e18:
                safe_err = 1e18
            elif err_norm == 0.0:
                safe_err = 1e-18
            else:
                safe_err = err_norm
            factor = SAFETY * safe_err**e1 * err_prev**e2 * err_prev2**e3
            # Advance the PID error history only on accepted steps.
            if accept:
                err_prev2 = err_prev
                err_prev = safe_err
            if factor < FACTOR_MIN:
                factor = FACTOR_MIN
            elif factor > FACTOR_MAX:
                factor = FACTOR_MAX
            dt = dt_use * factor
            t = t_new
            n_steps += 1

        accepted_out[i] = accepted_steps
        rejected_out[i] = rejected_steps
        loop_out[i] = n_steps

    return body


@functools.cache
def _make_kernel(
    ode_fn,
    n_vars: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
):
    """Transposed-global kernel: stage vectors are global ``(n_vars, n)`` arrays."""
    body = _make_body(ode_fn, n_vars, pcoeff, icoeff, dcoeff)

    @cuda.jit
    def kernel(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        weights,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
        y,
        u,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
    ):
        i = cuda.grid(1)
        if i >= y0.shape[1]:
            return
        body(
            y0,
            times,
            params,
            dt0,
            rtol,
            atol,
            max_steps,
            weights,
            hist,
            accepted_out,
            rejected_out,
            loop_out,
            y,
            u,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
            i,
            i,
        )

    return kernel


@functools.cache
def _make_shared_kernel(
    ode_fn,
    n_vars: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
):
    """Shared-memory kernel: stage vectors live in per-block shared memory.

    One thread per trajectory, ``_SHARED_BLOCK`` threads per block. The stage
    workspace is nine ``(n_vars, _SHARED_BLOCK)`` shared arrays (transposed so
    consecutive lanes are contiguous -> bank-conflict-free), indexed by
    ``threadIdx.x``. No scratch is passed in or written out.
    """
    body = _make_body(ode_fn, n_vars, pcoeff, icoeff, dcoeff)
    shape = (n_vars, _SHARED_BLOCK)

    @cuda.jit
    def kernel(
        y0,
        times,
        params,
        dt0,
        rtol,
        atol,
        max_steps,
        weights,
        hist,
        accepted_out,
        rejected_out,
        loop_out,
    ):
        i = cuda.grid(1)
        tx = cuda.threadIdx.x
        if i >= y0.shape[1]:
            return
        y = cuda.shared.array(shape=shape, dtype=np.float64)
        u = cuda.shared.array(shape=shape, dtype=np.float64)
        k1 = cuda.shared.array(shape=shape, dtype=np.float64)
        k2 = cuda.shared.array(shape=shape, dtype=np.float64)
        k3 = cuda.shared.array(shape=shape, dtype=np.float64)
        k4 = cuda.shared.array(shape=shape, dtype=np.float64)
        k5 = cuda.shared.array(shape=shape, dtype=np.float64)
        k6 = cuda.shared.array(shape=shape, dtype=np.float64)
        k7 = cuda.shared.array(shape=shape, dtype=np.float64)
        body(
            y0,
            times,
            params,
            dt0,
            rtol,
            atol,
            max_steps,
            weights,
            hist,
            accepted_out,
            rejected_out,
            loop_out,
            y,
            u,
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            k7,
            i,
            tx,
        )

    return kernel


def prepare_solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    backend="auto",
):
    del batch_size
    y0_arr, times, params_arr, dt0 = _normalize_inputs(
        y0, t_span, params, first_step, solver_name="Tsit5"
    )
    n, n_vars = y0_arr.shape
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    weights_arr = build_error_weights(error_weights, n, n_vars)

    workspace = get_workspace(_WORKSPACE_CACHE, n, n_vars, n_save, n_params)
    # State/weights live transposed (n_vars, n) on the device; params keep the
    # (n, n_params) row layout consumed by the callback.
    workspace.y0_dev.copy_to_device(np.ascontiguousarray(y0_arr.T))
    workspace.times_dev.copy_to_device(times)
    workspace.params_dev.copy_to_device(params_arr)
    workspace.weights_dev.copy_to_device(np.ascontiguousarray(weights_arr.T))

    uses_shared = _use_shared_backend(n, n_vars, backend)
    if uses_shared:
        kernel = _make_shared_kernel(ode_fn, n_vars, pcoeff, icoeff, dcoeff)
        threads = _SHARED_BLOCK
    else:
        kernel = _make_kernel(ode_fn, n_vars, pcoeff, icoeff, dcoeff)
        threads = 128
    blocks = (n + threads - 1) // threads
    return PreparedSolve(
        kernel=kernel,
        workspace=workspace,
        dt0=np.float64(dt0),
        rtol=np.float64(rtol),
        atol=np.float64(atol),
        max_steps=np.int32(max_steps),
        blocks=blocks,
        threads=threads,
        uses_shared=uses_shared,
    )


def run_prepared(prepared: PreparedSolve, *, return_stats=False, copy_solution=True):
    workspace = prepared.workspace
    # The shared-memory kernel allocates its stage workspace in shared memory and
    # takes no scratch arrays; the global kernel takes the nine work buffers.
    scratch = () if prepared.uses_shared else tuple(workspace.work)
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
    accepted_steps = workspace.accepted_dev.copy_to_host()
    rejected_steps = workspace.rejected_dev.copy_to_host()
    loop_steps = workspace.loop_dev.copy_to_host()
    return solution, numpy_stats(accepted_steps, rejected_steps, loop_steps)


@functools.cache
def _make_jax_launch(
    ode_fn,
    n: int,
    n_vars: int,
    n_save: int,
    n_params: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
):
    kernel = _make_kernel(ode_fn, n_vars, pcoeff, icoeff, dcoeff)
    f64_2d = types.float64[:, ::1]
    f64_1d = types.float64[::1]
    i32_1d = types.int32[::1]
    argtypes = (
        f64_2d,
        f64_1d,
        f64_2d,
        types.float64,
        types.float64,
        types.float64,
        types.int32,
        f64_2d,
        types.float64[:, :, ::1],
        i32_1d,
        i32_1d,
        i32_1d,
    ) + (f64_2d,) * 9
    threads = 128
    blocks = (n + threads - 1) // threads
    return make_launch(kernel, argtypes, grid=blocks, block=threads)


@functools.cache
def _make_shared_jax_launch(
    ode_fn,
    n: int,
    n_vars: int,
    n_save: int,
    n_params: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
):
    kernel = _make_shared_kernel(ode_fn, n_vars, pcoeff, icoeff, dcoeff)
    f64_2d = types.float64[:, ::1]
    f64_1d = types.float64[::1]
    i32_1d = types.int32[::1]
    # No scratch arrays: the shared kernel keeps its stage workspace on chip.
    argtypes = (
        f64_2d,
        f64_1d,
        f64_2d,
        types.float64,
        types.float64,
        types.float64,
        types.int32,
        f64_2d,
        types.float64[:, :, ::1],
        i32_1d,
        i32_1d,
        i32_1d,
    )
    threads = _SHARED_BLOCK
    blocks = (n + threads - 1) // threads
    return make_launch(kernel, argtypes, grid=blocks, block=threads)


def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    backend="auto",
):
    """JAX-callable Tsit5 custom-kernel solve.

    The solve is an XLA custom call into the numba-cuda kernel and is opaque to
    autodiff.

    ``error_weights`` is an optional per-component weight array, shape
    ``(n_vars,)`` or ``(N, n_vars)``, applied in the weighted RMS step-size
    error norm; a weight of 0 excludes that component from step-size control.

    ``pcoeff``/``icoeff``/``dcoeff`` are the PID step-controller gains; the
    default ``(0, 1, 0)`` is the classic I-controller.

    ``backend`` selects the kernel: ``"auto"`` (default) uses the shared-memory
    kernel for small ensembles at low dimension and the transposed-global kernel
    otherwise; ``"shared"`` and ``"global"`` force a backend (``"shared"`` errors
    if ``n_vars`` exceeds the shared-memory capacity).
    """

    def solve_impl(y0_arr, t_span_arr, params_arr):
        return _solve_impl(
            ode_fn,
            y0_arr,
            t_span_arr,
            params_arr,
            batch_size=batch_size,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            return_stats=return_stats,
            error_weights=error_weights,
            pcoeff=pcoeff,
            icoeff=icoeff,
            dcoeff=dcoeff,
            backend=backend,
        )

    return make_custom_vmap_solver(
        solve_impl,
        return_stats=return_stats,
        stats_postprocess=per_trajectory_stats_postprocess,
    )(y0, t_span, params)


def _solve_impl(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    backend="auto",
):
    del batch_size
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    dt0 = initial_step(times, first_step)
    weights_arr = jnp.asarray(build_error_weights(error_weights, n, n_vars))
    # State/stage/weights are transposed (n_vars, n) so the kernel's warp
    # accesses are coalesced; XLA materializes the transpose as a C-contiguous
    # operand. hist keeps the (n, n_save, n_vars) output layout.
    y0_t = y0_arr.T
    weights_t = weights_arr.T

    inputs = (y0_t, times, params_arr, weights_t)
    input_kinds = (
        ABI_ARRAY,
        ABI_ARRAY,
        ABI_ARRAY,
        ABI_SCALAR_F64,
        ABI_SCALAR_F64,
        ABI_SCALAR_F64,
        ABI_SCALAR_I32,
        ABI_ARRAY,
    )
    hist_spec = jax.ShapeDtypeStruct((n, n_save, n_vars), jnp.float64)
    int_spec = jax.ShapeDtypeStruct((n,), jnp.int32)
    if _use_shared_backend(n, n_vars, backend):
        launch = _make_shared_jax_launch(
            ode_fn, n, n_vars, n_save, n_params, pcoeff, icoeff, dcoeff
        )
        # Shared kernel keeps its stage workspace on chip: no scratch outputs.
        output_specs = (hist_spec, int_spec, int_spec, int_spec)
    else:
        launch = _make_jax_launch(
            ode_fn, n, n_vars, n_save, n_params, pcoeff, icoeff, dcoeff
        )
        work_spec = jax.ShapeDtypeStruct((n_vars, n), jnp.float64)
        output_specs = (hist_spec, int_spec, int_spec, int_spec) + (work_spec,) * 9
    result = ffi_abi_call(
        launch,
        inputs,
        output_specs,
        input_kinds=input_kinds,
        output_kinds=(ABI_ARRAY,) * len(output_specs),
        scalar_f64_values=(dt0, rtol, atol),
        scalar_i32_values=(max_steps,),
    )
    hist, accepted, rejected, loop_steps = result[:4]
    if not return_stats:
        return hist
    return hist, jax_stats(accepted, rejected, loop_steps)
