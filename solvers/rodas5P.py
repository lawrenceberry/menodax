"""Cooperative packed-batch Rodas5P custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numba import cuda, types
from nvmath.device import LUPivotSolver

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
    as_cuda_device,
    as_launch_block_dim,
    block_threads_x,
    build_error_weights,
    copy_workspace_inputs,
    initial_step,
    jax_stats,
    make_cuda_striped_vector_writer,
    make_cuda_vector_writer,
    make_cuda_zero_vector_writer,
    numpy_stats,
)
from solvers._numba_common import (
    normalize_inputs as _normalize_inputs,
)

# fmt: off
# Rodas5P W-transformed coefficients (Steinebach 2023, BIT 63:27).
# Matches the Julia Rodas5P / GPURodas5P tableau. Stages 6-8 reuse the FSAL
# accumulation (u += k6, u += k7) so no A7/A8 rows are needed.
GAMMA = 0.21193756319429014

A21 = 3.0
A31 = 2.849394379747939
A32 = 0.45842242204463923
A41 = -6.954028509809101
A42 = 2.489845061869568
A43 = -10.358996098473584
A51 = 2.8029986275628964
A52 = 0.5072464736228206
A53 = -0.3988312541770524
A54 = -0.04721187230404641
A61 = -7.502846399306121
A62 = 2.561846144803919
A63 = -11.627539656261098
A64 = -0.18268767659942256
A65 = 0.030198172008377946

C21 = -14.155112264123755
C31 = -17.97296035885952
C32 = -2.859693295451294
C41 = 147.12150275711716
C42 = -1.41221402718213
C43 = 71.68940251302358
C51 = 165.43517024871676
C52 = -0.4592823456491126
C53 = 42.90938336958603
C54 = -5.961986721573306
C61 = 24.854864614690072
C62 = -3.0009227002832186
C63 = 47.4931110020768
C64 = 5.5814197821558125
C65 = -0.6610691825249471
C71 = 30.91273214028599
C72 = -3.1208243349937974
C73 = 77.79954646070892
C74 = 34.28646028294783
C75 = -19.097331116725623
C76 = -28.087943162872662
C81 = 37.80277123390563
C82 = -3.2571969029072276
C83 = 112.26918849496327
C84 = 66.9347231244047
C85 = -40.06618937091002
C86 = -54.66780262877968
C87 = -9.48861652309627

C2 = 0.6358126895828704
C3 = 0.4095798393397535
C4 = 0.9769306725060716
C5 = 0.4288403609558664
# C6 = C7 = C8 = 1.0

# Time-derivative ("d_i") coefficients. The stage RHS carries an additional
# dt*d_i*df/dt term (Hairer-Wanner II.7); without it, the method drops below
# order 5 when df/dt is nonzero. d6 = d7 = d8 = 0.
D1 = GAMMA  # = 0.21193756319429014
D2 = -0.42387512638858027
D3 = -0.3384627126235924
D4 =  1.8046452872882734
D5 =  2.325825639765069
# fmt: on


SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 6.0
# Elementary I-controller exponent (-1/k, k the error order). The PID terms in
# the kernel are expressed relative to this.
EXPONENT = -1.0 / 6.0

_WORKSPACE_CACHE: dict[tuple[int, int, int, int], object] = {}


@dataclass
class Workspace(NumbaWorkspace):
    work: list[Any]
    dT_dev: Any
    weights_dev: Any


@dataclass(frozen=True)
class PreparedSolve(PreparedNumbaSolve):
    kernel: Any
    lu_solver: Any
    workspace: Workspace


def make_lu_solver(
    n_vars: int,
    *,
    precision=np.float32,
    batches_per_block="suggested",
    block_dim="suggested",
):
    return LUPivotSolver(
        size=(n_vars, n_vars, 1),
        precision=precision,
        execution="Block",
        arrangement=("row_major", "row_major"),
        batches_per_block=batches_per_block,
        block_dim=block_dim,
    )


def get_workspace(
    cache: dict, n: int, n_vars: int, n_save: int, n_params: int
) -> Workspace:
    key = (n, n_vars, n_save, n_params)
    workspace = cache.get(key)
    if workspace is not None:
        return workspace

    workspace = Workspace(
        y0_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        times_dev=cuda.device_array(n_save, dtype=np.float64),
        params_dev=cuda.device_array((n, n_params), dtype=np.float64),
        hist_dev=cuda.device_array((n, n_save, n_vars), dtype=np.float64),
        accepted_dev=cuda.device_array(n, dtype=np.int32),
        rejected_dev=cuda.device_array(n, dtype=np.int32),
        loop_dev=cuda.device_array(n, dtype=np.int32),
        work=[cuda.device_array((n, n_vars), dtype=np.float64) for _ in range(10)],
        dT_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        weights_dev=cuda.device_array((n, n_vars), dtype=np.float64),
    )
    cache[key] = workspace
    return workspace


@functools.cache
def _make_kernel(
    ode_fn,
    jac_fn,
    time_jac_fn,
    n_vars: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp32",
    batches_per_block="suggested",
):
    # PID step-control exponents (Soderlind). Defaults (0, 1, 0) give E1=EXPONENT
    # and E2=E3=0, recovering the elementary I-controller exactly.
    e1 = EXPONENT * (icoeff + pcoeff + dcoeff)
    e2 = -EXPONENT * (pcoeff + 2.0 * dcoeff)
    e3 = EXPONENT * dcoeff
    # Precision of the LU factorisation and triangular solves. The state, ODE
    # right-hand side, Jacobian and error estimate are always float64; lu_dtype
    # governs only the shared LU matrix and RHS. The Rosenbrock--Wanner (W)
    # order conditions retain full order under an approximate Jacobian, so an
    # FP32 factorisation does not reduce the method's order. Defaults to fp32,
    # the historical kernel behaviour; fp64 is available for ill-conditioned
    # systems where the FP32 factorisation degrades the step-size control.
    lu_dtype = np.float32 if lu_precision == "fp32" else np.float64
    lu_solver = make_lu_solver(
        n_vars, precision=lu_dtype, batches_per_block=batches_per_block
    )
    ode_write = make_cuda_striped_vector_writer(ode_fn, n_vars)
    jac_device = as_cuda_device(jac_fn)

    @cuda.jit(device=True)
    def assemble_lu(y, t, p, lu_buf, a_off, dtgamma_inv, i, lane, stride):
        # Build the Rosenbrock--Wanner iteration matrix M = 1/(h*gamma)*I - J
        # straight into the shared LU buffer. Evaluating the Jacobian and
        # writing M here (rather than staging J through a global array and
        # reading it back) avoids a per-step global-memory round-trip of the
        # full n_vars*n_vars matrix. Each lane writes a disjoint row-stripe so
        # the O(n_vars^2) transform/write is shared across the batch's lanes.
        values = jac_device(y[i], t, p[i])
        for row in range(lane, n_vars, stride):
            base = a_off + row * n_vars
            for col in range(n_vars):
                v = values[row][col]
                if row == col:
                    lu_buf[base + col] = lu_dtype(dtgamma_inv - v)
                else:
                    lu_buf[base + col] = lu_dtype(-v)

    time_jac_write = (
        make_cuda_zero_vector_writer(n_vars)
        if time_jac_fn is None
        else make_cuda_vector_writer(time_jac_fn, n_vars)
    )
    batches_per_block = int(lu_solver.batches_per_block)

    block_dim = as_launch_block_dim(lu_solver.block_dim)
    block_threads = block_threads_x(block_dim)
    vec_size = batches_per_block * n_vars
    a_size = int(lu_solver.a_size())
    b_size = int(lu_solver.b_size())
    ipiv_size = int(lu_solver.ipiv_size)

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
        y_global,
        u_global,
        work_global,
        dT_global,
    ):
        tx = cuda.threadIdx.x
        batch = tx % batches_per_block
        lane = tx // batches_per_block
        batch_lanes = (
            block_threads + batches_per_block - 1 - batch
        ) // batches_per_block
        block_start = cuda.blockIdx.x * batches_per_block
        i = block_start + batch

        n_save = times.shape[0]
        tf = times[n_save - 1]
        v_offset = batch * n_vars
        a_offset = batch * n_vars * n_vars
        b_offset = batch * n_vars

        smem_lu = cuda.shared.array(shape=a_size, dtype=lu_dtype)
        smem_rhs = cuda.shared.array(shape=b_size, dtype=lu_dtype)
        smem_ipiv = cuda.shared.array(shape=ipiv_size, dtype=np.int32)
        smem_info = cuda.shared.array(shape=batches_per_block, dtype=np.int32)

        smem_y = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_u = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k1 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k2 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k3 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k4 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k5 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k6 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k7 = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_k8 = cuda.shared.array(shape=vec_size, dtype=np.float64)

        smem_err = cuda.shared.array(shape=block_threads, dtype=np.float64)
        smem_err_prev = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_err_prev2 = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_t = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_dt = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_dt_use = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_inv_dt = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_t_end = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_save_idx = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_n_steps = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_accepted = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_rejected = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_accept = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_continue = cuda.shared.array(shape=1, dtype=np.int32)

        if i < y0.shape[0]:
            for j in range(lane, n_vars, batch_lanes):
                val = y0[i, j]
                smem_y[v_offset + j] = val
                y_global[i, j] = val
                hist[i, 0, j] = val
        if tx < batches_per_block:
            if i < y0.shape[0]:
                smem_t[batch] = times[0]
                smem_save_idx[batch] = 1
            else:
                smem_t[batch] = tf
                smem_save_idx[batch] = n_save
            smem_dt[batch] = dt0
            smem_n_steps[batch] = 0
            smem_accepted[batch] = 0
            smem_rejected[batch] = 0
            smem_accept[batch] = 0
            smem_err_prev[batch] = 1.0
            smem_err_prev2[batch] = 1.0
        if tx == 0:
            smem_continue[0] = 1
        cuda.syncthreads()

        while smem_continue[0] != 0:
            active = (
                i < y0.shape[0]
                and smem_save_idx[batch] < n_save
                and smem_t[batch] < tf
                and smem_n_steps[batch] < max_steps
            )

            if lane == 0:
                if active:
                    dt_use = smem_dt[batch]
                    if dt_use > tf - smem_t[batch]:
                        dt_use = tf - smem_t[batch]
                    if dt_use < 1e-30:
                        dt_use = 1e-30
                    smem_dt_use[batch] = dt_use
                    smem_inv_dt[batch] = 1.0 / dt_use
                    smem_t_end[batch] = smem_t[batch] + dt_use
                else:
                    smem_dt_use[batch] = dt0
                    smem_inv_dt[batch] = 1.0 / dt0
                    smem_t_end[batch] = smem_t[batch]
            cuda.syncthreads()

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    y_global[i, j] = smem_y[v_offset + j]
            cuda.syncthreads()

            dtgamma_inv = 1.0 / (smem_dt_use[batch] * GAMMA)
            if active:
                assemble_lu(
                    y_global,
                    smem_t[batch],
                    params,
                    smem_lu,
                    a_offset,
                    dtgamma_inv,
                    i,
                    lane,
                    batch_lanes,
                )
                if lane == 0:
                    time_jac_write(y_global, smem_t[batch], params, dT_global, i)
            else:
                for idx_local in range(lane, n_vars * n_vars, batch_lanes):
                    row = idx_local // n_vars
                    col = idx_local - row * n_vars
                    smem_lu[a_offset + idx_local] = 1.0 if row == col else 0.0
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = 0.0
            cuda.syncthreads()
            lu_solver.factorize(smem_lu, smem_ipiv, smem_info)
            cuda.syncthreads()

            if active:
                ode_write(
                    y_global, smem_t[batch], params, work_global, i, lane, batch_lanes
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = lu_dtype(
                        work_global[i, j] + smem_dt_use[batch] * D1 * dT_global[i, j]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k1[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = (
                        smem_y[v_offset + j] + A21 * smem_k1[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if active:
                ode_write(
                    u_global,
                    smem_t[batch] + C2 * smem_dt_use[batch],
                    params,
                    work_global,
                    i,
                    lane,
                    batch_lanes,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = lu_dtype(
                        work_global[i, j]
                        + smem_dt_use[batch] * D2 * dT_global[i, j]
                        + C21 * smem_k1[v_offset + j] * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k2[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = smem_y[v_offset + j] + (
                        A31 * smem_k1[v_offset + j] + A32 * smem_k2[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if active:
                ode_write(
                    u_global,
                    smem_t[batch] + C3 * smem_dt_use[batch],
                    params,
                    work_global,
                    i,
                    lane,
                    batch_lanes,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = lu_dtype(
                        work_global[i, j]
                        + smem_dt_use[batch] * D3 * dT_global[i, j]
                        + (C31 * smem_k1[v_offset + j] + C32 * smem_k2[v_offset + j])
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k3[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = smem_y[v_offset + j] + (
                        A41 * smem_k1[v_offset + j]
                        + A42 * smem_k2[v_offset + j]
                        + A43 * smem_k3[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if active:
                ode_write(
                    u_global,
                    smem_t[batch] + C4 * smem_dt_use[batch],
                    params,
                    work_global,
                    i,
                    lane,
                    batch_lanes,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = lu_dtype(
                        work_global[i, j]
                        + smem_dt_use[batch] * D4 * dT_global[i, j]
                        + (
                            C41 * smem_k1[v_offset + j]
                            + C42 * smem_k2[v_offset + j]
                            + C43 * smem_k3[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k4[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = smem_y[v_offset + j] + (
                        A51 * smem_k1[v_offset + j]
                        + A52 * smem_k2[v_offset + j]
                        + A53 * smem_k3[v_offset + j]
                        + A54 * smem_k4[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if active:
                ode_write(
                    u_global,
                    smem_t[batch] + C5 * smem_dt_use[batch],
                    params,
                    work_global,
                    i,
                    lane,
                    batch_lanes,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = lu_dtype(
                        work_global[i, j]
                        + smem_dt_use[batch] * D5 * dT_global[i, j]
                        + (
                            C51 * smem_k1[v_offset + j]
                            + C52 * smem_k2[v_offset + j]
                            + C53 * smem_k3[v_offset + j]
                            + C54 * smem_k4[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k5[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_u[v_offset + j] = smem_y[v_offset + j] + (
                        A61 * smem_k1[v_offset + j]
                        + A62 * smem_k2[v_offset + j]
                        + A63 * smem_k3[v_offset + j]
                        + A64 * smem_k4[v_offset + j]
                        + A65 * smem_k5[v_offset + j]
                    )
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()
            if active:
                ode_write(
                    u_global,
                    smem_t_end[batch],
                    params,
                    work_global,
                    i,
                    lane,
                    batch_lanes,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = lu_dtype(
                        work_global[i, j]
                        + (
                            C61 * smem_k1[v_offset + j]
                            + C62 * smem_k2[v_offset + j]
                            + C63 * smem_k3[v_offset + j]
                            + C64 * smem_k4[v_offset + j]
                            + C65 * smem_k5[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k6[v_offset + j] = np.float64(smem_rhs[b_offset + j])
                    smem_u[v_offset + j] += smem_k6[v_offset + j]
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()

            if active:
                ode_write(
                    u_global,
                    smem_t_end[batch],
                    params,
                    work_global,
                    i,
                    lane,
                    batch_lanes,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = lu_dtype(
                        work_global[i, j]
                        + (
                            C71 * smem_k1[v_offset + j]
                            + C72 * smem_k2[v_offset + j]
                            + C73 * smem_k3[v_offset + j]
                            + C74 * smem_k4[v_offset + j]
                            + C75 * smem_k5[v_offset + j]
                            + C76 * smem_k6[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k7[v_offset + j] = np.float64(smem_rhs[b_offset + j])
                    smem_u[v_offset + j] += smem_k7[v_offset + j]
                    u_global[i, j] = smem_u[v_offset + j]
            cuda.syncthreads()

            if active:
                ode_write(
                    u_global,
                    smem_t_end[batch],
                    params,
                    work_global,
                    i,
                    lane,
                    batch_lanes,
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    smem_rhs[b_offset + j] = lu_dtype(
                        work_global[i, j]
                        + (
                            C81 * smem_k1[v_offset + j]
                            + C82 * smem_k2[v_offset + j]
                            + C83 * smem_k3[v_offset + j]
                            + C84 * smem_k4[v_offset + j]
                            + C85 * smem_k5[v_offset + j]
                            + C86 * smem_k6[v_offset + j]
                            + C87 * smem_k7[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
            cuda.syncthreads()
            lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
            cuda.syncthreads()
            if i < y0.shape[0]:
                for j in range(lane, n_vars, batch_lanes):
                    smem_k8[v_offset + j] = np.float64(smem_rhs[b_offset + j])

            err_local = 0.0
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    y_new_j = smem_u[v_offset + j] + smem_k8[v_offset + j]
                    scale = atol + rtol * max(
                        math.fabs(smem_y[v_offset + j]), math.fabs(y_new_j)
                    )
                    r = weights[i, j] * smem_k8[v_offset + j] / scale
                    err_local += r * r
            smem_err[tx] = err_local
            cuda.syncthreads()

            if lane == 0:
                for other_lane in range(1, batch_lanes):
                    smem_err[tx] += smem_err[batch + other_lane * batches_per_block]

                if active:
                    err_norm = math.sqrt(smem_err[tx] / n_vars)
                    accept = err_norm <= 1.0 and not math.isnan(err_norm)
                    smem_accept[batch] = 1 if accept else 0

                    if math.isnan(err_norm) or err_norm > 1e18:
                        safe_err = 1e18
                    elif err_norm == 0.0:
                        safe_err = 1e-18
                    else:
                        safe_err = err_norm
                    factor = (
                        SAFETY
                        * safe_err**e1
                        * smem_err_prev[batch] ** e2
                        * smem_err_prev2[batch] ** e3
                    )
                    # Advance the PID error history only on accepted steps.
                    if accept:
                        smem_err_prev2[batch] = smem_err_prev[batch]
                        smem_err_prev[batch] = safe_err
                    if factor < FACTOR_MIN:
                        factor = FACTOR_MIN
                    elif factor > FACTOR_MAX:
                        factor = FACTOR_MAX
                    smem_dt[batch] = smem_dt_use[batch] * factor
                else:
                    smem_accept[batch] = 0
            cuda.syncthreads()

            if smem_accept[batch] != 0:
                t_old = smem_t[batch]
                t_new = t_old + smem_dt_use[batch]
                save_idx = smem_save_idx[batch]
                while save_idx < n_save and times[save_idx] <= t_new + 1e-12 * max(
                    1.0, math.fabs(times[save_idx])
                ):
                    theta = (times[save_idx] - t_old) / smem_dt_use[batch]
                    theta1 = 1.0 - theta
                    for j in range(lane, n_vars, batch_lanes):
                        h1 = (
                            25.948786856663858 * smem_k1[v_offset + j]
                            - 2.5579724845846235 * smem_k2[v_offset + j]
                            + 10.433815404888879 * smem_k3[v_offset + j]
                            - 2.3679251022685204 * smem_k4[v_offset + j]
                            + 0.524948541321073 * smem_k5[v_offset + j]
                            + 1.1241088310450404 * smem_k6[v_offset + j]
                            + 0.4272876194431874 * smem_k7[v_offset + j]
                            - 0.17202221070155493 * smem_k8[v_offset + j]
                        )
                        h2 = (
                            -9.91568850695171 * smem_k1[v_offset + j]
                            - 0.9689944594115154 * smem_k2[v_offset + j]
                            + 3.0438037242978453 * smem_k3[v_offset + j]
                            - 24.495224566215796 * smem_k4[v_offset + j]
                            + 20.176138334709044 * smem_k5[v_offset + j]
                            + 15.98066361424651 * smem_k6[v_offset + j]
                            - 6.789040303419874 * smem_k7[v_offset + j]
                            - 6.710236069923372 * smem_k8[v_offset + j]
                        )
                        h3 = (
                            11.419903575922262 * smem_k1[v_offset + j]
                            + 2.8879645146136994 * smem_k2[v_offset + j]
                            + 72.92137995996029 * smem_k3[v_offset + j]
                            + 80.12511834622643 * smem_k4[v_offset + j]
                            - 52.072871366152654 * smem_k5[v_offset + j]
                            - 59.78993625266729 * smem_k6[v_offset + j]
                            - 0.15582684282751913 * smem_k7[v_offset + j]
                            + 4.883087185713722 * smem_k8[v_offset + j]
                        )
                        y_new_j = smem_u[v_offset + j] + smem_k8[v_offset + j]
                        hist[i, save_idx, j] = theta1 * smem_y[
                            v_offset + j
                        ] + theta * (
                            y_new_j + theta1 * (h1 + theta * (h2 + theta * h3))
                        )
                    save_idx += 1
                if lane == 0:
                    smem_save_idx[batch] = save_idx
                for j in range(lane, n_vars, batch_lanes):
                    smem_y[v_offset + j] = smem_u[v_offset + j] + smem_k8[v_offset + j]
            cuda.syncthreads()

            if lane == 0 and active:
                if smem_accept[batch] != 0:
                    smem_t[batch] += smem_dt_use[batch]
                    smem_accepted[batch] += 1
                else:
                    smem_rejected[batch] += 1
                smem_n_steps[batch] += 1
            cuda.syncthreads()

            if tx == 0:
                keep_going = 0
                for b in range(batches_per_block):
                    bi = block_start + b
                    if (
                        bi < y0.shape[0]
                        and smem_save_idx[b] < n_save
                        and smem_t[b] < tf
                        and smem_n_steps[b] < max_steps
                    ):
                        keep_going = 1
                smem_continue[0] = keep_going
            cuda.syncthreads()

        if tx < batches_per_block and i < y0.shape[0]:
            accepted_out[i] = smem_accepted[batch]
            rejected_out[i] = smem_rejected[batch]
            loop_out[i] = smem_n_steps[batch]

    return kernel, lu_solver


def prepare_solve(
    ode_fn,
    jac_fn,
    y0,
    t_span,
    params,
    *,
    time_jac_fn=None,
    batch_size=None,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    lu_precision: str = "fp32",
    batches_per_block="suggested",
):
    del batch_size
    y0_arr, times, params_arr, dt0 = _normalize_inputs(
        y0, t_span, params, first_step, solver_name="Rodas5"
    )
    n, n_vars = y0_arr.shape
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    weights_arr = build_error_weights(error_weights, n, n_vars)

    workspace = get_workspace(_WORKSPACE_CACHE, n, n_vars, n_save, n_params)
    copy_workspace_inputs(workspace, y0_arr, times, params_arr)
    workspace.weights_dev.copy_to_device(weights_arr)

    kernel, lu_solver = _make_kernel(
        ode_fn,
        jac_fn,
        time_jac_fn,
        n_vars,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
    )
    batches_per_block = int(lu_solver.batches_per_block)
    threads = as_launch_block_dim(lu_solver.block_dim)
    blocks = (n + batches_per_block - 1) // batches_per_block

    return PreparedSolve(
        kernel=kernel,
        lu_solver=lu_solver,
        workspace=workspace,
        dt0=np.float64(dt0),
        rtol=np.float64(rtol),
        atol=np.float64(atol),
        max_steps=np.int32(max_steps),
        blocks=blocks,
        threads=threads,
    )


def run_prepared(prepared: PreparedSolve, *, return_stats=False, copy_solution=True):
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
        workspace.work[0],
        workspace.work[1],
        workspace.work[2],
        workspace.dT_dev,
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
    jac_fn,
    time_jac_fn,
    n: int,
    n_vars: int,
    n_save: int,
    n_params: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp32",
    batches_per_block="suggested",
):
    kernel, lu_solver = _make_kernel(
        ode_fn,
        jac_fn,
        time_jac_fn,
        n_vars,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
    )
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
        f64_2d,
        f64_2d,
        f64_2d,
        f64_2d,
    )
    batches_per_block = int(lu_solver.batches_per_block)
    threads = as_launch_block_dim(lu_solver.block_dim)
    blocks = (n + batches_per_block - 1) // batches_per_block
    return make_launch(kernel, argtypes, grid=blocks, block=threads)


def solve(
    ode_fn,
    jac_fn,
    y0,
    t_span,
    params,
    *,
    time_jac_fn=None,
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
    lu_precision: str = "fp32",
    batches_per_block="suggested",
):
    """JAX-callable Rodas5 custom-kernel solve.

    ``time_jac_fn`` is the partial time derivative ``df/dt`` of the ODE
    right-hand side, with pure CUDA-device signature
    ``(y_row, t, p_row) -> dT_row``. Required for non-autonomous
    systems to preserve fifth-order accuracy. When ``None``, defaults to a
    zero stub — correct only for autonomous problems (``df/dt = 0``).

    ``lu_precision`` (``"fp32"`` or ``"fp64"``) selects the precision of the
    per-step LU factorisation and triangular solves. The state, right-hand
    side, Jacobian and error estimate are always float64; because the
    Rosenbrock--Wanner order conditions retain full order under an approximate
    Jacobian, the ``"fp32"`` default does not reduce the method's order while
    halving the LU shared-memory footprint and exploiting FP32 throughput.
    ``"fp64"`` is available for ill-conditioned systems where the FP32
    factorisation degrades step-size control.

    ``error_weights`` is an optional per-component weight array, shape
    ``(n_vars,)`` or ``(N, n_vars)``, applied in the weighted RMS step-size
    error norm; a weight of 0 excludes that component from step-size control.

    ``pcoeff``/``icoeff``/``dcoeff`` are the PID step-controller gains; the
    default ``(0, 1, 0)`` is the classic I-controller.

    ``batches_per_block`` sets how many trajectories are packed into (and solved
    cooperatively by) a single CUDA block; it is forwarded to the nvmath
    ``LUPivotSolver`` and the whole kernel is sized from it. The default
    ``"suggested"`` lets nvmath pick a value tuned for LU throughput; an explicit
    integer overrides that to trade occupancy against per-trajectory lanes and
    shared-memory footprint (bounded by available shared memory).
    """

    def solve_impl(y0_arr, t_span_arr, params_arr):
        return _solve_impl(
            ode_fn,
            jac_fn,
            y0_arr,
            t_span_arr,
            params_arr,
            time_jac_fn=time_jac_fn,
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
            lu_precision=lu_precision,
            batches_per_block=batches_per_block,
        )

    return make_custom_vmap_solver(
        solve_impl,
        return_stats=return_stats,
        stats_postprocess=per_trajectory_stats_postprocess,
    )(y0, t_span, params)


def _solve_impl(
    ode_fn,
    jac_fn,
    y0,
    t_span,
    params,
    *,
    time_jac_fn,
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
    lu_precision: str = "fp32",
    batches_per_block="suggested",
):
    del batch_size
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    dt0 = initial_step(times, first_step)
    weights_arr = jnp.asarray(build_error_weights(error_weights, n, n_vars))

    launch = _make_jax_launch(
        ode_fn,
        jac_fn,
        time_jac_fn,
        n,
        n_vars,
        n_save,
        n_params,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
    )
    hist_spec = jax.ShapeDtypeStruct((n, n_save, n_vars), jnp.float64)
    int_spec = jax.ShapeDtypeStruct((n,), jnp.int32)
    work_spec = jax.ShapeDtypeStruct((n, n_vars), jnp.float64)
    output_specs = (hist_spec, int_spec, int_spec, int_spec) + (work_spec,) * 4
    result = ffi_abi_call(
        launch,
        (y0_arr, times, params_arr, weights_arr),
        output_specs,
        input_kinds=(
            ABI_ARRAY,
            ABI_ARRAY,
            ABI_ARRAY,
            ABI_SCALAR_F64,
            ABI_SCALAR_F64,
            ABI_SCALAR_F64,
            ABI_SCALAR_I32,
            ABI_ARRAY,
        ),
        output_kinds=(ABI_ARRAY,) * len(output_specs),
        scalar_f64_values=(dt0, rtol, atol),
        scalar_i32_values=(max_steps,),
    )
    hist, accepted, rejected, loop_steps = result[:4]
    if not return_stats:
        return hist
    return hist, jax_stats(accepted, rejected, loop_steps)
