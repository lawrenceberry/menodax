"""Generic KenCarp5 custom kernel using numba-cuda."""

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
    numpy_stats,
)
from solvers._numba_common import (
    normalize_inputs as _normalize_inputs,
)

GAMMA = 41.0 / 200.0

B_SOL0 = -872700587467.0 / 9133579230613.0
B_SOL1 = 0.0
B_SOL2 = 0.0
B_SOL3 = 22348218063261.0 / 9555858737531.0
B_SOL4 = -1143369518992.0 / 8141816002931.0
B_SOL5 = -39379526789629.0 / 19018526304540.0
B_SOL6 = 32727382324388.0 / 42900044865799.0
B_SOL7 = GAMMA

B_ERR0 = B_SOL0 - (-975461918565.0 / 9796059967033.0)
B_ERR1 = 0.0
B_ERR2 = 0.0
B_ERR3 = B_SOL3 - (78070527104295.0 / 32432590147079.0)
B_ERR4 = B_SOL4 - (-548382580838.0 / 3424219808633.0)
B_ERR5 = B_SOL5 - (-33438840321285.0 / 15594753105479.0)
B_ERR6 = B_SOL6 - (3629800801594.0 / 4656183773603.0)
B_ERR7 = B_SOL7 - (4035322873751.0 / 18575991585200.0)

C1 = 41.0 / 100.0
C2 = 2935347310677.0 / 11292855782101.0
C3 = 1426016391358.0 / 7196633302097.0
C4 = 92.0 / 100.0
C5 = 24.0 / 100.0
C6 = 3.0 / 5.0
C7 = 1.0


SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 10.0
NEWTON_MAX_ITERS = 10
# Elementary I-controller exponent (-1/k, k the error order). The PID terms in
# the kernel are expressed relative to this.
EXPONENT = -1.0 / 5.0

_WORKSPACE_CACHE: dict[tuple[int, int, int, int], object] = {}


def clear_caches() -> None:
    _WORKSPACE_CACHE.clear()
    _make_kernel.cache_clear()
    _make_jax_launch.cache_clear()
    gc.collect()


@dataclass
class Workspace(NumbaWorkspace):
    u_dev: Any
    tmp_dev: Any
    weights_dev: Any


@dataclass(frozen=True)
class PreparedSolve(PreparedNumbaSolve):
    kernel: Any
    lu_solver: Any
    workspace: Workspace


def make_lu_solver(
    n_vars: int,
    *,
    precision=np.float64,
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
        u_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        tmp_dev=cuda.device_array((n, n_vars), dtype=np.float64),
        weights_dev=cuda.device_array((n, n_vars), dtype=np.float64),
    )
    cache[key] = workspace
    return workspace


@cuda.jit(device=True)
def _c(stage):
    if stage == 1:
        return C1
    if stage == 2:
        return C2
    if stage == 3:
        return C3
    if stage == 4:
        return C4
    if stage == 5:
        return C5
    if stage == 6:
        return C6
    if stage == 7:
        return C7
    return 0.0


@cuda.jit(device=True)
def _b_err(stage):
    if stage == 0:
        return B_ERR0
    if stage == 1:
        return B_ERR1
    if stage == 2:
        return B_ERR2
    if stage == 3:
        return B_ERR3
    if stage == 4:
        return B_ERR4
    if stage == 5:
        return B_ERR5
    if stage == 6:
        return B_ERR6
    return B_ERR7


@cuda.jit(device=True)
def _a_explicit(stage, prev):
    if stage == 1 and prev == 0:
        return 41.0 / 100.0
    if stage == 2:
        if prev == 0:
            return 367902744464.0 / 2072280473677.0
        if prev == 1:
            return 677623207551.0 / 8224143866563.0
    if stage == 3:
        if prev == 0:
            return 1268023523408.0 / 10340822734521.0
        if prev == 2:
            return 1029933939417.0 / 13636558850479.0
    if stage == 4:
        if prev == 0:
            return 14463281900351.0 / 6315353703477.0
        if prev == 2:
            return 66114435211212.0 / 5879490589093.0
        if prev == 3:
            return -54053170152839.0 / 4284798021562.0
    if stage == 5:
        if prev == 0:
            return 14090043504691.0 / 34967701212078.0
        if prev == 2:
            return 15191511035443.0 / 11219624916014.0
        if prev == 3:
            return -18461159152457.0 / 12425892160975.0
        if prev == 4:
            return -281667163811.0 / 9011619295870.0
    if stage == 6:
        if prev == 0:
            return 19230459214898.0 / 13134317526959.0
        if prev == 2:
            return 21275331358303.0 / 2942455364971.0
        if prev == 3:
            return -38145345988419.0 / 4862620318723.0
        if prev == 4 or prev == 5:
            return -1.0 / 8.0
    if stage == 7:
        if prev == 0:
            return -19977161125411.0 / 11928030595625.0
        if prev == 2:
            return -40795976796054.0 / 6384907823539.0
        if prev == 3:
            return 177454434618887.0 / 12078138498510.0
        if prev == 4:
            return 782672205425.0 / 8267701900261.0
        if prev == 5:
            return -69563011059811.0 / 9646580694205.0
        if prev == 6:
            return 7356628210526.0 / 4942186776405.0
    return 0.0


@cuda.jit(device=True)
def _a_implicit(stage, prev):
    if stage == 1 and prev == 1:
        return GAMMA
    if stage == 2:
        if prev == 0:
            return 41.0 / 400.0
        if prev == 1:
            return -567603406766.0 / 11931857230679.0
        if prev == 2:
            return GAMMA
    if stage == 3:
        if prev == 0:
            return 683785636431.0 / 9252920307686.0
        if prev == 2:
            return -110385047103.0 / 1367015193373.0
        if prev == 3:
            return GAMMA
    if stage == 4:
        if prev == 0:
            return 3016520224154.0 / 10081342136671.0
        if prev == 2:
            return 30586259806659.0 / 12414158314087.0
        if prev == 3:
            return -22760509404356.0 / 11113319521817.0
        if prev == 4:
            return GAMMA
    if stage == 5:
        if prev == 0:
            return 218866479029.0 / 1489978393911.0
        if prev == 2:
            return 638256894668.0 / 5436446318841.0
        if prev == 3:
            return -1179710474555.0 / 5321154724896.0
        if prev == 4:
            return -60928119172.0 / 8023461067671.0
        if prev == 5:
            return GAMMA
    if stage == 6:
        if prev == 0:
            return 1020004230633.0 / 5715676835656.0
        if prev == 2:
            return 25762820946817.0 / 25263940353407.0
        if prev == 3:
            return -2161375909145.0 / 9755907335909.0
        if prev == 4:
            return -211217309593.0 / 5846859502534.0
        if prev == 5:
            return -4269925059573.0 / 7827059040719.0
        if prev == 6:
            return GAMMA
    if stage == 7:
        if prev == 0:
            return B_SOL0
        if prev == 1:
            return B_SOL1
        if prev == 2:
            return B_SOL2
        if prev == 3:
            return B_SOL3
        if prev == 4:
            return B_SOL4
        if prev == 5:
            return B_SOL5
        if prev == 6:
            return B_SOL6
        if prev == 7:
            return GAMMA
    return 0.0


@cuda.jit(device=True)
def _predictor_coeff(stage, prev):
    if stage == 1:
        return 1.0 if prev == 0 else 0.0
    if stage == 2:
        if prev == 0:
            return 1.0 - C2 / C1
        if prev == 1:
            return C2 / C1
    if stage == 3:
        if prev == 0:
            return 1.0 - C3 / C1
        if prev == 1:
            return C3 / C1
    if stage == 4:
        if prev == 0:
            return 1.0 - C4 / C2
        if prev == 2:
            return C4 / C2
    if stage == 5:
        if prev == 0:
            return 1.0 - C5 / C4
        if prev == 4:
            return C5 / C4
    if stage == 6:
        if prev == 0:
            return 1.0 - C6 / C4
        if prev == 4:
            return C6 / C4
    if stage == 7:
        if prev == 0:
            return 1.0 - C7 / C4
        if prev == 4:
            return C7 / C4
    return 0.0


@functools.cache
def _make_kernel(
    explicit_ode_fn,
    implicit_ode_fn,
    implicit_jac_fn,
    n_vars: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp64",
    batches_per_block="suggested",
):
    # PID step-control exponents (Soderlind). Defaults (0, 1, 0) give E1=EXPONENT
    # and E2=E3=0, recovering the elementary I-controller exactly.
    e1 = EXPONENT * (icoeff + pcoeff + dcoeff)
    e2 = -EXPONENT * (pcoeff + 2.0 * dcoeff)
    e3 = EXPONENT * dcoeff
    # Precision of the factorization / triangular solve. The Jacobian, residual
    # and convergence test stay in float64; ``lu_dtype`` only governs the shared
    # LU matrix and RHS, so fp32 changes the modified-Newton convergence rate,
    # not the converged stage solution. fp32 also halves the LU shared-memory
    # footprint.
    lu_dtype = np.float32 if lu_precision == "fp32" else np.float64
    lu_solver = make_lu_solver(
        n_vars, precision=lu_dtype, batches_per_block=batches_per_block
    )
    explicit_ode_write = make_cuda_striped_vector_writer(explicit_ode_fn, n_vars)
    implicit_ode_write = make_cuda_striped_vector_writer(implicit_ode_fn, n_vars)
    implicit_jac_device = as_cuda_device(implicit_jac_fn)

    @cuda.jit(device=True)
    def assemble_w(y, t, p, lu_buf, a_off, gamma_dt, i, lane, stride):
        # Build the ESDIRK Newton iteration matrix W = I - gamma*dt*J straight
        # into the shared LU buffer. Each lane evaluates the full implicit
        # Jacobian (cheap and wall-clock-free under SIMT lockstep) and writes a
        # disjoint row-stripe, so neither the n_vars*n_vars matrix nor its
        # transform is staged through global memory or serialized on one lane.
        values = implicit_jac_device(y[i], t, p[i])
        for row in range(lane, n_vars, stride):
            base = a_off + row * n_vars
            for col in range(n_vars):
                v = -gamma_dt * values[row][col]
                if row == col:
                    v += 1.0
                lu_buf[base + col] = lu_dtype(v)

    batches_per_block = int(lu_solver.batches_per_block)

    block_dim = as_launch_block_dim(lu_solver.block_dim)
    block_threads = block_threads_x(block_dim)
    a_size = int(lu_solver.a_size())
    b_size = int(lu_solver.b_size())
    ipiv_size = int(lu_solver.ipiv_size)
    vec_size = batches_per_block * n_vars
    stage_vec_size = batches_per_block * 8 * n_vars

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
        u_global,
        tmp_global,
    ):
        tx = cuda.threadIdx.x
        batch = tx % batches_per_block
        lane = tx // batches_per_block
        batch_lanes = (
            block_threads + batches_per_block - 1 - batch
        ) // batches_per_block
        block_start = cuda.blockIdx.x * batches_per_block
        i = block_start + batch
        a_offset = batch * n_vars * n_vars
        b_offset = batch * n_vars
        v_offset = batch * n_vars
        s_offset = batch * 8 * n_vars

        n_save = times.shape[0]
        tf = times[n_save - 1]

        smem_lu = cuda.shared.array(shape=a_size, dtype=lu_dtype)
        smem_rhs = cuda.shared.array(shape=b_size, dtype=lu_dtype)
        smem_ipiv = cuda.shared.array(shape=ipiv_size, dtype=np.int32)
        smem_info = cuda.shared.array(shape=batches_per_block, dtype=np.int32)

        # The complete per-trajectory state lives in shared memory: the working
        # vectors below and the eight-stage history (solution, explicit RHS,
        # implicit RHS) further down. The global ``u``/``tmp`` arrays are only
        # marshalling scratch for the user ODE/Jacobian callbacks (which read and
        # write global rows ``[i]``); no integration state round-trips through
        # global memory. Keeping the stage history in shared memory caps the
        # usable ODE dimension at the per-block shared-memory budget.
        smem_y = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_u = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_base = cuda.shared.array(shape=vec_size, dtype=np.float64)
        smem_stage_y = cuda.shared.array(shape=stage_vec_size, dtype=np.float64)
        smem_stage_fe = cuda.shared.array(shape=stage_vec_size, dtype=np.float64)
        smem_stage_fi = cuda.shared.array(shape=stage_vec_size, dtype=np.float64)

        smem_delta = cuda.shared.array(shape=block_threads, dtype=np.float64)
        smem_active = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_t = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_dt = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_dt_use = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_err_prev = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_err_prev2 = cuda.shared.array(shape=batches_per_block, dtype=np.float64)
        smem_save_idx = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_n_steps = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_accepted = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_rejected = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_converged = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_failed = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_accept = cuda.shared.array(shape=batches_per_block, dtype=np.int32)
        smem_newton_go = cuda.shared.array(shape=1, dtype=np.int32)
        smem_continue = cuda.shared.array(shape=1, dtype=np.int32)

        if i < y0.shape[0]:
            for j in range(lane, n_vars, batch_lanes):
                val = y0[i, j]
                smem_y[v_offset + j] = val
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
            smem_err_prev[batch] = 1.0
            smem_err_prev2[batch] = 1.0
        if tx == 0:
            smem_continue[0] = 1
        cuda.syncthreads()

        while smem_continue[0] != 0:
            if lane == 0:
                if (
                    i < y0.shape[0]
                    and smem_save_idx[batch] < n_save
                    and smem_t[batch] < tf
                    and smem_n_steps[batch] < max_steps
                ):
                    smem_active[batch] = 1
                else:
                    smem_active[batch] = 0
            cuda.syncthreads()
            active = smem_active[batch] != 0

            # Clamp the trial step to the final integration time. Accepted
            # steps may cross save times; those are filled by dense output.
            if lane == 0:
                if active:
                    dt_use = smem_dt[batch]
                    if dt_use > tf - smem_t[batch]:
                        dt_use = tf - smem_t[batch]
                    if dt_use < 1e-30:
                        dt_use = 1e-30
                    smem_dt_use[batch] = dt_use
                else:
                    smem_dt_use[batch] = dt0
                smem_failed[batch] = 0
            cuda.syncthreads()

            dt_use = smem_dt_use[batch]
            gamma_dt = GAMMA * dt_use

            # --- Stage 0: explicit/implicit RHS at the step start ----------
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    val = smem_y[v_offset + j]
                    smem_u[v_offset + j] = val
                    u_global[i, j] = val
                    smem_stage_y[s_offset + j] = val
            cuda.syncthreads()
            if active:
                explicit_ode_write(
                    u_global, smem_t[batch], params, tmp_global, i, lane, batch_lanes
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    fe0 = tmp_global[i, j]
                    smem_stage_fe[s_offset + j] = fe0
                    if not math.isfinite(fe0):
                        smem_failed[batch] = 1
            cuda.syncthreads()
            if active:
                implicit_ode_write(
                    u_global, smem_t[batch], params, tmp_global, i, lane, batch_lanes
                )
            cuda.syncthreads()
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    fi0 = tmp_global[i, j]
                    smem_stage_fi[s_offset + j] = fi0
                    if not math.isfinite(fi0):
                        smem_failed[batch] = 1
            cuda.syncthreads()

            # --- Freeze and factor the Newton matrix W once per step -------
            # KenCarp5 is an ESDIRK method: every implicit diagonal coefficient
            # equals GAMMA, so each implicit stage shares the iteration matrix
            # W = I - gamma*dt*J.  Evaluate J once at the step-start point
            # (u == y, time t) and factor W a single time; the factors in
            # smem_lu/smem_ipiv are reused by every stage solve and every Newton
            # iteration.  Inactive batches load an identity so the cooperative,
            # block-wide factorization and solves stay well defined across the
            # whole block.
            if active:
                assemble_w(
                    u_global,
                    smem_t[batch],
                    params,
                    smem_lu,
                    a_offset,
                    gamma_dt,
                    i,
                    lane,
                    batch_lanes,
                )
            else:
                for idx_local in range(lane, n_vars * n_vars, batch_lanes):
                    row = idx_local // n_vars
                    col = idx_local - row * n_vars
                    smem_lu[a_offset + idx_local] = lu_dtype(1.0 if row == col else 0.0)
            cuda.syncthreads()
            lu_solver.factorize(smem_lu, smem_ipiv, smem_info)
            cuda.syncthreads()

            # --- Implicit stages 1..7 --------------------------------------
            for stage in range(1, 8):
                t_stage = smem_t[batch] + _c(stage) * dt_use
                if active:
                    for j in range(lane, n_vars, batch_lanes):
                        base = smem_y[v_offset + j]
                        pred = 0.0
                        for prev in range(stage):
                            ae = _a_explicit(stage, prev)
                            ai = _a_implicit(stage, prev)
                            pc = _predictor_coeff(stage, prev)
                            if ae != 0.0:
                                base += (
                                    dt_use
                                    * ae
                                    * smem_stage_fe[s_offset + prev * n_vars + j]
                                )
                            if ai != 0.0:
                                base += (
                                    dt_use
                                    * ai
                                    * smem_stage_fi[s_offset + prev * n_vars + j]
                                )
                            if pc != 0.0:
                                pred += pc * smem_stage_y[s_offset + prev * n_vars + j]
                        smem_base[v_offset + j] = base
                        smem_u[v_offset + j] = pred
                if lane == 0:
                    smem_converged[batch] = 0
                cuda.syncthreads()

                # Modified-Newton iteration with the frozen factorization. The
                # trip count is data dependent per trajectory, so the loop runs
                # a fixed maximum and a block-wide flag breaks every lane out
                # together once all active batches have converged or failed,
                # keeping the cooperative solve and every barrier uniform.
                for _it in range(NEWTON_MAX_ITERS):
                    do_iter = (
                        active
                        and smem_converged[batch] == 0
                        and smem_failed[batch] == 0
                    )
                    if do_iter:
                        for j in range(lane, n_vars, batch_lanes):
                            u_global[i, j] = smem_u[v_offset + j]
                    cuda.syncthreads()
                    if do_iter:
                        implicit_ode_write(
                            u_global, t_stage, params, tmp_global, i, lane, batch_lanes
                        )
                    cuda.syncthreads()
                    if do_iter:
                        for j in range(lane, n_vars, batch_lanes):
                            res = (
                                smem_u[v_offset + j]
                                - smem_base[v_offset + j]
                                - gamma_dt * tmp_global[i, j]
                            )
                            smem_rhs[b_offset + j] = lu_dtype(res)
                    else:
                        for j in range(lane, n_vars, batch_lanes):
                            smem_rhs[b_offset + j] = lu_dtype(0.0)
                    cuda.syncthreads()
                    lu_solver.solve(smem_lu, smem_ipiv, smem_rhs)
                    cuda.syncthreads()
                    delta_acc = 0.0
                    if do_iter:
                        for j in range(lane, n_vars, batch_lanes):
                            delta = np.float64(smem_rhs[b_offset + j])
                            u_old = smem_u[v_offset + j]
                            u_new = u_old - delta
                            scale = atol + rtol * max(
                                math.fabs(u_old), math.fabs(u_new)
                            )
                            ratio = delta / scale
                            delta_acc += ratio * ratio
                            if not math.isfinite(u_new) or not math.isfinite(delta):
                                smem_failed[batch] = 1
                            smem_u[v_offset + j] = u_new
                    smem_delta[tx] = delta_acc
                    cuda.syncthreads()
                    if lane == 0 and do_iter:
                        acc = smem_delta[tx]
                        for other_lane in range(1, batch_lanes):
                            acc += smem_delta[batch + other_lane * batches_per_block]
                        delta_norm = math.sqrt(acc / n_vars)
                        if math.isnan(delta_norm):
                            smem_failed[batch] = 1
                        elif delta_norm <= 1.0:
                            smem_converged[batch] = 1
                    cuda.syncthreads()
                    if tx == 0:
                        go = 0
                        for b in range(batches_per_block):
                            if (
                                smem_active[b] != 0
                                and smem_converged[b] == 0
                                and smem_failed[b] == 0
                            ):
                                go = 1
                        smem_newton_go[0] = go
                    cuda.syncthreads()
                    if smem_newton_go[0] == 0:
                        break

                if lane == 0 and active and smem_converged[batch] == 0:
                    smem_failed[batch] = 1
                cuda.syncthreads()

                # Converged stage solution and its explicit/implicit RHS.
                if active:
                    for j in range(lane, n_vars, batch_lanes):
                        sol = smem_u[v_offset + j]
                        smem_stage_y[s_offset + stage * n_vars + j] = sol
                        u_global[i, j] = sol
                cuda.syncthreads()
                if active:
                    implicit_ode_write(
                        u_global, t_stage, params, tmp_global, i, lane, batch_lanes
                    )
                cuda.syncthreads()
                if active:
                    for j in range(lane, n_vars, batch_lanes):
                        fij = tmp_global[i, j]
                        smem_stage_fi[s_offset + stage * n_vars + j] = fij
                        if not math.isfinite(fij):
                            smem_failed[batch] = 1
                cuda.syncthreads()
                if active:
                    explicit_ode_write(
                        u_global, t_stage, params, tmp_global, i, lane, batch_lanes
                    )
                cuda.syncthreads()
                if active:
                    for j in range(lane, n_vars, batch_lanes):
                        fej = tmp_global[i, j]
                        smem_stage_fe[s_offset + stage * n_vars + j] = fej
                        if not math.isfinite(fej):
                            smem_failed[batch] = 1
                cuda.syncthreads()

            # --- Embedded error estimate and PID step control --------------
            err_acc = 0.0
            if active:
                for j in range(lane, n_vars, batch_lanes):
                    y_new = smem_stage_y[s_offset + 7 * n_vars + j]
                    err_est = 0.0
                    for stage in range(8):
                        err_est += (
                            dt_use
                            * _b_err(stage)
                            * (
                                smem_stage_fe[s_offset + stage * n_vars + j]
                                + smem_stage_fi[s_offset + stage * n_vars + j]
                            )
                        )
                    scale = atol + rtol * max(
                        math.fabs(smem_y[v_offset + j]), math.fabs(y_new)
                    )
                    r = weights[i, j] * err_est / scale
                    err_acc += r * r
                    if not math.isfinite(y_new) or not math.isfinite(err_est):
                        smem_failed[batch] = 1
            smem_delta[tx] = err_acc
            cuda.syncthreads()

            if lane == 0:
                if active:
                    acc = smem_delta[tx]
                    for other_lane in range(1, batch_lanes):
                        acc += smem_delta[batch + other_lane * batches_per_block]
                    err_norm = math.sqrt(acc / n_vars)
                    accept = (
                        err_norm <= 1.0
                        and (not math.isnan(err_norm))
                        and smem_failed[batch] == 0
                    )
                    if (
                        smem_failed[batch] != 0
                        or math.isnan(err_norm)
                        or err_norm > 1e18
                    ):
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
                    smem_dt[batch] = dt_use * factor
                    smem_accept[batch] = 1 if accept else 0
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
                    c0 = theta * (
                        (
                            -9257016797708.0 / 5021505065439.0 * theta
                            + 43486358583215.0 / 12773830924787.0
                        )
                        * theta
                        - 17674230611817.0 / 10670229744614.0
                    )
                    c1 = 0.0
                    c2 = 0.0
                    c3 = theta * (
                        (
                            26096422576131.0 / 11239449250142.0 * theta
                            - 91478233927265.0 / 11067650958493.0
                        )
                        * theta
                        + 65168852399939.0 / 7868540260826.0
                    )
                    c4 = theta * (
                        (
                            92396832856987.0 / 20362823103730.0 * theta
                            - 79368583304911.0 / 10890268929626.0
                        )
                        * theta
                        + 15494834004392.0 / 5936557850923.0
                    )
                    c5 = theta * (
                        (
                            30029262896817.0 / 10175596800299.0 * theta
                            - 12239297817655.0 / 9152339842473.0
                        )
                        * theta
                        - 99329723586156.0 / 26959484932159.0
                    )
                    c6 = theta * (
                        (
                            -26136350496073.0 / 3983972220547.0 * theta
                            + 115839755401235.0 / 10719374521269.0
                        )
                        * theta
                        - 19024464361622.0 / 5461577185407.0
                    )
                    c7 = theta * (
                        (
                            -5289405421727.0 / 3760307252460.0 * theta
                            + 5843115559534.0 / 2180450260947.0
                        )
                        * theta
                        - 6511271360970.0 / 6095937251113.0
                    )
                    for j in range(lane, n_vars, batch_lanes):
                        hist[i, save_idx, j] = smem_y[v_offset + j] + smem_dt_use[
                            batch
                        ] * (
                            c0
                            * (
                                smem_stage_fe[s_offset + j]
                                + smem_stage_fi[s_offset + j]
                            )
                            + c1
                            * (
                                smem_stage_fe[s_offset + n_vars + j]
                                + smem_stage_fi[s_offset + n_vars + j]
                            )
                            + c2
                            * (
                                smem_stage_fe[s_offset + 2 * n_vars + j]
                                + smem_stage_fi[s_offset + 2 * n_vars + j]
                            )
                            + c3
                            * (
                                smem_stage_fe[s_offset + 3 * n_vars + j]
                                + smem_stage_fi[s_offset + 3 * n_vars + j]
                            )
                            + c4
                            * (
                                smem_stage_fe[s_offset + 4 * n_vars + j]
                                + smem_stage_fi[s_offset + 4 * n_vars + j]
                            )
                            + c5
                            * (
                                smem_stage_fe[s_offset + 5 * n_vars + j]
                                + smem_stage_fi[s_offset + 5 * n_vars + j]
                            )
                            + c6
                            * (
                                smem_stage_fe[s_offset + 6 * n_vars + j]
                                + smem_stage_fi[s_offset + 6 * n_vars + j]
                            )
                            + c7
                            * (
                                smem_stage_fe[s_offset + 7 * n_vars + j]
                                + smem_stage_fi[s_offset + 7 * n_vars + j]
                            )
                        )
                    save_idx += 1
                if lane == 0:
                    smem_save_idx[batch] = save_idx
                for j in range(lane, n_vars, batch_lanes):
                    smem_y[v_offset + j] = smem_stage_y[s_offset + 7 * n_vars + j]
            cuda.syncthreads()

            if lane == 0 and active:
                if smem_accept[batch] != 0:
                    smem_t[batch] += dt_use
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

        if i < y0.shape[0] and lane == 0:
            accepted_out[i] = smem_accepted[batch]
            rejected_out[i] = smem_rejected[batch]
            loop_out[i] = smem_n_steps[batch]

    return kernel, lu_solver


def prepare_solve(
    explicit_ode_fn,
    implicit_ode_fn,
    implicit_jac_fn,
    y0,
    t_span,
    params,
    *,
    lu_precision: str = "fp64",
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    batches_per_block="suggested",
):
    y0_arr, times, params_arr, dt0 = _normalize_inputs(
        y0, t_span, params, first_step, solver_name="KenCarp5"
    )
    n, n_vars = y0_arr.shape
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    weights_arr = build_error_weights(error_weights, n, n_vars)

    workspace = get_workspace(_WORKSPACE_CACHE, n, n_vars, n_save, n_params)
    copy_workspace_inputs(workspace, y0_arr, times, params_arr)
    workspace.weights_dev.copy_to_device(weights_arr)

    kernel, lu_solver = _make_kernel(
        explicit_ode_fn,
        implicit_ode_fn,
        implicit_jac_fn,
        n_vars,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
    )
    threads = as_launch_block_dim(lu_solver.block_dim)
    batches_per_block = int(lu_solver.batches_per_block)
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
        workspace.u_dev,
        workspace.tmp_dev,
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
    explicit_ode_fn,
    implicit_ode_fn,
    implicit_jac_fn,
    n: int,
    n_vars: int,
    n_save: int,
    n_params: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp64",
    batches_per_block="suggested",
):
    kernel, lu_solver = _make_kernel(
        explicit_ode_fn,
        implicit_ode_fn,
        implicit_jac_fn,
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
    f64_3d = types.float64[:, :, ::1]
    argtypes = (
        f64_2d,
        f64_1d,
        f64_2d,
        types.float64,
        types.float64,
        types.float64,
        types.int32,
        f64_2d,
        f64_3d,
        i32_1d,
        i32_1d,
        i32_1d,
        f64_2d,
        f64_2d,
    )
    threads = as_launch_block_dim(lu_solver.block_dim)
    batches_per_block = int(lu_solver.batches_per_block)
    blocks = (n + batches_per_block - 1) // batches_per_block
    return make_launch(kernel, argtypes, grid=blocks, block=threads)


def solve(
    explicit_ode_fn,
    implicit_ode_fn,
    implicit_jac_fn,
    y0,
    t_span,
    params,
    *,
    lu_precision: str = "fp64",
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    batches_per_block="suggested",
):
    """JAX-callable KenCarp5 custom-kernel solve.

    ``lu_precision`` (``"fp32"`` or ``"fp64"``) selects the precision of the
    per-step LU factorization and triangular solves. The Jacobian, residual and
    convergence test stay in float64, so fp32 only affects the modified-Newton
    convergence rate (not the converged solution) while halving the LU
    shared-memory footprint.

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
    shared-memory footprint. The full per-trajectory state -- including the
    eight-stage history -- lives in shared memory, so the usable ODE dimension
    is bounded by the per-block shared-memory budget.
    """

    def solve_impl(y0_arr, t_span_arr, params_arr):
        return _solve_impl(
            explicit_ode_fn,
            implicit_ode_fn,
            implicit_jac_fn,
            y0_arr,
            t_span_arr,
            params_arr,
            lu_precision=lu_precision,
            rtol=rtol,
            atol=atol,
            first_step=first_step,
            max_steps=max_steps,
            return_stats=return_stats,
            error_weights=error_weights,
            pcoeff=pcoeff,
            icoeff=icoeff,
            dcoeff=dcoeff,
            batches_per_block=batches_per_block,
        )

    return make_custom_vmap_solver(
        solve_impl,
        return_stats=return_stats,
        stats_postprocess=per_trajectory_stats_postprocess,
    )(y0, t_span, params)


def _solve_impl(
    explicit_ode_fn,
    implicit_ode_fn,
    implicit_jac_fn,
    y0,
    t_span,
    params,
    *,
    lu_precision: str = "fp64",
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
    pcoeff=0.0,
    icoeff=1.0,
    dcoeff=0.0,
    batches_per_block="suggested",
):
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    dt0 = initial_step(times, first_step)
    weights_arr = jnp.asarray(build_error_weights(error_weights, n, n_vars))

    launch = _make_jax_launch(
        explicit_ode_fn,
        implicit_ode_fn,
        implicit_jac_fn,
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
    output_specs = (
        hist_spec,
        int_spec,
        int_spec,
        int_spec,
        work_spec,
        work_spec,
    )
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
