"""Cooperative packed-batch Rodas5P custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
from numba_cuda_mlir import cuda, types
from nvmath.device import LUPivotSolver

from solvers._jax_common import make_custom_vmap_solver, normalize_y0_params
from solvers._jax_numba_custom_call import make_launch
from solvers._numba_common import (
    SCRATCH_ARGTYPE,
    SOLVER_ARGTYPES,
    PreparedNumbaSolve,
    build_error_weights,
    copy_workspace_inputs,
    ensemble_ffi_call,
    get_workspace,
    initial_step,
    make_cuda_striped_vector_writer,
    run_kernel,
    solver_stats,
)
from solvers._numba_common import (
    normalize_inputs as _normalize_inputs,
)
from solvers._sensitivity import (
    SensitivitySpec,
    augmented_error_weights,
    augmented_y0,
    make_augmented_striped_writer,
    make_jacobian_column,
    make_second_tangent,
    make_sensitivity_solver,
    seed_table,
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


# Static shared memory a CUDA block may claim without opting in to the dynamic
# extension. The kernel's footprint is dominated by the ten stage vectors, which
# hold the *augmented* state, while nvmath sizes its batch from the LU, which
# holds only the state block -- so a joint solve has to re-fit the batch.
_SHARED_BUDGET = 48 * 1024


def _shared_bytes(lu_solver, size: int, itemsize: int) -> int:
    """Static shared memory the kernel declares for one block."""
    batches = lu_solver.batches_per_block
    threads = lu_solver.block_dim[0]
    return (
        int(lu_solver.a_size()) * itemsize  # smem_lu
        + int(lu_solver.b_size()) * itemsize  # smem_rhs
        + int(lu_solver.ipiv_size) * 4
        + batches * 4  # smem_info
        + 10 * batches * size * 8  # y, u, k1..k8
        + threads * 8  # smem_err
        + 7 * batches * 8  # t, dt, dt_use, inv_dt, t_end, err_prev, err_prev2
        + 5 * batches * 4  # save_idx, n_steps, accepted, rejected, accept
        + 4  # smem_continue
    )


def _fit_lu_solver(n_vars: int, size: int, precision, batches_per_block):
    """Size the LU batch so the *augmented* stage vectors still fit on chip.

    nvmath picks a batch count tuned for LU throughput at ``n_vars``, which is
    the right question for a plain solve and the wrong one for a joint solve:
    the stage vectors are ``size`` long, not ``n_vars``, so a batch that fits
    comfortably without sensitivities can overflow shared memory with them.
    """
    solver = make_lu_solver(
        n_vars, precision=precision, batches_per_block=batches_per_block
    )
    itemsize = np.dtype(precision).itemsize
    if size == n_vars or _shared_bytes(solver, size, itemsize) <= _SHARED_BUDGET:
        return solver
    # Everything but smem_err and smem_continue scales with the batch count.
    per_batch = (
        n_vars * n_vars * itemsize
        + n_vars * itemsize
        + n_vars * 4
        + 4
        + 10 * size * 8
        + 7 * 8
        + 5 * 4
    )
    fitted = (_SHARED_BUDGET - solver.block_dim[0] * 8 - 4) // per_batch
    if fitted < 1:
        raise ValueError(
            f"a Rodas5P solve carrying {size // n_vars - 1} sensitivity columns at "
            f"n_vars={n_vars} needs more shared memory than a CUDA block has; "
            "differentiate with respect to fewer inputs, or use Tsit5 if the "
            "system is not stiff"
        )
    return make_lu_solver(n_vars, precision=precision, batches_per_block=int(fitted))


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


@functools.cache
def _make_kernel(
    ode_fn,
    n_vars: int,
    n_params: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp32",
    batches_per_block="suggested",
    spec: SensitivitySpec | None = None,
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
    # With a sensitivity spec the kernel integrates the joint [y, S] system, so
    # the state, stage and error extents become the augmented ones. The LU does
    # not: the joint iteration matrix is block lower triangular with the *same*
    # M0 = I/(h*gamma) - J on every diagonal block, so one factorisation of the
    # n_vars block serves the state and every sensitivity column and the
    # coupling is a forward substitution. Factorising the whole n_aug matrix
    # would cost (1 + n_sens)**3 times the arithmetic and (1 + n_sens)**2 times
    # the shared memory to represent mostly structural zeros.
    size = n_vars if spec is None else spec.n_aug
    n_error = n_vars if spec is None else spec.n_error
    n_sens = 0 if spec is None else spec.n_sens
    lu_solver = _fit_lu_solver(n_vars, size, lu_dtype, batches_per_block)
    ode_write = (
        make_cuda_striped_vector_writer(ode_fn, n_vars)
        if spec is None
        else make_augmented_striped_writer(ode_fn, spec)
    )

    # df/dy and df/dt, forward-differentiated out of ode_fn by Enzyme. The
    # callback is the primal as written: numba-cuda-mlir flattens its tuple
    # arguments into one scalar parameter per element and lowers its tuple
    # return to a struct returned by value, which is the flat-scalar shape
    # Enzyme differentiates. Each sweep seeds one flattened argument, so the
    # call below reads a column of df/dy for col < n_vars and df/dt at
    # col == n_vars, where t sits. See "Derived Jacobians" in AGENTS.md for why
    # this is forward mode, why it is one column at a time, and what it costs.
    jacobian_column = make_jacobian_column(ode_fn, n_vars, n_params)

    if spec is None:

        @cuda.jit(device=True)
        def assemble_lu(y, t, p, lu_buf, a_off, dtgamma_inv, dT, i, lane, stride):
            # Build the Rosenbrock--Wanner iteration matrix M = 1/(h*gamma)*I - J
            # straight into the shared LU buffer. Evaluating the Jacobian and
            # writing M here (rather than staging J through a global array and
            # reading it back) avoids a per-step global-memory round-trip of the
            # full n_vars*n_vars matrix. Each lane takes a disjoint column-stripe,
            # so the O(n_vars^2) work is shared across the batch's lanes rather
            # than repeated in each of them.
            #
            # One forward sweep per column, with the seed chosen at run time.
            # Column n_vars seeds t instead of a state direction, so df/dt arrives
            # as a whole vector from one more sweep rather than per row.
            y_row = y[i]
            p_row = p[i]
            column = cuda.local.array(n_vars, types.float64)
            for col in range(lane, n_vars + 1, stride):
                jacobian_column(column, y_row, t, p_row, col)
                if col == n_vars:
                    for row in range(n_vars):
                        dT[i, row] = column[row]
                else:
                    for row in range(n_vars):
                        v = column[row]
                        if row == col:
                            lu_buf[a_off + row * n_vars + col] = lu_dtype(
                                dtgamma_inv - v
                            )
                        else:
                            lu_buf[a_off + row * n_vars + col] = lu_dtype(-v)

    else:
        n_y0_dirs = spec.n_y0_dirs
        seeds = seed_table(spec)
        length = max(n_vars, n_params)
        second_tangent = make_second_tangent(ode_fn, n_vars, n_params)

        @cuda.jit(device=True)
        def assemble_lu(y, t, p, lu_buf, a_off, dtgamma_inv, dT, i, lane, stride):
            # The LU block is the state block, exactly as above: every diagonal
            # block of the joint matrix is this same M0.
            seed = cuda.const.array_like(seeds)
            y_row = y[i]
            p_row = p[i]
            column = cuda.local.array(n_vars, types.float64)
            for col in range(lane, n_vars + 1, stride):
                jacobian_column(column, y_row, t, p_row, col)
                if col == n_vars:
                    for row in range(n_vars):
                        dT[i, row] = column[row]
                else:
                    for row in range(n_vars):
                        v = column[row]
                        if row == col:
                            lu_buf[a_off + row * n_vars + col] = lu_dtype(
                                dtgamma_inv - v
                            )
                        else:
                            lu_buf[a_off + row * n_vars + col] = lu_dtype(-v)
            # The sensitivity rows' dF/dt is d/dt (J_y S_k + J_p_k) at fixed
            # state: a second derivative of ode_fn, which the second-order
            # directional sweep returns by seeding the time direction on the
            # outside. Without it the method loses order on a non-autonomous
            # problem whose parameter dependence is itself time-dependent.
            for k in range(lane, n_sens, stride):
                base = n_vars + k * n_vars
                start = 0 if k < n_y0_dirs else length - (k - n_y0_dirs)
                second_tangent(
                    column,
                    y_row,
                    t,
                    p_row,
                    y_row[base : base + n_vars],
                    0.0,
                    seed[start : start + n_params],
                    seed[0:n_vars],
                    1.0,
                    seed[0:n_params],
                    # The inner direction does not vary: zero here leaves the
                    # plain second-order form.
                    seed[0:n_vars],
                    0.0,
                    seed[0:n_params],
                )
                for row in range(n_vars):
                    dT[i, base + row] = column[row]

    if spec is None:

        @cuda.jit(device=True)
        def block_solve(
            lu_buf,
            ipiv,
            rhs,
            staged,
            kout,
            y,
            p,
            t,
            i,
            lane,
            stride,
            active,
            stored,
            v_off,
            b_off,
        ):
            """Solve one Rosenbrock stage; with no sensitivities, one solve."""
            cuda.syncthreads()
            lu_solver.solve(lu_buf, ipiv, rhs)
            cuda.syncthreads()
            if stored:
                for j in range(lane, n_vars, stride):
                    kout[v_off + j] = np.float64(rhs[b_off + j])

    else:

        @cuda.jit(device=True)
        def block_solve(
            lu_buf,
            ipiv,
            rhs,
            staged,
            kout,
            y,
            p,
            t,
            i,
            lane,
            stride,
            active,
            stored,
            v_off,
            b_off,
        ):
            """Solve one Rosenbrock stage of the joint system by substitution.

            The joint iteration matrix is block lower triangular

                [  M0        ] [ k_y   ]   [ r_y   ]
                [ -L_k   M0  ] [ k_S_k ] = [ r_S_k ]

            so the state row solves first and each sensitivity row then solves
            against the *same* factorisation, its right-hand side corrected by
            ``L_k k_y``. ``L_k`` is never formed: the correction is one
            second-order directional sweep of ode_fn, seeded with the
            sensitivity column on the inside and the state increment on the
            outside. Like the Jacobian it belongs to, ``L_k`` is frozen at the
            step's base point; only the outer direction changes per stage.
            """
            seed = cuda.const.array_like(seeds)
            coupling = cuda.local.array(n_vars, types.float64)
            cuda.syncthreads()
            lu_solver.solve(lu_buf, ipiv, rhs)
            cuda.syncthreads()
            if stored:
                for j in range(lane, n_vars, stride):
                    kout[v_off + j] = np.float64(rhs[b_off + j])
            cuda.syncthreads()
            for k in range(n_sens):
                base = n_vars + k * n_vars
                if active:
                    y_row = y[i]
                    start = 0 if k < n_y0_dirs else length - (k - n_y0_dirs)
                    second_tangent(
                        coupling,
                        y_row,
                        t,
                        p[i],
                        y_row[base : base + n_vars],
                        0.0,
                        seed[start : start + n_params],
                        kout[v_off : v_off + n_vars],
                        0.0,
                        seed[0:n_params],
                        seed[0:n_vars],
                        0.0,
                        seed[0:n_params],
                    )
                    for j in range(lane, n_vars, stride):
                        rhs[b_off + j] = lu_dtype(staged[i, base + j] + coupling[j])
                cuda.syncthreads()
                lu_solver.solve(lu_buf, ipiv, rhs)
                cuda.syncthreads()
                if stored:
                    for j in range(lane, n_vars, stride):
                        kout[v_off + base + j] = np.float64(rhs[b_off + j])
                cuda.syncthreads()

    batches_per_block = lu_solver.batches_per_block
    block_threads = lu_solver.block_dim[0]
    vec_size = batches_per_block * size
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
        # A non-positive dt0 is the "no first step given" sentinel: start from
        # 1e-6 of the integration window.
        dt_init = dt0 if dt0 > 0.0 else (tf - times[0]) * 1e-6
        v_offset = batch * size
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
            for j in range(lane, size, batch_lanes):
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
            smem_dt[batch] = dt_init
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
                    smem_dt_use[batch] = dt_init
                    smem_inv_dt[batch] = 1.0 / dt_init
                    smem_t_end[batch] = smem_t[batch]
            cuda.syncthreads()

            if active:
                for j in range(lane, size, batch_lanes):
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
                    dT_global,
                    i,
                    lane,
                    batch_lanes,
                )
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
                for j in range(lane, size, batch_lanes):
                    stage_rhs = (
                        work_global[i, j] + smem_dt_use[batch] * D1 * dT_global[i, j]
                    )
                    # The state rows go straight to the shared solver
                    # buffer, which holds one n_vars block per batch;
                    # the sensitivity rows are staged until their own
                    # solve, after the coupling term is known.
                    if j < n_vars:
                        smem_rhs[b_offset + j] = lu_dtype(stage_rhs)
                    else:
                        work_global[i, j] = stage_rhs
            block_solve(
                smem_lu,
                smem_ipiv,
                smem_rhs,
                work_global,
                smem_k1,
                y_global,
                params,
                smem_t[batch],
                i,
                lane,
                batch_lanes,
                active,
                i < y0.shape[0],
                v_offset,
                b_offset,
            )

            if active:
                for j in range(lane, size, batch_lanes):
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
                for j in range(lane, size, batch_lanes):
                    stage_rhs = (
                        work_global[i, j]
                        + smem_dt_use[batch] * D2 * dT_global[i, j]
                        + C21 * smem_k1[v_offset + j] * smem_inv_dt[batch]
                    )
                    # The state rows go straight to the shared solver
                    # buffer, which holds one n_vars block per batch;
                    # the sensitivity rows are staged until their own
                    # solve, after the coupling term is known.
                    if j < n_vars:
                        smem_rhs[b_offset + j] = lu_dtype(stage_rhs)
                    else:
                        work_global[i, j] = stage_rhs
            block_solve(
                smem_lu,
                smem_ipiv,
                smem_rhs,
                work_global,
                smem_k2,
                y_global,
                params,
                smem_t[batch],
                i,
                lane,
                batch_lanes,
                active,
                i < y0.shape[0],
                v_offset,
                b_offset,
            )

            if active:
                for j in range(lane, size, batch_lanes):
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
                for j in range(lane, size, batch_lanes):
                    stage_rhs = (
                        work_global[i, j]
                        + smem_dt_use[batch] * D3 * dT_global[i, j]
                        + (C31 * smem_k1[v_offset + j] + C32 * smem_k2[v_offset + j])
                        * smem_inv_dt[batch]
                    )
                    # The state rows go straight to the shared solver
                    # buffer, which holds one n_vars block per batch;
                    # the sensitivity rows are staged until their own
                    # solve, after the coupling term is known.
                    if j < n_vars:
                        smem_rhs[b_offset + j] = lu_dtype(stage_rhs)
                    else:
                        work_global[i, j] = stage_rhs
            block_solve(
                smem_lu,
                smem_ipiv,
                smem_rhs,
                work_global,
                smem_k3,
                y_global,
                params,
                smem_t[batch],
                i,
                lane,
                batch_lanes,
                active,
                i < y0.shape[0],
                v_offset,
                b_offset,
            )

            if active:
                for j in range(lane, size, batch_lanes):
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
                for j in range(lane, size, batch_lanes):
                    stage_rhs = (
                        work_global[i, j]
                        + smem_dt_use[batch] * D4 * dT_global[i, j]
                        + (
                            C41 * smem_k1[v_offset + j]
                            + C42 * smem_k2[v_offset + j]
                            + C43 * smem_k3[v_offset + j]
                        )
                        * smem_inv_dt[batch]
                    )
                    # The state rows go straight to the shared solver
                    # buffer, which holds one n_vars block per batch;
                    # the sensitivity rows are staged until their own
                    # solve, after the coupling term is known.
                    if j < n_vars:
                        smem_rhs[b_offset + j] = lu_dtype(stage_rhs)
                    else:
                        work_global[i, j] = stage_rhs
            block_solve(
                smem_lu,
                smem_ipiv,
                smem_rhs,
                work_global,
                smem_k4,
                y_global,
                params,
                smem_t[batch],
                i,
                lane,
                batch_lanes,
                active,
                i < y0.shape[0],
                v_offset,
                b_offset,
            )

            if active:
                for j in range(lane, size, batch_lanes):
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
                for j in range(lane, size, batch_lanes):
                    stage_rhs = (
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
                    # The state rows go straight to the shared solver
                    # buffer, which holds one n_vars block per batch;
                    # the sensitivity rows are staged until their own
                    # solve, after the coupling term is known.
                    if j < n_vars:
                        smem_rhs[b_offset + j] = lu_dtype(stage_rhs)
                    else:
                        work_global[i, j] = stage_rhs
            block_solve(
                smem_lu,
                smem_ipiv,
                smem_rhs,
                work_global,
                smem_k5,
                y_global,
                params,
                smem_t[batch],
                i,
                lane,
                batch_lanes,
                active,
                i < y0.shape[0],
                v_offset,
                b_offset,
            )

            if active:
                for j in range(lane, size, batch_lanes):
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
                for j in range(lane, size, batch_lanes):
                    stage_rhs = (
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
                    # The state rows go straight to the shared solver
                    # buffer, which holds one n_vars block per batch;
                    # the sensitivity rows are staged until their own
                    # solve, after the coupling term is known.
                    if j < n_vars:
                        smem_rhs[b_offset + j] = lu_dtype(stage_rhs)
                    else:
                        work_global[i, j] = stage_rhs
            block_solve(
                smem_lu,
                smem_ipiv,
                smem_rhs,
                work_global,
                smem_k6,
                y_global,
                params,
                smem_t[batch],
                i,
                lane,
                batch_lanes,
                active,
                i < y0.shape[0],
                v_offset,
                b_offset,
            )
            if i < y0.shape[0]:
                for j in range(lane, size, batch_lanes):
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
                for j in range(lane, size, batch_lanes):
                    stage_rhs = (
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
                    # The state rows go straight to the shared solver
                    # buffer, which holds one n_vars block per batch;
                    # the sensitivity rows are staged until their own
                    # solve, after the coupling term is known.
                    if j < n_vars:
                        smem_rhs[b_offset + j] = lu_dtype(stage_rhs)
                    else:
                        work_global[i, j] = stage_rhs
            block_solve(
                smem_lu,
                smem_ipiv,
                smem_rhs,
                work_global,
                smem_k7,
                y_global,
                params,
                smem_t[batch],
                i,
                lane,
                batch_lanes,
                active,
                i < y0.shape[0],
                v_offset,
                b_offset,
            )
            if i < y0.shape[0]:
                for j in range(lane, size, batch_lanes):
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
                for j in range(lane, size, batch_lanes):
                    stage_rhs = (
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
                    # The state rows go straight to the shared solver
                    # buffer, which holds one n_vars block per batch;
                    # the sensitivity rows are staged until their own
                    # solve, after the coupling term is known.
                    if j < n_vars:
                        smem_rhs[b_offset + j] = lu_dtype(stage_rhs)
                    else:
                        work_global[i, j] = stage_rhs
            block_solve(
                smem_lu,
                smem_ipiv,
                smem_rhs,
                work_global,
                smem_k8,
                y_global,
                params,
                smem_t[batch],
                i,
                lane,
                batch_lanes,
                active,
                i < y0.shape[0],
                v_offset,
                b_offset,
            )

            err_local = 0.0
            if active:
                for j in range(lane, size, batch_lanes):
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
                    err_norm = math.sqrt(smem_err[tx] / n_error)
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
                    for j in range(lane, size, batch_lanes):
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
                        hist[i, save_idx, j] = theta1 * smem_y[v_offset + j] + theta * (
                            y_new_j + theta1 * (h1 + theta * (h2 + theta * h3))
                        )
                    save_idx += 1
                if lane == 0:
                    smem_save_idx[batch] = save_idx
                for j in range(lane, size, batch_lanes):
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
    y0,
    t_span,
    params,
    *,
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
    y0_arr, times, params_arr, dt0 = _normalize_inputs(y0, t_span, params, first_step)
    n, n_vars = y0_arr.shape
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    weights_arr = build_error_weights(error_weights, n, n_vars)

    # Scratch: the state and stage vectors the kernel stages through global
    # memory (y, u, the right-hand side) plus df/dt.
    workspace = get_workspace(
        _WORKSPACE_CACHE, n, n_vars, n_save, n_params, transposed=False, n_work=4
    )
    copy_workspace_inputs(workspace, y0_arr, times, params_arr, weights_arr)

    kernel, lu_solver = _make_kernel(
        ode_fn,
        n_vars,
        n_params,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
    )
    batches_per_block = lu_solver.batches_per_block
    threads = lu_solver.block_dim
    blocks = (n + batches_per_block - 1) // batches_per_block

    return PreparedNumbaSolve(
        kernel=kernel,
        workspace=workspace,
        dt0=np.float64(dt0),
        rtol=np.float64(rtol),
        atol=np.float64(atol),
        max_steps=np.int32(max_steps),
        blocks=blocks,
        threads=threads,
    )


def run_prepared(
    prepared: PreparedNumbaSolve, *, return_stats=False, copy_solution=True
):
    return run_kernel(
        prepared,
        prepared.workspace.work,
        return_stats=return_stats,
        copy_solution=copy_solution,
    )


@functools.cache
def _make_jax_launch(
    ode_fn,
    n: int,
    n_vars: int,
    n_params: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    lu_precision: str = "fp32",
    batches_per_block="suggested",
    spec: SensitivitySpec | None = None,
):
    kernel, lu_solver = _make_kernel(
        ode_fn,
        n_vars,
        n_params,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
        spec,
    )
    argtypes = SOLVER_ARGTYPES + (SCRATCH_ARGTYPE,) * 4
    batches_per_block = lu_solver.batches_per_block
    blocks = (n + batches_per_block - 1) // batches_per_block
    return make_launch(kernel, argtypes, grid=blocks, block=lu_solver.block_dim)


def solve(
    ode_fn,
    y0,
    t_span,
    params,
    *,
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
    sens_error_control=True,
):
    """JAX-callable Rodas5 custom-kernel solve.

    Only the right-hand side is supplied. The Jacobian ``df/dy``, and the
    partial time derivative ``df/dt`` that a non-autonomous system needs to
    keep fifth-order accuracy, are both differentiated out of ``ode_fn`` with
    Enzyme, so a non-autonomous problem needs nothing extra from the caller.

    ``lu_precision`` (``"fp32"`` or ``"fp64"``) selects the precision of the
    per-step LU factorisation and triangular solves. The state, right-hand
    side, Jacobian and error estimate are always float64; because the
    Rosenbrock--Wanner order conditions retain full order under an approximate
    Jacobian, the ``"fp32"`` default does not reduce the method's order while
    halving the LU shared-memory footprint and exploiting FP32 throughput.
    ``"fp64"`` is available for ill-conditioned systems where the FP32
    factorisation degrades step-size control.

    ``first_step`` pins the initial step size; the default ``None`` (like any
    non-positive value) lets the kernel start from 1e-6 of the integration
    window.

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

    The solve is an XLA custom call into the numba-cuda kernel, so it carries a
    ``jax.custom_jvp`` rule rather than being differentiated by XLA: asking for
    a derivative integrates the continuous forward-sensitivity system alongside
    the state (see ``solvers/_sensitivity.py``). ``jax.jvp``, ``jax.jacfwd``,
    ``jax.grad``, ``jax.jacrev`` and ``jax.value_and_grad`` all work with
    respect to ``y0`` and ``params``; ``t_span`` is not differentiable. An
    undifferentiated call runs the plain kernel and pays nothing.

    The joint system has ``n_vars * (1 + n_sens)`` components, but its iteration
    matrix is *not* factorised whole: it is block lower triangular with the same
    ``M0 = I/(h*gamma) - J_y`` on every diagonal block, so one ``n_vars``
    factorisation serves the state and every sensitivity column and the coupling
    is a forward substitution. The cost that does grow with ``n_sens`` is
    occupancy -- ten stage vectors of the augmented state live in shared memory,
    so the batch per block shrinks and ``batches_per_block`` is re-fitted
    automatically. This is still a solver for problems with few parameters
    relative to the state dimension.

    ``sens_error_control`` decides whether the sensitivity components take part
    in the step-size error norm. The default ``True`` controls them to the same
    ``rtol``/``atol`` as the state, so the gradient is as accurate as the value.
    ``False`` drops them from the norm, which makes the joint solve take exactly
    the step sequence the plain solve takes -- the value then matches a plain
    call bit for bit -- at the cost of nothing tying the sensitivities' accuracy
    to ``rtol``.

    The joint Jacobian's lower-left block ``d(J_y S + J_p)/dy`` is a second
    derivative of ``ode_fn``, and it is formed rather than dropped: Enzyme's
    forward-over-forward sweep applies it to the state increment without ever
    materialising the matrix. Dropping it would be legitimate under the W
    property -- order 5 survives an approximate Jacobian -- but not cheap: the
    error constant it costs was measured at 200x the steps on a right-hand side
    bilinear in state and parameters, which is most reaction networks.
    """

    settings = dict(
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
    # The JVP rule wraps the vmap-aware solvers rather than the other way
    # round: custom_vmap's own JVP path instantiates symbolic zeros, which is
    # what tells the rule which sensitivity blocks it has to integrate.
    primal_solver = make_custom_vmap_solver(
        functools.partial(_solve_impl, ode_fn, **settings),
        return_stats=return_stats,
    )

    def joint_solver_for(spec):
        return make_custom_vmap_solver(
            functools.partial(_solve_impl, ode_fn, spec=spec, **settings),
            return_stats=return_stats,
        )

    return make_sensitivity_solver(
        primal_solver,
        joint_solver_for,
        jnp.shape(y0)[-1],
        jnp.shape(params)[-1],
        return_stats,
        sens_error_control,
    )(y0, t_span, params)


def _solve_impl(
    ode_fn,
    y0,
    t_span,
    params,
    *,
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
    spec=None,
):
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    dt0 = initial_step(first_step)
    weights_host = build_error_weights(error_weights, n, n_vars)

    # With a spec the kernel integrates the joint [y, S] system, so every
    # per-component extent below is the augmented one.
    n_system = n_vars if spec is None else spec.n_aug
    if spec is not None:
        y0_arr = augmented_y0(y0_arr, spec)
        weights_host = augmented_error_weights(weights_host, spec)
    weights_arr = jnp.asarray(weights_host)

    launch = _make_jax_launch(
        ode_fn,
        n,
        n_vars,
        n_params,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        batches_per_block,
        spec,
    )
    # Scratch: y, u, the staged right-hand side, and df/dt.
    scratch_specs = (jax.ShapeDtypeStruct((n, n_system), jnp.float64),) * 4
    hist, accepted, rejected, loop_steps = ensemble_ffi_call(
        launch,
        (y0_arr, times, params_arr, weights_arr),
        scratch_specs,
        n=n,
        n_vars=n_system,
        n_save=n_save,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    if not return_stats:
        return hist
    return hist, solver_stats(accepted, rejected, loop_steps)
