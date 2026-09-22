"""Cooperative packed-batch Rodas5P custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
from numba_cuda_mlir import cuda, types

from menodax._codegen import compile_device_source
from menodax._jax_common import make_custom_vmap_solver, normalize_y0_params
from menodax._jax_numba_custom_call import make_launch
from menodax._numba_common import (
    HOOK_OUT_ARGTYPE,
    SOLVER_ARGTYPES,
    as_cuda_device,
    build_error_weights,
    ensemble_ffi_call,
    initial_step,
    make_cuda_local_vector_writer,
    solver_stats,
)
from menodax._sensitivity import (
    SensitivitySpec,
    augmented_error_weights,
    augmented_y0,
    make_augmented_local_writer,
    make_second_tangent,
    make_sensitivity_solver,
    make_tangent,
    seed_table,
)
from menodax._sparse_direct import sparse_direct_solver_for
from menodax._sparsity import dense_jacobian, normalize_sparsity

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

# One warp per block: the trajectories in a block step adaptively and so diverge,
# and a warp is the granularity at which that divergence is free.
_DEFAULT_TRAJECTORIES_PER_BLOCK = 32


def trajectories_per_block_or_default(requested=None) -> int:
    """How many trajectories a block carries, one per thread.

    Nothing on chip bounds this any more: every per-trajectory buffer, the
    iteration matrix included, is thread-local, so the choice is purely how wide
    a block should be. A warp is the default because the trajectories in a block
    step adaptively and diverge, and a warp is the granularity at which that
    divergence costs nothing.
    """
    if requested is None:
        return _DEFAULT_TRAJECTORIES_PER_BLOCK
    requested = int(requested)
    if requested < 1:
        raise ValueError(f"trajectories_per_block must be positive, got {requested}")
    return requested


@dataclass(frozen=True)
class KernelOptions:
    """Everything that selects a compiled kernel besides ``ode_fn`` and the shapes.

    Hashable, so it is the kernel cache key: two solves with equal options and
    the same callback share one compiled kernel. ``solve`` builds it once from
    its keyword arguments and hands it down unchanged; the sensitivity rule
    substitutes ``spec`` and nothing else.
    """

    pcoeff: float = 0.0
    icoeff: float = 1.0
    dcoeff: float = 0.0
    lu_precision: str = "fp32"
    trajectories_per_block: int = _DEFAULT_TRAJECTORIES_PER_BLOCK
    spec: SensitivitySpec | None = None
    sparsity: tuple[tuple[int, ...], ...] | None = None
    ordering: str = "amd"
    tf_index: int = -1
    max_registers: int | None = None
    array_rhs: object = None
    # A save hook ``hook(save_idx, y, t, p_row, acc)`` the kernel calls at every
    # save time with the dense-output state, accumulating into the trajectory's
    # ``hook_size``-wide output row; ``save_history=False`` then keeps only the
    # final state instead of the whole history.
    save_hook: object = None
    hook_size: int = 0
    save_history: bool = True


@functools.cache
def dense_lu_solver(n_vars: int):
    """Dense LU with partial pivoting, one system per thread.

    This is what the kernel uses when it is given no sparsity pattern:
    right-looking LU over the thread's own row-major buffer, then the two
    triangular solves in place. Returned as the same
    ``(factorize_local, solve_local)`` pair
    [`sparse_direct_solver`][menodax._sparse_direct.sparse_direct_solver]
    builds from a pattern, so the kernel calls one or the other and has no branch.

    It replaced nvmath's ``LUPivotSolver``, whose block-collective API was the
    only reason the kernel ever put a matrix in shared memory. That cost a
    barrier around every factorisation and every stage solve, and shared memory
    that an ensemble's occupancy could not spare. A thread owning its whole
    trajectory needs neither, and these matrices -- tens of variables, not
    thousands -- are far too small for cooperation to pay for itself.
    """
    n = n_vars

    @cuda.jit(device=True)
    def factorize_local(lu, ipiv):
        for i in range(n):
            # Partial pivoting: the largest remaining entry in this column.
            max_val = abs(lu[i * n + i])
            pivot = i
            for k in range(i + 1, n):
                val = abs(lu[k * n + i])
                if val > max_val:
                    max_val = val
                    pivot = k
            ipiv[i] = pivot

            if pivot != i:
                for j in range(n):
                    tmp = lu[i * n + j]
                    lu[i * n + j] = lu[pivot * n + j]
                    lu[pivot * n + j] = tmp

            # A singular column is left alone rather than guarded against: the
            # Rosenbrock-W property tolerates an approximate factorisation, and
            # the step controller rejects whatever comes out of one that is not.
            piv = lu[i * n + i]
            if piv != 0.0:
                inv_piv = 1.0 / piv
                for k in range(i + 1, n):
                    factor = lu[k * n + i] * inv_piv
                    lu[k * n + i] = factor
                    for j in range(i + 1, n):
                        lu[k * n + j] -= factor * lu[i * n + j]

    @cuda.jit(device=True)
    def solve_local(lu, ipiv, rhs):
        # Forward substitution through L, applying the pivots as they come.
        for i in range(n):
            pivot = ipiv[i]
            if pivot != i:
                tmp = rhs[i]
                rhs[i] = rhs[pivot]
                rhs[pivot] = tmp
            acc = rhs[i]
            for j in range(i):
                acc -= lu[i * n + j] * rhs[j]
            rhs[i] = acc
        # Back substitution through U.
        for i in range(n - 1, -1, -1):
            acc = rhs[i]
            for j in range(i + 1, n):
                acc -= lu[i * n + j] * rhs[j]
            rhs[i] = acc / lu[i * n + i]

    return factorize_local, solve_local


# Direction sets per jvp call. Several seeds in one call share one Enzyme entry
# function, so once nvJitLink has inlined it the primal work the sweeps have in
# common -- for an ODE whose coefficients depend on t alone, all of it -- is one
# computation for LLVM to CSE rather than one per sweep. On DISCO-EB that was
# the difference between 580 ms and 509 ms at N128, on top of the literal
# seeds. numba-enzyme accepts up to 8 sets per call (its _MAX_DIRECTIONS), so
# 12 colours take two calls.
SEED_BATCH = 8


def _make_literal_seed_jacobian(*, n_vars, n_params, n_colours, store_table, namespace):
    """Generate the Jacobian writer with every colour's seed row spelled out.

    ``store_table`` is the layout's destination table, ``(row, colour) ->
    slot`` with ``-1`` for none, or ``None`` for the plain colour grid where
    entry ``(row, g)`` sits at ``row * n_colours + g``.
    """
    zero_row = n_colours
    batch = SEED_BATCH

    def slots_for(g):
        for row in range(n_vars):
            if store_table is None:
                yield row, row * n_colours + g
            else:
                slot = int(store_table[row * n_colours + g])
                if slot >= 0:
                    yield row, slot

    lines = [
        "def write_negated_jacobian(y_local, t, p_row, lu, dT):",
        "    seed = cuda.const.array_like(colour_seeds)",
        f"    column = cuda.local.array({n_vars}, float64)",
        "    clear_matrix(lu)",
    ]
    if batch > 1:
        lines.append(f"    block = cuda.local.array(({batch}, {n_vars}), float64)")
    for start in range(0, n_colours, batch):
        group = list(range(start, min(start + batch, n_colours)))
        if len(group) == 1:
            g = group[0]
            lines.append(
                f"    tangent_of(column, y_local, t, p_row, seed[{g}, 0:{n_vars}], 0.0, "
                f"seed[{zero_row}, 0:{n_params}])"
            )
            for row, slot in slots_for(g):
                lines.append(f"    lu[{slot}] = lu_dtype(-column[{row}])")
        else:
            dirs = ", ".join(
                f"seed[{g}, 0:{n_vars}], 0.0, seed[{zero_row}, 0:{n_params}]"
                for g in group
            )
            lines.append(f"    tangent_of(block, y_local, t, p_row, {dirs})")
            for k, g in enumerate(group):
                for row, slot in slots_for(g):
                    lines.append(f"    lu[{slot}] = lu_dtype(-block[{k}, {row}])")
    lines += [
        f"    tangent_of(column, y_local, t, p_row, seed[{zero_row}, 0:{n_vars}], 1.0, "
        f"seed[{zero_row}, 0:{n_params}])",
        f"    for row in range({n_vars}):",
        "        dT[row] = column[row]",
    ]
    return compile_device_source("write_negated_jacobian", lines, namespace)


@functools.cache
def _make_kernel(ode_fn, n_vars: int, n_params: int, options: KernelOptions):
    """Rodas5P, one trajectory per CUDA thread.

    A thread owns its trajectory outright: the state, the ten stage vectors,
    ``df/dt``, the step controller *and* the iteration matrix are all its own
    local arrays and scalars. Nothing is shared and nothing synchronises inside
    a step, so neither the state dimension nor the matrix bounds the block's
    trajectory count, and each thread writes its own Jacobian straight into its
    own matrix -- the ``O(n_vars**2)`` assembly is parallel across trajectories
    rather than across lanes of one.

    The linear solve is one code path over a buffer the layout describes. With
    no pattern that layout is the dense row-major matrix and the solver is
    [`dense_lu_solver`][menodax.rodas5P.dense_lu_solver]; with one it is the
    sparse factorisation's own CSR image and the solver is compiled for it.
    The two differ in what they were built from and in nothing else the kernel
    can see.
    """
    spec, sparsity, lu_precision = options.spec, options.sparsity, options.lu_precision
    tf_index, array_rhs = options.tf_index, options.array_rhs
    # Both compile-time constants: numba prunes the hook call and the history
    # writes it does not need, so a solve without a hook compiles as before.
    HAS_HOOK = options.save_hook is not None
    SAVE_HISTORY = bool(options.save_history)
    hook_size = max(1, int(options.hook_size))
    if HAS_HOOK:
        save_hook = as_cuda_device(options.save_hook)
    else:

        @cuda.jit(device=True)
        def save_hook(save_idx, y, t, p_row, acc):
            return
    e1 = EXPONENT * (options.icoeff + options.pcoeff + options.dcoeff)
    e2 = -EXPONENT * (options.pcoeff + 2.0 * options.dcoeff)
    e3 = EXPONENT * options.dcoeff
    # The state, right-hand side, Jacobian and error estimate are always
    # float64; lu_dtype governs only the iteration matrix and its solves. The
    # Rosenbrock--Wanner order conditions hold under an approximate Jacobian, so
    # an FP32 factorisation does not reduce the method's order.
    lu_dtype = np.float32 if lu_precision == "fp32" else np.float64
    # cuda.local.array wants the numba type rather than the numpy dtype.
    lu_local_dtype = types.float32 if lu_precision == "fp32" else types.float64
    if sparsity is None:
        structure = dense_jacobian(n_vars)
        # The dense LU pivots the whole state; the sparse one pivots nothing.
        factorize_local, solve_local = dense_lu_solver(n_vars)
        ipiv_per = n_vars
    else:
        solver = sparse_direct_solver_for(sparsity, options.ordering)
        structure = solver.compressed
        factorize_local, solve_local = solver.factorize_local, solver.solve_local
        ipiv_per = solver.ipiv_size
    # With a spec the kernel integrates the joint [y, S] system, so the state,
    # stage and error extents become the augmented ones. The matrix does not:
    # the joint iteration matrix is block lower triangular with the same
    # M0 = I/(h*gamma) - J on every diagonal block, so one factorisation of the
    # n_vars block serves the state and every sensitivity column.
    size = n_vars if spec is None else spec.n_aug
    n_error = n_vars if spec is None else spec.n_error
    n_sens = 0 if spec is None else spec.n_sens

    tpb = int(options.trajectories_per_block)
    n_colours = structure.n_colours

    lu_size = structure.size
    diag_table = np.asarray(structure.diagonal, dtype=np.int32)
    store_table = structure.store_slots()
    # Only a layout with slots nothing writes needs clearing first. The dense
    # grid writes every slot; a sparse layout has one slot per entry of J and
    # leaves the factorisation's fill-in for nobody to write.
    needs_clear = store_table is not None and lu_size > int((store_table >= 0).sum())

    if spec is not None:
        ode_write = make_augmented_local_writer(ode_fn, spec)
    elif array_rhs is not None:
        # The primal stage evaluations need no tuple form: that exists for
        # Enzyme, which only ever sees ode_fn. A caller whose right-hand side
        # also comes as ``f(y, t, p, out)`` over arrays can hand that in, and
        # the eight stage evaluations per step call it directly.
        _array_rhs = as_cuda_device(array_rhs)

        @cuda.jit(device=True)
        def ode_write(y_row, t, p_row, out):
            _array_rhs(y_row, t, p_row, out)

    else:
        ode_write = make_cuda_local_vector_writer(ode_fn, n_vars)

    if spec is not None:
        n_y0_dirs = spec.n_y0_dirs
        param_cols = spec.param_seed_columns
        seeds = seed_table(spec)
        length = max(n_vars, n_params)
        second_tangent = make_second_tangent(ode_fn, n_vars, n_params)

        @cuda.jit(device=True)
        def write_sensitivity_time_derivative(y_local, t, p_row, dT):
            """``dF/dt`` for the sensitivity rows of the joint system.

            It is ``d/dt (J_y S_k + J_p_k)`` at fixed state -- a second
            derivative of ode_fn, which the second-order directional sweep
            returns by seeding the time direction on the outside. Without it the
            method loses order on a non-autonomous problem whose parameter
            dependence is itself time-dependent, and leaves these rows of ``dT``
            holding whatever was last in that local memory.
            """
            seed = cuda.const.array_like(seeds)
            column = cuda.local.array(n_vars, types.float64)
            for k in range(n_sens):
                base = n_vars + k * n_vars
                start = 0 if k < n_y0_dirs else length - param_cols[k - n_y0_dirs]
                second_tangent(
                    column,
                    y_local,
                    t,
                    p_row,
                    y_local[base : base + n_vars],
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
                    dT[base + row] = column[row]

    else:

        @cuda.jit(device=True)
        def write_sensitivity_time_derivative(y_local, t, p_row, dT):
            pass

    # --- the negated Jacobian, straight into the thread's own buffer ---------
    # ``df/dy`` and ``df/dt`` are forward-differentiated out of ode_fn by
    # Enzyme. Each sweep is seeded with a whole colour group rather than one
    # unit vector: the columns in a group share no row, so their contributions
    # to ``J v`` never collide and one sweep yields the lot. That is
    # ``n_colours + 1`` sweeps where a column at a time costs ``n_vars + 1``,
    # and for a banded or bordered system the difference is an order of
    # magnitude. See "Derived Jacobians" in AGENTS.md.
    #
    # This leaves -J in the buffer and df/dt in dT; the kernel then adds
    # 1/(h*gamma) on the diagonal slots to finish M. Writing straight here,
    # rather than staging J through global memory and reading it back, avoids a
    # per-step round-trip of the whole matrix.
    tangent_of = make_tangent(ode_fn, n_vars, n_params)
    colour_seeds = structure.seed_table(n_params)

    # The grid writes every slot and a packed layout has one slot per entry,
    # so both come out fully defined by the sweeps. Only a layout with slots
    # nothing writes -- the dense expansion's structural zeros, or a packed
    # diagonal the pattern left out -- is cleared first.
    if needs_clear:

        @cuda.jit(device=True)
        def clear_matrix(lu):
            for i in range(lu_size):
                lu[i] = lu_dtype(0.0)

    else:

        @cuda.jit(device=True)
        def clear_matrix(lu):
            pass

    # The colour loop is unrolled into call sites whose seed rows are
    # *literals*. The derivative links as LTO IR and nvJitLink inlines it into
    # the kernel before constant propagation, so a seed the compiler can see
    # folds: the zero components kill their tangent arithmetic and each sweep
    # collapses to its own colour group's columns. The same seed read through
    # a loop variable arrives in registers and cannot fold, which is what had
    # this kernel at the register cap with a 13 KB spill frame. Several seeds
    # go into each call (SEED_BATCH) so the sweeps' shared primal is computed
    # once, and the stores are literal too, one per entry the colour holds.
    write_negated_jacobian = _make_literal_seed_jacobian(
        n_vars=n_vars,
        n_params=n_params,
        n_colours=n_colours,
        store_table=store_table,
        namespace=dict(
            cuda=cuda,
            float64=types.float64,
            tangent_of=tangent_of,
            colour_seeds=colour_seeds,
            clear_matrix=clear_matrix,
            lu_dtype=lu_dtype,
        ),
    )

    # --- one Rosenbrock stage, state row plus any sensitivity rows -----------
    if spec is None:

        @cuda.jit(device=True)
        def stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t):
            solve_local(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[s, j] = np.float64(rhs[j])

    else:

        @cuda.jit(device=True)
        def stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t):
            """Solve one stage of the joint system by forward substitution.

            The joint iteration matrix is block lower triangular

                [  M0        ] [ k_y   ]   [ r_y   ]
                [ -L_k   M0  ] [ k_S_k ] = [ r_S_k ]

            so the state row solves first and each sensitivity row then solves
            against the *same* factorisation, its right-hand side corrected by
            ``L_k k_y``. ``L_k`` is never formed: the correction is one
            second-order directional sweep of ode_fn, seeded with the
            sensitivity column on the inside and the state increment on the
            outside. Like the Jacobian it belongs to, it is frozen at the step's
            base point; only the outer direction changes per stage.
            """
            seed = cuda.const.array_like(seeds)
            coupling = cuda.local.array(n_vars, types.float64)
            # The stage vectors carry lu_precision, but Enzyme's endpoint is
            # built for float64 directions, so the state increment is widened
            # here rather than the stages being stored wider throughout.
            k_state = cuda.local.array(n_vars, types.float64)
            solve_local(lu, ipiv, rhs)
            for j in range(n_vars):
                k_stages[s, j] = np.float64(rhs[j])
            for j in range(n_vars):
                k_state[j] = np.float64(k_stages[s, j])
            for k in range(n_sens):
                base = n_vars + k * n_vars
                start = 0 if k < n_y0_dirs else length - param_cols[k - n_y0_dirs]
                second_tangent(
                    coupling,
                    y,
                    t,
                    p_row,
                    y[base : base + n_vars],
                    0.0,
                    seed[start : start + n_params],
                    k_state,
                    0.0,
                    seed[0:n_params],
                    # The inner direction does not vary: zero here leaves
                    # the plain bilinear form.
                    seed[0:n_vars],
                    0.0,
                    seed[0:n_params],
                )
                for j in range(n_vars):
                    rhs[j] = lu_dtype(work[base + j] + coupling[j])
                solve_local(lu, ipiv, rhs)
                for j in range(n_vars):
                    k_stages[s, base + j] = np.float64(rhs[j])

    if spec is None:

        @cuda.jit(device=True)
        def put_stage_rhs(rhs_buf, work, j, value):
            rhs_buf[j] = lu_dtype(value)

    else:

        @cuda.jit(device=True)
        def put_stage_rhs(rhs_buf, work, j, value):
            # State rows go straight to the solver's buffer, which holds one
            # n_vars block per trajectory; sensitivity rows are staged until
            # their own solve, once the coupling term is known.
            if j < n_vars:
                rhs_buf[j] = lu_dtype(value)
            else:
                work[j] = value

    # --- the stages themselves, one device function per tableau row ----------
    # Every coefficient is a closure constant and every loop below has a
    # constant trip count, so each function lowers to the straight-line
    # arithmetic the row spelled out by hand would: the compiled kernels carry
    # the same instruction mix as the hand-unrolled form did. The split on
    # ``d`` keeps the three stages without a df/dt term from emitting an
    # ``x * 0.0`` the compiler may not drop.
    c_rows = (
        (),
        (C21,),
        (C31, C32),
        (C41, C42, C43),
        (C51, C52, C53, C54),
        (C61, C62, C63, C64, C65),
        (C71, C72, C73, C74, C75, C76),
        (C81, C82, C83, C84, C85, C86, C87),
    )
    d_row = (D1, D2, D3, D4, D5, 0.0, 0.0, 0.0)
    a_rows = (
        (A21,),
        (A31, A32),
        (A41, A42, A43),
        (A51, A52, A53, A54),
        (A61, A62, A63, A64, A65),
    )

    def make_stage(s):
        c_row = c_rows[s]
        d = d_row[s]

        if s == 0:

            @cuda.jit(device=True)
            def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_row, t, dt_use, inv_dt):
                for j in range(size):
                    put_stage_rhs(rhs, work, j, work[j] + dt_use * d * dT[j])
                stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t)

        elif d != 0.0:

            @cuda.jit(device=True)
            def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_row, t, dt_use, inv_dt):
                for j in range(size):
                    acc = c_row[0] * k_stages[0, j]
                    for k in range(1, s):
                        acc += c_row[k] * k_stages[k, j]
                    put_stage_rhs(
                        rhs, work, j, work[j] + dt_use * d * dT[j] + acc * inv_dt
                    )
                stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t)

        else:

            @cuda.jit(device=True)
            def stage(lu, ipiv, rhs, k_stages, work, dT, y, p_row, t, dt_use, inv_dt):
                for j in range(size):
                    acc = c_row[0] * k_stages[0, j]
                    for k in range(1, s):
                        acc += c_row[k] * k_stages[k, j]
                    put_stage_rhs(rhs, work, j, work[j] + acc * inv_dt)
                stage_solve(lu, ipiv, rhs, k_stages, s, work, y, p_row, t)

        return stage

    def make_advance(s):
        a_row = a_rows[s]
        n_prev = s + 1

        @cuda.jit(device=True)
        def advance(u, y, k_stages):
            for j in range(size):
                acc = a_row[0] * k_stages[0, j]
                for k in range(1, n_prev):
                    acc += a_row[k] * k_stages[k, j]
                u[j] = y[j] + acc

        return advance

    stage_0, stage_1, stage_2, stage_3, stage_4, stage_5, stage_6, stage_7 = (
        make_stage(s) for s in range(8)
    )
    advance_0, advance_1, advance_2, advance_3, advance_4 = (
        make_advance(s) for s in range(5)
    )

    jit_options = (
        {}
        if options.max_registers is None
        else {"max_registers": options.max_registers}
    )

    @cuda.jit(**jit_options)
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
        hook_out,
    ):
        i = cuda.blockIdx.x * tpb + cuda.threadIdx.x
        # Nothing in a step is collective any more, so a thread past the end of
        # the ensemble simply leaves rather than shadowing a trajectory through
        # barriers it would otherwise have to reach.
        if i >= y0.shape[0]:
            return
        diag = cuda.const.array_like(diag_table)

        n_save = times.shape[0]
        tf = times[n_save - 1]
        # A non-positive dt0 is the "no first step given" sentinel: start from
        # 1e-6 of the integration window.
        dt_init = dt0 if dt0 > 0.0 else (tf - times[0]) * 1e-6

        lu_buf = cuda.local.array(lu_size, lu_local_dtype)
        rhs_buf = cuda.local.array(n_vars, lu_local_dtype)
        ipiv_buf = cuda.local.array(ipiv_per, np.int32)

        y = cuda.local.array(size, np.float64)
        u = cuda.local.array(size, np.float64)
        work = cuda.local.array(size, np.float64)
        dT = cuda.local.array(size, np.float64)
        # The stage increments are what came out of the linear solve, so they
        # already carry only lu_precision; storing them wider keeps rounding
        # that is not there and doubles the thread's local-memory footprint,
        # which is what bounds occupancy once the matrix is off the shared path.
        k_stages = cuda.local.array((8, size), lu_local_dtype)
        # The dense-output state a save hook is handed; a single dummy slot
        # when there is no hook, so the default kernel carries no extra local
        # memory.
        y_save = cuda.local.array(size if HAS_HOOK else 1, np.float64)

        for j in range(size):
            y[j] = y0[i, j]
        if SAVE_HISTORY:
            for j in range(size):
                hist[i, 0, j] = y[j]

        t = times[0]
        save_idx = 1
        dt = dt_init
        n_steps = 0
        accepted = 0
        rejected = 0
        err_prev = 1.0
        err_prev2 = 1.0

        # Each trajectory may finish at its own end time, read from a parameter
        # column; the save times it never reaches hold its final state.
        tf_local = tf
        if tf_index >= 0:
            tf_local = params[i, tf_index]

        p_row = params[i]

        if HAS_HOOK:
            for j in range(hook_size):
                hook_out[i, j] = 0.0
            save_hook(0, y, times[0], p_row, hook_out[i])

        # Purely this thread's loop: it steps until its own trajectory is done
        # and then falls out, with no reference to what the rest of the block is
        # doing.
        while save_idx < n_save and t < tf_local and n_steps < max_steps:
            dt_use = dt
            if dt_use > tf_local - t:
                dt_use = tf_local - t
            if dt_use < 1e-30:
                dt_use = 1e-30
            inv_dt = 1.0 / dt_use
            t_end = t + dt_use

            dtgamma_inv = 1.0 / (dt_use * GAMMA)
            write_negated_jacobian(y, t, p_row, lu_buf, dT)
            write_sensitivity_time_derivative(y, t, p_row, dT)
            for d in range(n_vars):
                lu_buf[diag[d]] += lu_dtype(dtgamma_inv)
            factorize_local(lu_buf, ipiv_buf)

            # The eight Rosenbrock stages. Each forms its right-hand side from
            # the tableau row, solves against the step's one factorisation, and
            # the u update between them is the next stage's argument.
            ode_write(y, t, p_row, work)
            stage_0(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_0(u, y, k_stages)
            ode_write(u, t + C2 * dt_use, p_row, work)
            stage_1(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_1(u, y, k_stages)
            ode_write(u, t + C3 * dt_use, p_row, work)
            stage_2(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_2(u, y, k_stages)
            ode_write(u, t + C4 * dt_use, p_row, work)
            stage_3(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_3(u, y, k_stages)
            ode_write(u, t + C5 * dt_use, p_row, work)
            stage_4(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            advance_4(u, y, k_stages)
            # Stages 6-8 share the end point and accumulate into u (FSAL form).
            ode_write(u, t_end, p_row, work)
            stage_5(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            for j in range(size):
                u[j] += k_stages[5, j]
            ode_write(u, t_end, p_row, work)
            stage_6(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )
            for j in range(size):
                u[j] += k_stages[6, j]
            ode_write(u, t_end, p_row, work)
            stage_7(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                work,
                dT,
                y,
                p_row,
                t,
                dt_use,
                inv_dt,
            )

            # Weighted RMS error estimate.
            err_local = 0.0
            for j in range(size):
                y_new_j = u[j] + k_stages[7, j]
                scale = atol + rtol * max(math.fabs(y[j]), math.fabs(y_new_j))
                r = weights[i, j] * k_stages[7, j] / scale
                err_local += r * r
            err_norm = math.sqrt(err_local / n_error)
            accept = err_norm <= 1.0 and not math.isnan(err_norm)

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

            if accept:
                t_old = t
                t_new = t_old + dt_use
                while save_idx < n_save and times[save_idx] <= t_new + 1e-12 * max(
                    1.0, math.fabs(times[save_idx])
                ):
                    theta = (times[save_idx] - t_old) / dt_use
                    theta1 = 1.0 - theta
                    for j in range(size):
                        h1 = (
                            25.948786856663858 * k_stages[0, j]
                            - 2.5579724845846235 * k_stages[1, j]
                            + 10.433815404888879 * k_stages[2, j]
                            - 2.3679251022685204 * k_stages[3, j]
                            + 0.524948541321073 * k_stages[4, j]
                            + 1.1241088310450404 * k_stages[5, j]
                            + 0.4272876194431874 * k_stages[6, j]
                            - 0.17202221070155493 * k_stages[7, j]
                        )
                        h2 = (
                            -9.91568850695171 * k_stages[0, j]
                            - 0.9689944594115154 * k_stages[1, j]
                            + 3.0438037242978453 * k_stages[2, j]
                            - 24.495224566215796 * k_stages[3, j]
                            + 20.176138334709044 * k_stages[4, j]
                            + 15.98066361424651 * k_stages[5, j]
                            - 6.789040303419874 * k_stages[6, j]
                            - 6.710236069923372 * k_stages[7, j]
                        )
                        h3 = (
                            11.419903575922262 * k_stages[0, j]
                            + 2.8879645146136994 * k_stages[1, j]
                            + 72.92137995996029 * k_stages[2, j]
                            + 80.12511834622643 * k_stages[3, j]
                            - 52.072871366152654 * k_stages[4, j]
                            - 59.78993625266729 * k_stages[5, j]
                            - 0.15582684282751913 * k_stages[6, j]
                            + 4.883087185713722 * k_stages[7, j]
                        )
                        y_new_j = u[j] + k_stages[7, j]
                        # Rosenbrock continuous extension between y_old and
                        # y_new. y[j] is still y_old here; it is advanced
                        # after this loop.
                        val = theta1 * y[j] + theta * (
                            y_new_j + theta1 * (h1 + theta * (h2 + theta * h3))
                        )
                        if SAVE_HISTORY:
                            hist[i, save_idx, j] = val
                        if HAS_HOOK:
                            y_save[j] = val
                    if HAS_HOOK:
                        save_hook(save_idx, y_save, times[save_idx], p_row, hook_out[i])
                    save_idx += 1
                for j in range(size):
                    y[j] = u[j] + k_stages[7, j]
                t += dt_use
                accepted += 1
            else:
                rejected += 1
            n_steps += 1

        # Save times past this trajectory's own end time hold its final state.
        while save_idx < n_save:
            if SAVE_HISTORY:
                for j in range(size):
                    hist[i, save_idx, j] = y[j]
            if HAS_HOOK:
                save_hook(save_idx, y, times[save_idx], p_row, hook_out[i])
            save_idx += 1
        if not SAVE_HISTORY:
            # Without the history the one slot holds the final state.
            for j in range(size):
                hist[i, 0, j] = y[j]
        accepted_out[i] = accepted
        rejected_out[i] = rejected
        loop_out[i] = n_steps

    return kernel, tpb


@functools.cache
def _make_jax_launch(
    ode_fn, n: int, n_vars: int, n_params: int, options: KernelOptions
):
    kernel, trajectories_per_block = _make_kernel(ode_fn, n_vars, n_params, options)
    blocks = (n + trajectories_per_block - 1) // trajectories_per_block
    return make_launch(
        kernel,
        SOLVER_ARGTYPES + (HOOK_OUT_ARGTYPE,),
        grid=blocks,
        block=(trajectories_per_block, 1, 1),
    )


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
    trajectories_per_block=None,
    sens_error_control=True,
    sens_param_columns=None,
    sparsity=None,
    ordering="amd",
    tf_index=None,
    max_registers=None,
    array_rhs=None,
    save_hook=None,
    hook_size: int = 0,
    save_history: bool = True,
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


    The solve is an XLA custom call into the numba-cuda kernel, so it carries a
    ``jax.custom_jvp`` rule rather than being differentiated by XLA: asking for
    a derivative integrates the continuous forward-sensitivity system alongside
    the state (see ``menodax/_sensitivity.py``). ``jax.jvp``, ``jax.jacfwd``,
    ``jax.grad``, ``jax.jacrev`` and ``jax.value_and_grad`` all work with
    respect to ``y0`` and ``params``; ``t_span`` is not differentiable. An
    undifferentiated call runs the plain kernel and pays nothing.

    The joint system has ``n_vars * (1 + n_sens)`` components, but its iteration
    matrix is *not* factorised whole: it is block lower triangular with the same
    ``M0 = I/(h*gamma) - J_y`` on every diagonal block, so one ``n_vars``
    factorisation serves the state and every sensitivity column and the coupling
    is a forward substitution. What grows with ``n_sens`` is the thread's own
    working set -- ten stage vectors of the augmented state, in local memory --
    rather than the block's shared budget, which carries only the ``n_vars``
    matrix. This is still a solver for problems with few parameters relative to
    the state dimension.

    ``sens_param_columns`` restricts the parameter sensitivities to the columns
    named, rather than carrying one block per parameter. The joint system is
    ``n_vars * (1 + n_sens)`` components and each direction costs a
    second-order Enzyme sweep per stage, so this is the difference between
    paying for the parameters you want and paying for the whole ``params`` row
    -- which matters most when that row also carries things that are not
    parameters at all, such as integration bounds or an integer selecting a
    table. The default ``None`` carries every column, as before.

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

    ``sparsity`` is where a structured problem pays off, and it is the only
    thing a caller has to supply to get one. Pass an ``(n_vars, n_vars)`` mask,
    a scipy sparse matrix, or an ``(nnz, 2)`` array of indices, and two things
    follow. The Jacobian costs one Enzyme sweep per *colour* of the pattern's
    column intersection graph rather than one per column, since columns sharing
    no row can be seeded together and the pattern says which output component
    belongs to which ([`menodax._sparsity`][]). And the iteration matrix is
    ordered, factorised symbolically and given an in-kernel sparse LU and sparse
    triangular solves compiled for that exact structure
    ([`menodax._sparse_direct`][]). The default -- no pattern -- colours every
    column apart and factorises densely, which is the same mechanism at its
    uninformative end rather than a second path.

    The pattern must be a **superset** of the true nonzeros. Colouring a
    superset only costs sweeps; colouring a subset silently corrupts the entries
    where two columns of a group do overlap after all. It need *not* include the
    factorisation's fill-in, which the symbolic pass works out and gives slots
    of its own.

    ``ordering`` picks the fill-reducing permutation: ``"amd"`` by default,
    SuiteSparse's approximate minimum degree out of ``cvxopt``, or
    ``"natural"`` to skip the ordering. It is ignored without a pattern.

    Forward sensitivities work with either solver: the joint iteration matrix is
    block lower triangular with the same ``M0`` on every diagonal block, so only
    the ``n_vars`` block is ever factorised and the coupling between blocks is a
    forward substitution the kernel does itself.

    ``tf_index`` names a column of ``params`` holding each trajectory's own end
    time, for ensembles whose members finish at different times; save times
    past a trajectory's end hold its final state. The default ``None`` ends
    every trajectory at ``t_span[-1]``.

    ``trajectories_per_block`` is one thread's worth of work each, defaulting to
    a warp; nothing on chip bounds it, since every per-trajectory buffer is
    thread-local. ``max_registers`` caps the
    kernel's per-thread register count, trading spills against occupancy.

    ``save_hook`` is a device function ``hook(save_idx, y, t, p_row, acc)`` the
    kernel calls at every save time -- the initial state, each dense-output
    save, and the frozen saves past a trajectory's own end time -- with the
    state at that time, so a consumer of the history can be evaluated inside
    the launch instead of after it. ``acc`` is the trajectory's row of the
    ``(n, hook_size)`` output, zeroed at the start and persistent across saves,
    so the hook can accumulate (a line-of-sight integral, say) or store derived
    quantities per save. With ``save_history=False`` the history output shrinks
    to the final state, shape ``(n, 1, n_vars)``. A solve with a hook returns
    ``(hist, hook_out)`` (plus the stats when asked) and supports neither
    ``jax.vmap`` nor differentiation.
    """
    n_vars = jnp.shape(y0)[-1]
    # Normalised here, not in the kernel builder, because that is cached on its
    # arguments and a pattern has to arrive as the same hashable value twice.
    if sparsity is not None:
        sparsity = normalize_sparsity(sparsity, n_vars)
    options = KernelOptions(
        pcoeff=pcoeff,
        icoeff=icoeff,
        dcoeff=dcoeff,
        lu_precision=lu_precision,
        trajectories_per_block=trajectories_per_block_or_default(
            trajectories_per_block
        ),
        sparsity=sparsity,
        ordering=ordering,
        tf_index=-1 if tf_index is None else int(tf_index),
        max_registers=max_registers,
        array_rhs=array_rhs,
        save_hook=save_hook,
        hook_size=int(hook_size),
        save_history=bool(save_history),
    )
    settings = dict(
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_steps=max_steps,
        return_stats=return_stats,
        error_weights=error_weights,
    )

    if save_hook is not None:
        if hook_size <= 0:
            raise ValueError("a save_hook needs a positive hook_size")
        # The hook's accumulator is a second output the vmap and JVP rules do
        # not know, so a hooked solve is the plain ensemble launch.
        return _solve_impl(ode_fn, y0, t_span, params, options=options, **settings)
    if not save_history:
        raise ValueError("save_history=False needs a save_hook to consume the saves")

    # The JVP rule wraps the vmap-aware solvers rather than the other way
    # round: custom_vmap's own JVP path instantiates symbolic zeros, which is
    # what tells the rule which sensitivity blocks it has to integrate.
    def solver_for(options):
        return make_custom_vmap_solver(
            functools.partial(_solve_impl, ode_fn, options=options, **settings),
            return_stats=return_stats,
        )

    primal_solver = solver_for(options)

    def joint_solver_for(spec):
        return solver_for(replace(options, spec=spec))

    return make_sensitivity_solver(
        primal_solver,
        joint_solver_for,
        jnp.shape(y0)[-1],
        jnp.shape(params)[-1],
        return_stats,
        sens_error_control,
        None
        if sens_param_columns is None
        else tuple(int(c) for c in sens_param_columns),
    )(y0, t_span, params)


def _solve_impl(
    ode_fn,
    y0,
    t_span,
    params,
    *,
    options: KernelOptions,
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100000,
    return_stats=False,
    error_weights=None,
):
    spec = options.spec
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

    launch = _make_jax_launch(ode_fn, n, n_vars, n_params, options)
    # No global scratch: the kernel keeps the state, the ten stage vectors and
    # df/dt in registers and thread-local memory.
    hist, accepted, rejected, loop_steps, hook_out = ensemble_ffi_call(
        launch,
        (y0_arr, times, params_arr, weights_arr),
        (),
        n=n,
        n_vars=n_system,
        n_save=n_save,
        dt0=dt0,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        n_save_hist=n_save if options.save_history else 1,
        hook_size=max(1, options.hook_size),
    )
    result = (hist,) if options.save_hook is None else (hist, hook_out)
    if return_stats:
        result += (solver_stats(accepted, rejected, loop_steps),)
    return result[0] if len(result) == 1 else result
