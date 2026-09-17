"""Cooperative packed-batch Rodas5P custom kernel using numba-cuda."""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import numpy as np
from numba_cuda_mlir import cuda, types

from solvers._jax_common import make_custom_vmap_solver, normalize_y0_params
from solvers._jax_numba_custom_call import make_launch
from solvers._numba_common import (
    SOLVER_ARGTYPES,
    PreparedNumbaSolve,
    build_error_weights,
    copy_workspace_inputs,
    ensemble_ffi_call,
    get_workspace,
    initial_step,
    make_cuda_local_vector_writer,
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
    make_augmented_local_writer,
    make_second_tangent,
    make_sensitivity_solver,
    make_tangent,
    seed_table,
)
from solvers._sparsity import (
    CompressedJacobian,
    colour_sparsity,
    dense_jacobian,
    normalize_sparsity,
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


class _DenseLUSolver:
    """The default solver, in the same shape a caller's own would take."""

    def __init__(self, factorize_local, solve_local):
        self.factorize_local = factorize_local
        self.solve_local = solve_local


@functools.cache
def dense_lu_solver(n_vars: int):
    """Dense LU with partial pivoting, one system per thread.

    This is the default linear solver, and it is an ordinary instance of the
    protocol :func:`check_linear_solver` documents rather than a privileged
    path: right-looking LU over the thread's own row-major buffer, then the two
    triangular solves in place.

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

    return _DenseLUSolver(factorize_local, solve_local)


_LITERAL_SEED_SOURCES = 0

# Direction sets per jvp call. Several seeds in one call share one Enzyme entry
# function, so once nvJitLink has inlined it the primal work the sweeps have in
# common -- for an ODE whose coefficients depend on t alone, all of it -- is one
# computation for LLVM to CSE rather than one per sweep. On DISCO-EB that was
# the difference between 580 ms and 509 ms at N128, on top of the literal
# seeds. numba-enzyme accepts up to 8 sets per call (its _MAX_DIRECTIONS), so
# 12 colours take two calls; the cap here mirrors that limit.
SEED_BATCH = 8
_MAX_SEED_BATCH = 8


def _make_literal_seed_jacobian(*, n_vars, n_params, n_colours, store_table, namespace):
    """Generate the Jacobian writer with every colour's seed row spelled out.

    ``store_table`` is the layout's destination table, ``(row, colour) ->
    slot`` with ``-1`` for none, or ``None`` for the plain colour grid where
    entry ``(row, g)`` sits at ``row * n_colours + g``.
    """
    global _LITERAL_SEED_SOURCES
    import linecache

    zero_row = n_colours
    batch = max(1, min(_MAX_SEED_BATCH, SEED_BATCH))

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
    source = "\n".join(lines) + "\n"
    _LITERAL_SEED_SOURCES += 1
    filename = f"<modax literal-seed jacobian {_LITERAL_SEED_SOURCES}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    code = compile(source, filename, "exec")
    ns = dict(namespace)
    exec(code, ns)
    return ns["cuda"].jit(device=True)(ns["write_negated_jacobian"])


def check_linear_solver(linear_solver):
    """Validate a linear solver against the documented protocol.

    A solver owns neither the buffer it works on nor the Jacobian that fills
    it: the kernel allocates one trajectory's matrix in that thread's own
    memory from a :class:`CompressedJacobian` layout, and Enzyme's colour sweeps
    fill it. What is left is two device functions, each called by the single
    thread that owns the trajectory, so neither may synchronise:

    ``factorize_local(lu, ipiv)``
        factorise ``lu`` in place, recording pivots in ``ipiv``.
    ``solve_local(lu, ipiv, rhs)``
        solve ``M x = rhs`` in place.

    A solver may also declare ``ipiv_size`` when it pivots something smaller
    than the whole state; the default is ``n_vars``.

    :func:`dense_lu_solver` is the default and satisfies this like any other.
    """
    required = ("factorize_local", "solve_local")
    missing = [name for name in required if not hasattr(linear_solver, name)]
    if missing:
        raise TypeError(
            f"{type(linear_solver).__name__} is not a Rodas5P linear solver: it is "
            f"missing {', '.join(missing)}. See check_linear_solver for the protocol."
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
    trajectories_per_block: int = _DEFAULT_TRAJECTORIES_PER_BLOCK,
    spec: SensitivitySpec | None = None,
    linear_solver=None,
    compressed: CompressedJacobian | None = None,
    tf_index: int = -1,
    max_registers: int | None = None,
    array_rhs=None,
):
    """Rodas5P, one trajectory per CUDA thread.

    A thread owns its trajectory outright: the state, the ten stage vectors,
    ``df/dt``, the step controller *and* the iteration matrix are all its own
    local arrays and scalars. Nothing is shared and nothing synchronises inside
    a step, so neither the state dimension nor the matrix bounds the block's
    trajectory count, and each thread writes its own Jacobian straight into its
    own matrix -- the ``O(n_vars**2)`` assembly is parallel across trajectories
    rather than across lanes of one.

    The linear solver is a parameter, not a branch: the default
    :func:`dense_lu_solver` satisfies the same protocol a caller's own does, so
    there is one code path either way, over a buffer laid out by ``compressed``
    -- which for an uninformative sparsity pattern *is* the dense matrix.
    """
    e1 = EXPONENT * (icoeff + pcoeff + dcoeff)
    e2 = -EXPONENT * (pcoeff + 2.0 * dcoeff)
    e3 = EXPONENT * dcoeff
    # The state, right-hand side, Jacobian and error estimate are always
    # float64; lu_dtype governs only the iteration matrix and its solves. The
    # Rosenbrock--Wanner order conditions hold under an approximate Jacobian, so
    # an FP32 factorisation does not reduce the method's order.
    lu_dtype = np.float32 if lu_precision == "fp32" else np.float64
    # cuda.local.array wants the numba type rather than the numpy dtype.
    lu_local_dtype = types.float32 if lu_precision == "fp32" else types.float64
    structure = dense_jacobian(n_vars) if compressed is None else compressed
    # With a spec the kernel integrates the joint [y, S] system, so the state,
    # stage and error extents become the augmented ones. The matrix does not:
    # the joint iteration matrix is block lower triangular with the same
    # M0 = I/(h*gamma) - J on every diagonal block, so one factorisation of the
    # n_vars block serves the state and every sensitivity column.
    size = n_vars if spec is None else spec.n_aug
    n_error = n_vars if spec is None else spec.n_error
    n_sens = 0 if spec is None else spec.n_sens

    tpb = int(trajectories_per_block)
    n_colours = structure.n_colours

    # A pattern and a linear solver are separable decisions. The pattern buys
    # the *sweeps*: n_colours of them instead of n_vars, which is the expensive
    # half and costs the caller nothing but the pattern. Storing the result
    # compressed buys the *space*, but only a solver written against that
    # layout can read it back. With no such solver the sweeps stay compressed
    # and their results are scattered into an ordinary row-major matrix, which
    # dense_lu_solver factorises as it always has -- the cheap win without the
    # expensive obligation.
    expand_to_dense = linear_solver is None and not structure.is_row_major_dense
    if expand_to_dense:
        lu_size = n_vars * n_vars
        diag_table = np.arange(n_vars, dtype=np.int32) * (n_vars + 1)
        store_table = structure.dense_slots()
    else:
        lu_size = structure.size
        diag_table = np.asarray(structure.diagonal, dtype=np.int32)
        store_table = structure.store_slots()
    # Only a layout with slots nothing writes needs clearing first. The grid
    # writes every slot and a packed layout has one slot per entry, so both
    # come out fully defined; only the dense expansion has structural zeros.
    needs_clear = store_table is not None and lu_size > int((store_table >= 0).sum())

    if linear_solver is None:
        linear_solver = dense_lu_solver(n_vars)
    check_linear_solver(linear_solver)
    factorize_local = linear_solver.factorize_local
    solve_local = linear_solver.solve_local
    # How many pivots the factorisation records is the solver's business, not
    # the layout's: a bordered-block solver pivots only its dense core.
    ipiv_per = int(getattr(linear_solver, "ipiv_size", n_vars))

    if spec is not None:
        ode_write = make_augmented_local_writer(ode_fn, spec)
    elif array_rhs is not None:
        # The primal stage evaluations need no tuple form: that exists for
        # Enzyme, which only ever sees ode_fn. A caller whose right-hand side
        # also comes as ``f(y, t, p, out)`` over arrays can hand that in, and
        # the eight stage evaluations per step call it directly.
        from solvers._numba_common import as_cuda_device as _as_device

        _array_rhs = _as_device(array_rhs)

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

    jit_options = {} if max_registers is None else {"max_registers": max_registers}

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

        for j in range(size):
            val = y0[i, j]
            y[j] = val
            hist[i, 0, j] = val

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

            # Stage 1
            ode_write(y, t, p_row, work)
            for j in range(size):
                put_stage_rhs(rhs_buf, work, j, work[j] + dt_use * D1 * dT[j])
            stage_solve(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                0,
                work,
                y,
                p_row,
                t,
            )
            for j in range(size):
                u[j] = y[j] + A21 * k_stages[0, j]

            # Stage 2
            ode_write(u, t + C2 * dt_use, p_row, work)
            for j in range(size):
                stage_rhs = (
                    work[j] + dt_use * D2 * dT[j] + C21 * k_stages[0, j] * inv_dt
                )
                put_stage_rhs(rhs_buf, work, j, stage_rhs)
            stage_solve(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                1,
                work,
                y,
                p_row,
                t,
            )
            for j in range(size):
                u[j] = y[j] + (A31 * k_stages[0, j] + A32 * k_stages[1, j])

            # Stage 3
            ode_write(u, t + C3 * dt_use, p_row, work)
            for j in range(size):
                stage_rhs = (
                    work[j]
                    + dt_use * D3 * dT[j]
                    + (C31 * k_stages[0, j] + C32 * k_stages[1, j]) * inv_dt
                )
                put_stage_rhs(rhs_buf, work, j, stage_rhs)
            stage_solve(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                2,
                work,
                y,
                p_row,
                t,
            )
            for j in range(size):
                u[j] = y[j] + (
                    A41 * k_stages[0, j] + A42 * k_stages[1, j] + A43 * k_stages[2, j]
                )

            # Stage 4
            ode_write(u, t + C4 * dt_use, p_row, work)
            for j in range(size):
                stage_rhs = (
                    work[j]
                    + dt_use * D4 * dT[j]
                    + (
                        C41 * k_stages[0, j]
                        + C42 * k_stages[1, j]
                        + C43 * k_stages[2, j]
                    )
                    * inv_dt
                )
                put_stage_rhs(rhs_buf, work, j, stage_rhs)
            stage_solve(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                3,
                work,
                y,
                p_row,
                t,
            )
            for j in range(size):
                u[j] = y[j] + (
                    A51 * k_stages[0, j]
                    + A52 * k_stages[1, j]
                    + A53 * k_stages[2, j]
                    + A54 * k_stages[3, j]
                )

            # Stage 5
            ode_write(u, t + C5 * dt_use, p_row, work)
            for j in range(size):
                stage_rhs = (
                    work[j]
                    + dt_use * D5 * dT[j]
                    + (
                        C51 * k_stages[0, j]
                        + C52 * k_stages[1, j]
                        + C53 * k_stages[2, j]
                        + C54 * k_stages[3, j]
                    )
                    * inv_dt
                )
                put_stage_rhs(rhs_buf, work, j, stage_rhs)
            stage_solve(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                4,
                work,
                y,
                p_row,
                t,
            )
            for j in range(size):
                u[j] = y[j] + (
                    A61 * k_stages[0, j]
                    + A62 * k_stages[1, j]
                    + A63 * k_stages[2, j]
                    + A64 * k_stages[3, j]
                    + A65 * k_stages[4, j]
                )

            # Stage 6
            ode_write(u, t_end, p_row, work)
            for j in range(size):
                stage_rhs = (
                    work[j]
                    + (
                        C61 * k_stages[0, j]
                        + C62 * k_stages[1, j]
                        + C63 * k_stages[2, j]
                        + C64 * k_stages[3, j]
                        + C65 * k_stages[4, j]
                    )
                    * inv_dt
                )
                put_stage_rhs(rhs_buf, work, j, stage_rhs)
            stage_solve(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                5,
                work,
                y,
                p_row,
                t,
            )
            for j in range(size):
                u[j] += k_stages[5, j]

            # Stage 7
            ode_write(u, t_end, p_row, work)
            for j in range(size):
                stage_rhs = (
                    work[j]
                    + (
                        C71 * k_stages[0, j]
                        + C72 * k_stages[1, j]
                        + C73 * k_stages[2, j]
                        + C74 * k_stages[3, j]
                        + C75 * k_stages[4, j]
                        + C76 * k_stages[5, j]
                    )
                    * inv_dt
                )
                put_stage_rhs(rhs_buf, work, j, stage_rhs)
            stage_solve(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                6,
                work,
                y,
                p_row,
                t,
            )
            for j in range(size):
                u[j] += k_stages[6, j]

            # Stage 8
            ode_write(u, t_end, p_row, work)
            for j in range(size):
                stage_rhs = (
                    work[j]
                    + (
                        C81 * k_stages[0, j]
                        + C82 * k_stages[1, j]
                        + C83 * k_stages[2, j]
                        + C84 * k_stages[3, j]
                        + C85 * k_stages[4, j]
                        + C86 * k_stages[5, j]
                        + C87 * k_stages[6, j]
                    )
                    * inv_dt
                )
                put_stage_rhs(rhs_buf, work, j, stage_rhs)
            stage_solve(
                lu_buf,
                ipiv_buf,
                rhs_buf,
                k_stages,
                7,
                work,
                y,
                p_row,
                t,
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
                        hist[i, save_idx, j] = theta1 * y[j] + theta * (
                            y_new_j + theta1 * (h1 + theta * (h2 + theta * h3))
                        )
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
            for j in range(size):
                hist[i, save_idx, j] = y[j]
            save_idx += 1
        accepted_out[i] = accepted
        rejected_out[i] = rejected
        loop_out[i] = n_steps

    return kernel, tpb


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
    trajectories_per_block=None,
):
    y0_arr, times, params_arr, dt0 = _normalize_inputs(y0, t_span, params, first_step)
    n, n_vars = y0_arr.shape
    n_save = times.shape[0]
    n_params = params_arr.shape[1]
    weights_arr = build_error_weights(error_weights, n, n_vars)
    trajectories_per_block = trajectories_per_block_or_default(trajectories_per_block)

    # No global scratch: every per-trajectory vector is thread-local.
    workspace = get_workspace(
        _WORKSPACE_CACHE, n, n_vars, n_save, n_params, transposed=False, n_work=0
    )
    copy_workspace_inputs(workspace, y0_arr, times, params_arr, weights_arr)

    kernel, trajectories_per_block = _make_kernel(
        ode_fn,
        n_vars,
        n_params,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        trajectories_per_block,
    )
    threads = (trajectories_per_block, 1, 1)
    blocks = (n + trajectories_per_block - 1) // trajectories_per_block

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
    trajectories_per_block: int = _DEFAULT_TRAJECTORIES_PER_BLOCK,
    spec: SensitivitySpec | None = None,
    linear_solver=None,
    compressed: CompressedJacobian | None = None,
    tf_index: int = -1,
    max_registers: int | None = None,
    array_rhs=None,
):
    kernel, trajectories_per_block = _make_kernel(
        ode_fn,
        n_vars,
        n_params,
        pcoeff,
        icoeff,
        dcoeff,
        lu_precision,
        trajectories_per_block,
        spec,
        linear_solver,
        compressed,
        tf_index,
        max_registers,
        array_rhs,
    )
    blocks = (n + trajectories_per_block - 1) // trajectories_per_block
    return make_launch(
        kernel,
        SOLVER_ARGTYPES,
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
    linear_solver=None,
    sparsity=None,
    tf_index=None,
    max_registers=None,
    array_rhs=None,
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
    the state (see ``solvers/_sensitivity.py``). ``jax.jvp``, ``jax.jacfwd``,
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

    ``sparsity`` is where a structured problem pays off. Forward-mode AD
    returns ``J v``, not ``J``, so the Jacobian costs one sweep per column
    unless columns can be seeded together -- and columns sharing no row can be,
    since their contributions never collide. Colouring the column intersection
    graph (:mod:`solvers._sparsity`) finds the fewest such groups; the Jacobian
    then costs ``n_colours + 1`` sweeps and is stored column-compressed,
    ``n_vars x n_colours``. Pass an ``(n_vars, n_vars)`` mask, a scipy sparse
    matrix, or an ``(nnz, 2)`` array of indices. The default -- no pattern --
    colours every column apart, which is the dense matrix at the dense cost, so
    this is one mechanism rather than two paths.

    The pattern must be a **superset** of the true nonzeros. Colouring a
    superset only costs sweeps; colouring a subset silently corrupts the entries
    where two columns of a group do overlap after all. For a structured
    ``linear_solver`` the pattern to give is everything its factorisation reads
    or writes, fill-in included: a superset by construction, and it leaves the
    factors room in the same buffer.

    A pattern on its own is enough: with no ``linear_solver`` the sweeps stay
    compressed and each one is scattered into an ordinary row-major matrix,
    which the default :func:`dense_lu_solver` factorises. That keeps the saving
    a pattern is mostly there for -- ``n_colours + 1`` sweeps rather than
    ``n_vars + 1`` -- at a dense matrix's storage, and asks nothing of the
    caller. Passing a solver is what turns the pattern into a saving in space
    and factorisation cost as well.

    A ``CompressedJacobian`` may be passed as ``sparsity`` in place of a
    pattern -- which is what a caller with its own solver should do, since the
    layout it bound the solver to is then the layout the kernel uses. Running
    it through :func:`pack` first trades the grid's straight write for a
    scatter and gets one slot per declared entry instead of
    ``n_vars * n_colours``, which is per-thread local memory saved.

    ``linear_solver`` replaces the default :func:`dense_lu_solver` with a
    factorisation that exploits that layout -- see :func:`check_linear_solver`.
    It owns neither the buffer nor the Jacobian, so it is two device functions,
    and it may declare ``ipiv_size`` if it pivots something smaller than the
    whole state. Forward sensitivities work with one: the joint iteration
    matrix is block lower triangular with the same ``M0`` on every diagonal
    block, so the solver only ever factorises the ``n_vars`` block it was
    written for and the coupling between blocks is a forward substitution the
    kernel does itself.

    ``tf_index`` names a column of ``params`` holding each trajectory's own end
    time, for ensembles whose members finish at different times; save times
    past a trajectory's end hold its final state. The default ``None`` ends
    every trajectory at ``t_span[-1]``.

    ``trajectories_per_block`` is one thread's worth of work each, defaulting to
    a warp; nothing on chip bounds it, since every per-trajectory buffer is
    thread-local. ``max_registers`` caps the
    kernel's per-thread register count, trading spills against occupancy.
    """
    n_vars = jnp.shape(y0)[-1]
    if linear_solver is not None:
        check_linear_solver(linear_solver)
    if sparsity is None:
        compressed = dense_jacobian(n_vars)
    elif isinstance(sparsity, CompressedJacobian):
        # A caller with a structured solver has already built the layout to
        # bind it to; taking that object back is what guarantees the two agree.
        compressed = sparsity
    else:
        compressed = colour_sparsity(normalize_sparsity(sparsity, n_vars))
    trajectories_per_block = trajectories_per_block_or_default(trajectories_per_block)

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
        trajectories_per_block=trajectories_per_block,
        linear_solver=linear_solver,
        compressed=compressed,
        tf_index=-1 if tf_index is None else int(tf_index),
        array_rhs=array_rhs,
        max_registers=max_registers,
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
    trajectories_per_block: int = _DEFAULT_TRAJECTORIES_PER_BLOCK,
    spec=None,
    linear_solver=None,
    compressed: CompressedJacobian | None = None,
    tf_index: int = -1,
    max_registers: int | None = None,
    array_rhs=None,
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
        trajectories_per_block,
        spec,
        linear_solver,
        compressed,
        tf_index,
        max_registers,
        array_rhs,
    )
    # No global scratch: the kernel keeps the state, the ten stage vectors and
    # df/dt in registers and thread-local memory.
    hist, accepted, rejected, loop_steps = ensemble_ffi_call(
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
    )
    if not return_stats:
        return hist
    return hist, solver_stats(accepted, rejected, loop_steps)
