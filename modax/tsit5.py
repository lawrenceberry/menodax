"""Tsit5 custom kernel using numba-cuda: one thread per trajectory, storage on chip or local."""

from __future__ import annotations

import functools
import gc
import math

import jax.numpy as jnp
from numba_cuda_mlir import cuda, types

from modax._jax_common import make_custom_vmap_solver, normalize_y0_params
from modax._jax_numba_custom_call import make_launch
from modax._numba_common import (
    SOLVER_ARGTYPES,
    build_error_weights,
    ensemble_ffi_call,
    initial_step,
    make_cuda_local_vector_writer,
    solver_stats,
)
from modax._sensitivity import (
    SensitivitySpec,
    augmented_error_weights,
    augmented_y0,
    make_augmented_local_writer,
    make_sensitivity_solver,
)
from modax._sensitivity import (
    clear_caches as clear_sensitivity_caches,
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

# One thread per trajectory in both kernels; they differ only in where the nine
# per-trajectory vectors (the state, the trial state and the seven stages) live.
#
# The thread-local kernel keeps them in the thread's own ``cuda.local`` arrays,
# which the compiler promotes to registers where the indexing is resolvable and
# otherwise places in local memory: off-chip DRAM, cached in L2, its latency
# hidden by occupancy once the device is saturated. Nothing on chip bounds its
# block size, but a block is a warp anyway, as in Rodas5P: a small ensemble
# then spreads over as many SMs as it has warps instead of sitting on one, and
# a large one is indifferent (Lorenz at 1000 trajectories, measured: 3.2 ms in
# 32-thread blocks against 6.9 ms in 128-thread blocks; 83 ms either way at
# 100000).
#
# The shared-memory kernel keeps the same nine vectors in per-block shared
# memory, laid out ``(n_vars, _SHARED_BLOCK)`` so that consecutive lanes hold
# consecutive words and a warp's access is bank-conflict-free. It costs
# ``9 * n_vars * _SHARED_BLOCK * 8`` bytes of static shared memory per block,
# which caps ``n_vars``, and by capping the blocks an SM can hold it loses to
# the thread-local kernel once the ensemble saturates the device: up to 14%
# at 100000 trajectories of 8 to 16 components. Below that the two are within
# noise of each other on an RTX 4070 SUPER -- at these sizes the local arrays
# are promoted to registers, and the on-chip copy buys nothing over them --
# so ``"auto"`` takes shared wherever the system fits and the ensemble is
# small enough, where it is never slower, and local beyond. ``backend``
# overrides the choice.
_LOCAL_BLOCK = 32
_SHARED_BLOCK = 32
_SHARED_MAX_NVARS = 16
_SHARED_MAX_ENSEMBLE = 16384


def _use_shared_backend(n: int, n_system: int, backend: str) -> bool:
    if backend == "local":
        return False
    if backend == "shared":
        if n_system > _SHARED_MAX_NVARS:
            raise ValueError(
                "shared backend requires a system size <= "
                f"{_SHARED_MAX_NVARS}; got {n_system}. A solve carrying forward "
                "sensitivities integrates n_vars * (1 + n_sens) components, so "
                "it may need backend='local' where the plain solve does not"
            )
        return True
    if backend != "auto":
        raise ValueError(
            f"backend must be 'auto', 'shared', or 'local'; got {backend!r}"
        )
    return n_system <= _SHARED_MAX_NVARS and n <= _SHARED_MAX_ENSEMBLE


def clear_caches() -> None:
    """Drop the compiled kernels.

    Useful when sweeping problem sizes in a single process: each unique
    ``n_vars`` compiles a separate kernel, and nothing releases it because the
    module-level caches hold it.
    """
    _make_body.cache_clear()
    _make_kernel.cache_clear()
    _make_jax_launch.cache_clear()
    clear_sensitivity_caches()
    gc.collect()


@functools.cache
def _make_body(
    ode_fn,
    n_vars: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    spec: SensitivitySpec | None = None,
):
    """Build the per-trajectory Tsit5 integration loop as a CUDA device function.

    ``i`` is the trajectory, indexing the ensemble's global arrays (``y0``,
    ``params``, ``weights``, ``hist`` and the counters). ``y``, ``u`` and
    ``k1``..``k7`` are the nine per-trajectory vectors, each a 1-D array of
    ``n_system`` components that the body reads and writes as its own: the
    kernels below hand it either the thread's ``cuda.local`` arrays or one
    column of a per-block ``cuda.shared`` array, and the body is the same
    either way, which is what keeps the two bit-identical.
    """
    # PID step-control exponents (Soderlind). Defaults (0, 1, 0) give E1=EXPONENT
    # and E2=E3=0, recovering the elementary I-controller exactly.
    e1 = EXPONENT * (icoeff + pcoeff + dcoeff)
    e2 = -EXPONENT * (pcoeff + 2.0 * dcoeff)
    e3 = EXPONENT * dcoeff
    # With a sensitivity spec the loop integrates the joint [y, S] system: the
    # writer emits [f, J_y S + J_p] and every stage vector, the error norm and
    # the dense-output write run over n_aug components instead of n_vars. The
    # integrator itself is unchanged -- forward sensitivities are just a larger
    # ODE, which is what makes them cheap to bolt onto an existing kernel.
    if spec is None:
        ode_write = make_cuda_local_vector_writer(ode_fn, n_vars)
        n_system = n_vars
    else:
        ode_write = make_augmented_local_writer(ode_fn, spec)
        n_system = spec.n_aug
    n_error = n_vars if spec is None else spec.n_error

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
    ):
        prow = params[i]
        for j in range(n_system):
            y[j] = y0[i, j]
            hist[i, 0, j] = y0[i, j]
            k7[j] = 0.0

        n_save = times.shape[0]
        t = times[0]
        tf = times[n_save - 1]
        # A non-positive dt0 is the "no first step given" sentinel: start from
        # 1e-6 of the integration window.
        dt = dt0 if dt0 > 0.0 else (tf - t) * 1e-6
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
                for j in range(n_system):
                    k1[j] = k7[j]
            else:
                ode_write(y, t, prow, k1)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (A21 * k1[j])
            ode_write(u, t + C2 * dt_use, prow, k2)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (A31 * k1[j] + A32 * k2[j])
            ode_write(u, t + C3 * dt_use, prow, k3)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (A41 * k1[j] + A42 * k2[j] + A43 * k3[j])
            ode_write(u, t + C4 * dt_use, prow, k4)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (
                    A51 * k1[j] + A52 * k2[j] + A53 * k3[j] + A54 * k4[j]
                )
            ode_write(u, t + C5 * dt_use, prow, k5)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (
                    A61 * k1[j] + A62 * k2[j] + A63 * k3[j] + A64 * k4[j] + A65 * k5[j]
                )
            ode_write(u, t + C6 * dt_use, prow, k6)

            for j in range(n_system):
                u[j] = y[j] + dt_use * (
                    B1 * k1[j]
                    + B2 * k2[j]
                    + B3 * k3[j]
                    + B4 * k4[j]
                    + B5 * k5[j]
                    + B6 * k6[j]
                )
            ode_write(u, t + C7 * dt_use, prow, k7)

            err_sum = 0.0
            for j in range(n_system):
                err_est = dt_use * (
                    E1 * k1[j]
                    + E2 * k2[j]
                    + E3 * k3[j]
                    + E4 * k4[j]
                    + E5 * k5[j]
                    + E6 * k6[j]
                    + E7 * k7[j]
                )
                scale = atol + rtol * max(abs(y[j]), abs(u[j]))
                r = weights[i, j] * err_est / scale
                err_sum += r * r
            err_norm = math.sqrt(err_sum / n_error)
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
                    for j in range(n_system):
                        hist[i, save_idx, j] = y[j] + dt_use * (
                            b1 * k1[j]
                            + b2 * k2[j]
                            + b3 * k3[j]
                            + b4 * k4[j]
                            + b5 * k5[j]
                            + b6 * k6[j]
                            + b7 * k7[j]
                        )
                    save_idx += 1
                for j in range(n_system):
                    y[j] = u[j]
                accepted_steps += 1
                has_fsal = True
            else:
                rejected_steps += 1
                for j in range(n_system):
                    k7[j] = 0.0
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
    spec: SensitivitySpec | None = None,
    shared: bool = False,
):
    """Build a Tsit5 kernel: one thread per trajectory, storage chosen by ``shared``.

    Either way the launch carries no scratch and ``y0``/``weights``/``hist``
    keep their natural ``(n, ...)`` layouts. ``shared=False`` allocates the
    nine per-trajectory vectors as the thread's own ``cuda.local`` arrays, as
    in Rodas5P. ``shared=True`` allocates them as nine ``(n_system,
    _SHARED_BLOCK)`` ``cuda.shared`` arrays and hands the body the thread's
    column of each: a strided 1-D view, which the body indexes exactly as it
    does a local array. Every thread touches only its own column, so nothing
    synchronises and a thread past the ensemble's end simply leaves.
    """
    body = _make_body(ode_fn, n_vars, pcoeff, icoeff, dcoeff, spec)
    n_system = n_vars if spec is None else spec.n_aug

    if not shared:

        @cuda.jit
        def local_kernel(
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
            i = cuda.grid(1)  # ty: ignore[unresolved-attribute]
            if i >= y0.shape[0]:
                return
            y = cuda.local.array(n_system, types.float64)
            u = cuda.local.array(n_system, types.float64)
            k1 = cuda.local.array(n_system, types.float64)
            k2 = cuda.local.array(n_system, types.float64)
            k3 = cuda.local.array(n_system, types.float64)
            k4 = cuda.local.array(n_system, types.float64)
            k5 = cuda.local.array(n_system, types.float64)
            k6 = cuda.local.array(n_system, types.float64)
            k7 = cuda.local.array(n_system, types.float64)
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
            )

        return local_kernel

    shape = (n_system, _SHARED_BLOCK)

    # The column views are taken here, one device function down from the
    # kernel, and not in the kernel itself: there the shared arrays' shapes are
    # static and numba-cuda-mlir mis-types the slice (a ``memref.collapse_shape``
    # whose stride it expects static and emits dynamic, which fails
    # verification). As a device-function argument the array is a dynamic
    # memref and the slice lowers as it should. Inlining folds the layer away.
    @cuda.jit(device=True)
    def run_column(
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
        tx,
        i,
    ):
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
            y[:, tx],
            u[:, tx],
            k1[:, tx],
            k2[:, tx],
            k3[:, tx],
            k4[:, tx],
            k5[:, tx],
            k6[:, tx],
            k7[:, tx],
            i,
        )

    @cuda.jit
    def shared_kernel(
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
        i = cuda.grid(1)  # ty: ignore[unresolved-attribute]
        tx = cuda.threadIdx.x  # ty: ignore[unresolved-attribute]
        if i >= y0.shape[0]:
            return
        y = cuda.shared.array(shape, types.float64)
        u = cuda.shared.array(shape, types.float64)
        k1 = cuda.shared.array(shape, types.float64)
        k2 = cuda.shared.array(shape, types.float64)
        k3 = cuda.shared.array(shape, types.float64)
        k4 = cuda.shared.array(shape, types.float64)
        k5 = cuda.shared.array(shape, types.float64)
        k6 = cuda.shared.array(shape, types.float64)
        k7 = cuda.shared.array(shape, types.float64)
        run_column(
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
            tx,
            i,
        )

    return shared_kernel


@functools.cache
def _make_jax_launch(
    ode_fn,
    n: int,
    n_vars: int,
    pcoeff: float = 0.0,
    icoeff: float = 1.0,
    dcoeff: float = 0.0,
    spec: SensitivitySpec | None = None,
    shared: bool = False,
):
    """Compile, load and size the kernel behind one JAX-side ensemble launch."""
    kernel = _make_kernel(ode_fn, n_vars, pcoeff, icoeff, dcoeff, spec, shared)
    threads = _SHARED_BLOCK if shared else _LOCAL_BLOCK
    blocks = (n + threads - 1) // threads
    return make_launch(kernel, SOLVER_ARGTYPES, grid=blocks, block=threads)


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
    backend="auto",
    sens_error_control=True,
    sens_param_columns=None,
):
    """JAX-callable Tsit5 custom-kernel solve.

    ``backend`` chooses where the kernel keeps the state and its stage vectors:
    ``"shared"`` in per-block shared memory, ``"local"`` in the thread's own
    local memory. The two are bit-identical; shared is faster where the device
    is under-occupied (small ensembles, low dimension) and local where it is
    saturated, and ``"auto"`` picks by the ensemble's size and the system's,
    taking shared whenever the system fits and the ensemble is small enough.

    The solve is an XLA custom call into the numba-cuda kernel, so it carries a
    ``jax.custom_jvp`` rule rather than being differentiated by XLA: asking for
    a derivative integrates the continuous forward-sensitivity system alongside
    the state (see ``modax/_sensitivity.py``). ``jax.jvp``, ``jax.jacfwd``,
    ``jax.grad``, ``jax.jacrev`` and ``jax.value_and_grad`` all work with
    respect to ``y0`` and ``params``; ``t_span`` is not differentiable. An
    undifferentiated call runs the plain kernel and pays nothing.

    ``sens_error_control`` decides whether the sensitivity components take part
    in the step-size error norm. The default ``True`` controls them to the same
    ``rtol``/``atol`` as the state, so the gradient is as accurate as the value.
    ``False`` drops them from the norm, which makes the joint solve take exactly
    the step sequence the plain solve takes -- the value then matches a plain
    call bit for bit -- at the cost of nothing tying the sensitivities' accuracy
    to ``rtol``.
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
        backend=backend,
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
            functools.partial(
                _solve_impl,
                ode_fn,
                spec=spec,
                **settings,
            ),
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
    backend="auto",
    spec=None,
):
    y0_arr, params_arr, n, n_vars = normalize_y0_params(y0, params)
    times = jnp.asarray(t_span, dtype=jnp.float64)
    n_save = times.shape[0]
    dt0 = initial_step(first_step)
    weights_host = build_error_weights(error_weights, n, n_vars)

    # With a spec the kernel integrates the joint [y, S] system, so every
    # per-component extent below is the augmented one.
    n_system = n_vars if spec is None else spec.n_aug
    if spec is not None:
        y0_arr = augmented_y0(y0_arr, spec)
        weights_host = augmented_error_weights(weights_host, spec)
    weights_arr = jnp.asarray(weights_host)

    uses_shared = _use_shared_backend(n, n_system, backend)
    launch = _make_jax_launch(
        ode_fn, n, n_vars, pcoeff, icoeff, dcoeff, spec, uses_shared
    )
    # No global scratch either way: the kernel keeps the state and the stage
    # vectors on chip or in thread-local memory.
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
