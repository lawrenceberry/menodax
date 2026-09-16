"""Continuous forward sensitivity analysis for the numba-cuda ensemble solvers.

The solvers reach JAX through an XLA FFI custom call, so the solve itself is
opaque to autodiff: there is no traced graph for JAX to differentiate and no
practical way to run Enzyme over the kernel (see "Why forward sensitivities" in
the README). Derivatives are therefore supplied by a ``jax.custom_jvp`` rule
that integrates the sensitivity system alongside the state.

Writing ``S(t) = dy(t)/dtheta`` for a direction block ``theta``, differentiating
``y' = f(t, y, p)`` with respect to ``theta`` gives the variational equation

    S' = J_y(t) S + J_p(t),    J_y = df/dy,  J_p = df/dtheta,

with ``J_p = 0`` and ``S(t0) = I`` for the initial-state block, and ``J_p =
df/dp`` and ``S(t0) = 0`` for the parameter block.  Stacking it under the state
gives one joint system

    d/dt [y, S] = [f(t, y, p), J_y S + J_p]

which the existing kernels integrate as an ordinary ODE of ``n_aug = n_vars *
(1 + n_sens)`` variables.  Because the rule returns the primal and the tangent
together, ``jax.value_and_grad`` costs one joint solve rather than a solve for
the value plus another for the derivative.

Layout
------
The augmented state is the state followed by one ``n_vars``-long block per
sensitivity direction::

    z[j]                        = y[j]
    z[n_vars + k * n_vars + r]  = S[r, k]

Direction ``k`` runs over the initial-state block first (``n_vars`` directions,
present only when ``y0`` is differentiated) and then the parameter block
(``n_params`` directions, present only when ``params`` is differentiated).  Only
the blocks JAX actually asks for are integrated, so differentiating with respect
to parameters alone does not pay for the ``n_vars`` initial-state columns.

``J_y`` and ``J_p`` both come from ``numba_enzyme.jacfwd_column`` applied to the
user's ``ode_fn``, which seeds one entry of the callback's *flattened* argument
list ``(y_0 ... y_n-1, t, p_0 ... p_m-1)``: columns ``0 .. n_vars - 1`` are the
Jacobian, column ``n_vars`` is ``df/dt`` (what Rodas5P already uses), and
columns ``n_vars + 1 ...`` are ``df/dp``.  So the parameter derivatives the
sensitivity system needs come from the same device function the stiff kernel
already builds, with no second derivatives and nothing extra from the caller.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero
from numba_cuda_mlir import cuda, types
from numba_enzyme import jacfwd_column, jvp

from solvers._numba_common import as_cuda_device


@dataclass(frozen=True)
class SensitivitySpec:
    """Which sensitivity blocks a joint solve carries.

    Hashable and used as part of the kernel cache keys, so a solve
    differentiated only with respect to ``params`` compiles (and integrates) a
    smaller augmented system than one differentiated with respect to both.

    ``error_control`` decides whether the sensitivity components take part in
    the step-size error norm; it belongs here rather than in the solve because
    the error-norm denominator it selects is a compile-time constant of the
    kernel.
    """

    n_vars: int
    n_params: int
    wrt_y0: bool
    wrt_params: bool
    error_control: bool = True

    def __post_init__(self):
        if not (self.wrt_y0 or self.wrt_params):
            raise ValueError(
                "a sensitivity spec must carry at least one direction block"
            )

    @property
    def n_y0_dirs(self) -> int:
        return self.n_vars if self.wrt_y0 else 0

    @property
    def n_param_dirs(self) -> int:
        return self.n_params if self.wrt_params else 0

    @property
    def n_sens(self) -> int:
        return self.n_y0_dirs + self.n_param_dirs

    @property
    def n_aug(self) -> int:
        """Size of the joint ``[y, S]`` system the kernel actually integrates."""
        return self.n_vars * (1 + self.n_sens)

    @property
    def n_error(self) -> int:
        """Component count the weighted-RMS error norm divides by.

        With the sensitivities excluded from step-size control their weights
        are zero, so they contribute exactly 0.0 to the sum; dividing by
        ``n_vars`` rather than ``n_aug`` then reproduces the plain solve's error
        norm bit for bit, and with it its exact step sequence.
        """
        return self.n_aug if self.error_control else self.n_vars


@functools.cache
def make_jacobian_column(ode_fn, n_vars: int, n_params: int):
    """One forward-mode column of ``ode_fn``'s flattened Jacobian.

    Column ``c < n_vars`` is ``df/dy_c``, column ``n_vars`` is ``df/dt``, and
    column ``n_vars + 1 + k`` is ``df/dp_k``.  The signature is required: an
    array cannot say how long the tuple it stands for is, and the kernels'
    own calls specialise ``ode_fn`` for array arguments instead.
    """
    return jacfwd_column(
        as_cuda_device(ode_fn),
        signature=types.UniTuple(types.float64, n_vars)(
            types.UniTuple(types.float64, n_vars),
            types.float64,
            types.UniTuple(types.float64, n_params),
        ),
    )


@functools.cache
def _primal_signature(n_vars: int, n_params: int):
    """The concrete ``ode_fn`` specialisation every derivative is built for."""
    return types.UniTuple(types.float64, n_vars)(
        types.UniTuple(types.float64, n_vars),
        types.float64,
        types.UniTuple(types.float64, n_params),
    )


@functools.cache
def make_tangent(ode_fn, n_vars: int, n_params: int):
    """``D f(y, t, p)[u]`` in one forward sweep, for an arbitrary direction.

    The sensitivity right-hand side ``J_y S_k + J_p_k`` *is* a directional
    derivative: seed ``(S_k, 0, e_k)`` and it comes out whole.  Assembling it
    from unit columns instead costs ``n_vars + 1`` sweeps where this costs one.
    """
    return jvp(as_cuda_device(ode_fn), signature=_primal_signature(n_vars, n_params))


@functools.cache
def make_second_tangent(ode_fn, n_vars: int, n_params: int):
    """``D2 f(y, t, p)[u, v] + D f(y, t, p)[w]``, forward over forward.

    The joint system's Jacobian is the derivative of a right-hand side that
    already contains first derivatives of ``ode_fn``, so its off-diagonal block
    holds second derivatives of the original problem: ``d2f/dy2`` contracted
    with ``S_k``, plus the mixed ``d2f/dy dp_k``.  Seeding ``u = (S_k, 0, e_k)``
    and ``v = (w, 0, 0)`` applies that block to ``w`` without forming it;
    seeding ``v = (0, 1, 0)`` returns the sensitivity block's ``dF/dt``.

    ``jvp`` composes with itself, so this is literally the forward sweep of the
    forward sweep -- numba-enzyme records the chain and emits both markers into
    one module, rather than trying to differentiate an already-compiled
    derivative.  Composing makes the derivative of the *whole* tangent map, a
    function of ``(x, u)``, so the call carries a fourth direction ``w`` for the
    inner direction's own variation; every caller here passes zero for it,
    which leaves the plain bilinear form.
    """
    signature = _primal_signature(n_vars, n_params)
    return jvp(jvp(as_cuda_device(ode_fn), signature=signature), signature=signature)


@functools.cache
def seed_table(spec: SensitivitySpec):
    """Read-only table every unit and zero direction is a window into.

    ``2 * L`` zeros with a single ``1.0`` at ``L``, so the window starting at
    ``L - k`` carries its ``1.0`` at position ``k``, and any window inside
    ``[0, L)`` is all zeros.  One table in constant memory replaces a per-thread
    one-hot buffer, which could not promote to registers because its store index
    is dynamic.
    """
    length = max(spec.n_vars, spec.n_params)
    table = np.zeros(2 * length, dtype=np.float64)
    table[length] = 1.0
    return table


@functools.cache
def make_augmented_transposed_writer(ode_fn, spec: SensitivitySpec):
    """Joint ``[f, J_y S + J_p]`` writer for Tsit5's transposed ``(n_aug, n)`` state.

    One thread owns a whole trajectory here, so the columns can be written
    straight out with no cross-lane coordination.  Each sensitivity direction
    costs exactly one forward sweep, whatever ``n_vars`` is: its right-hand
    side is the directional derivative of ``ode_fn`` seeded with
    ``(S_k, 0, e_k)``, so no Jacobian is ever formed.
    """
    n_vars = spec.n_vars
    n_sens = spec.n_sens
    n_y0_dirs = spec.n_y0_dirs
    n_params = spec.n_params
    length = max(n_vars, n_params)
    fn_device = as_cuda_device(ode_fn)
    tangent_of = make_tangent(ode_fn, n_vars, n_params)
    seeds = seed_table(spec)

    @cuda.jit(device=True)
    def write_vector(z, t, prow, out, s):
        # The callback and the derivative both read only the leading n_vars
        # entries of the column they are handed, so the augmented state can be
        # passed where the plain state is expected. The slices are strided; the
        # Enzyme entry point loads through the memref's own stride, so that is
        # fine.
        seed = cuda.const.array_like(seeds)
        zs = z[:, s]
        values = fn_device(zs, t, prow)
        for j in range(n_vars):
            out[j, s] = values[j]

        tangent = cuda.local.array(n_vars, types.float64)
        for k in range(n_sens):
            base = n_vars + k * n_vars
            # An initial-state direction has no parameter component; a
            # parameter direction seeds the unit vector e_(k - n_y0_dirs).
            start = 0 if k < n_y0_dirs else length - (k - n_y0_dirs)
            tangent_of(
                tangent,
                zs,
                t,
                prow,
                z[base : base + n_vars, s],
                0.0,
                seed[start : start + n_params],
            )
            for r in range(n_vars):
                out[base + r, s] = tangent[r]

    return write_vector


@functools.cache
def make_augmented_local_writer(ode_fn, spec: SensitivitySpec):
    """Joint ``[f, J_y S + J_p]`` writer for Rodas5P's thread-local state.

    Rodas5P runs one trajectory per thread, so the whole augmented vector --
    the state block and every sensitivity direction -- is written by the thread
    that owns it, out of and into its own local arrays.  The directions stay
    independent of one another (each is its own forward sweep seeded with
    ``(S_k, 0, e_k)``); nothing is shared, so there is no race and no
    synchronisation inside a device function whose callers invoke it under a
    divergent ``if running``.
    """
    n_vars = spec.n_vars
    n_sens = spec.n_sens
    n_y0_dirs = spec.n_y0_dirs
    n_params = spec.n_params
    length = max(n_vars, n_params)
    fn_device = as_cuda_device(ode_fn)
    tangent_of = make_tangent(ode_fn, n_vars, n_params)
    seeds = seed_table(spec)

    @cuda.jit(device=True)
    def write_vector(z_row, t, p_row, out):
        seed = cuda.const.array_like(seeds)
        values = fn_device(z_row, t, p_row)
        for j in range(n_vars):
            out[j] = values[j]

        tangent = cuda.local.array(n_vars, types.float64)
        for k in range(n_sens):
            base = n_vars + k * n_vars
            start = 0 if k < n_y0_dirs else length - (k - n_y0_dirs)
            tangent_of(
                tangent,
                z_row,
                t,
                p_row,
                z_row[base : base + n_vars],
                0.0,
                seed[start : start + n_params],
            )
            for r in range(n_vars):
                out[base + r] = tangent[r]

    return write_vector


def clear_caches() -> None:
    """Drop the cached device writers and Enzyme derivatives.

    Each ``(ode_fn, spec)`` pair compiles its own writer, and nothing releases
    them: the module-level caches hold them for the process's lifetime.
    """
    make_jacobian_column.cache_clear()
    make_tangent.cache_clear()
    make_second_tangent.cache_clear()
    seed_table.cache_clear()
    make_augmented_transposed_writer.cache_clear()
    make_augmented_local_writer.cache_clear()


def augmented_y0(y0_arr, spec: SensitivitySpec):
    """``(n, n_vars)`` initial state -> ``(n, n_aug)`` joint initial state.

    ``S(t0)`` is the identity for the initial-state block (``dy0/dy0``) and zero
    for the parameter block (the initial state does not depend on ``p``).
    """
    n = y0_arr.shape[0]
    blocks = [y0_arr]
    if spec.wrt_y0:
        eye = jnp.eye(spec.n_vars, dtype=y0_arr.dtype).reshape(1, -1)
        blocks.append(jnp.broadcast_to(eye, (n, spec.n_vars * spec.n_vars)))
    if spec.wrt_params:
        blocks.append(jnp.zeros((n, spec.n_vars * spec.n_params), y0_arr.dtype))
    return jnp.concatenate(blocks, axis=1)


def augmented_error_weights(weights_arr, spec: SensitivitySpec):
    """``(n, n_vars)`` error weights -> ``(n, n_aug)``.

    A zero weight drops a component from the step-size error norm, so
    ``spec.error_control=False`` makes the joint solve take exactly the step
    sequence the plain solve would -- the value from ``jax.value_and_grad`` is
    then bit-identical to the value from a plain call, and the sensitivities
    ride along on steps chosen for the state alone.  That is not the default:
    nothing then ties the sensitivities' accuracy to ``rtol``, and a stiff
    solver taking large steps on an easy state can be badly wrong about them.
    """
    tail = np.ones if spec.error_control else np.zeros
    return np.concatenate(
        [
            weights_arr,
            tail((weights_arr.shape[0], spec.n_aug - spec.n_vars), dtype=np.float64),
        ],
        axis=1,
    )


def split_augmented(hist, spec: SensitivitySpec):
    """``(n, n_save, n_aug)`` -> state ``(n, n_save, n_vars)`` and ``S`` ``(n, n_save, n_vars, n_sens)``."""
    n, n_save, _ = hist.shape
    state = hist[:, :, : spec.n_vars]
    sens = hist[:, :, spec.n_vars :].reshape(n, n_save, spec.n_sens, spec.n_vars)
    return state, jnp.swapaxes(sens, 2, 3)


def _match_shape(tangent, axis_size: int, name: str):
    """Broadcast an input tangent the way the primal argument was broadcast.

    ``normalize_y0_params`` broadcasts a 1-D argument across the ensemble, so
    the tangent has to be broadcast identically for the contraction to be the
    true directional derivative.  ``jnp.broadcast_to`` is linear, so reverse
    mode transposes it back to the sum over trajectories that a shared argument
    should receive.
    """
    arr = jnp.asarray(tangent)
    if arr.ndim == 1:
        return jnp.broadcast_to(arr, (axis_size,) + arr.shape)
    if arr.ndim != 2:
        raise ValueError(f"{name} tangent must be 1-D or 2-D; got shape {arr.shape}")
    return arr


def _zero_tangent(x):
    """A zero tangent of the type JAX expects for ``x``.

    Integer outputs (the step counters) take the empty ``float0`` type rather
    than a zero array of their own dtype.
    """
    if jnp.issubdtype(jnp.result_type(x), jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(jnp.shape(x), dtype=jax.dtypes.float0)


def make_sensitivity_solver(
    primal_solver,
    joint_solver_for,
    n_vars: int,
    n_params: int,
    return_stats: bool,
    sens_error_control: bool,
):
    """Attach a forward-sensitivity JVP rule to one solver implementation.

    ``primal_solver(y0, t_span, params)`` runs the plain solve.
    ``joint_solver_for(spec)`` returns the corresponding solver for the joint
    ``[y, S]`` system described by ``spec``; its output has the ordinary solver
    shape, with ``n_aug`` components in place of ``n_vars``.  Both are expected
    to be already wrapped for ``vmap``: the rule has to sit *outside*
    ``custom_vmap``, whose JVP path traces to a jaxpr and so instantiates every
    symbolic zero, which would hide exactly the information the rule uses to
    decide which sensitivity blocks to integrate.

    The rule fires only under a JAX differentiation transform, so an
    undifferentiated call pays nothing for sensitivities.  The tangent is formed
    by contracting ``S`` -- which depends only on the primal inputs -- with the
    input tangents.  That contraction is linear in the tangents, so JAX can
    transpose it: ``jax.grad`` and ``jax.jacrev`` work off the same rule as
    ``jax.jvp`` and ``jax.jacfwd``, with no adjoint solve.
    """

    @jax.custom_jvp
    def solve_fn(y0, t_span, params):
        return primal_solver(y0, t_span, params)

    @functools.partial(solve_fn.defjvp, symbolic_zeros=True)
    def solve_jvp(primals, tangents):
        y0, t_span, params = primals
        dy0, dt_span, dparams = tangents

        if not isinstance(dt_span, SymbolicZero):
            raise NotImplementedError(
                "differentiating a modax solve with respect to t_span is not "
                "supported; wrap the save times in jax.lax.stop_gradient, or "
                "differentiate with respect to y0 and/or params only"
            )

        wrt_y0 = not isinstance(dy0, SymbolicZero)
        wrt_params = not isinstance(dparams, SymbolicZero)
        if not (wrt_y0 or wrt_params):
            out = primal_solver(y0, t_span, params)
            return out, jax.tree_util.tree_map(_zero_tangent, out)

        spec = SensitivitySpec(n_vars, n_params, wrt_y0, wrt_params, sens_error_control)
        out = joint_solver_for(spec)(y0, t_span, params)
        hist, stats = out if return_stats else (out, None)
        state, sens = split_augmented(hist, spec)

        axis_size = state.shape[0]
        tangent = jnp.zeros_like(state)
        offset = 0
        if wrt_y0:
            tangent += jnp.einsum(
                "nsij,nj->nsi",
                sens[..., : spec.n_y0_dirs],
                _match_shape(dy0, axis_size, "y0"),
            )
            offset = spec.n_y0_dirs
        if wrt_params:
            tangent += jnp.einsum(
                "nsik,nk->nsi",
                sens[..., offset:],
                _match_shape(dparams, axis_size, "params"),
            )
        if stats is None:
            return state, tangent
        # Step counters are integers and carry no derivative, but JAX still
        # wants a tangent of the right (float0) type for every output.
        return (state, stats), (tangent, jax.tree_util.tree_map(_zero_tangent, stats))

    return solve_fn
