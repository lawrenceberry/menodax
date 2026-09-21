"""Shared Diffrax ensemble driver behind ``diffrax_tsit5`` and ``diffrax_kvaerno5``.

The public modules differ only in the Diffrax solver they instantiate and how
they build the step-size controller; everything else -- broadcasting ``y0``,
choosing ``dt0``, wrapping the tuple-form ``ode_fn`` and ``vmap``-ing over the
ensemble -- lives here. The callers keep their own ``jax.jit`` decoration so
that this function is only ever traced inside one of them.
"""

import diffrax
import jax
import jax.numpy as jnp


def solve_ensemble(
    ode_fn,
    y0,
    t_span,
    params,
    solver,
    controller,
    *,
    first_step,
    max_steps,
):
    """Integrate every trajectory of an ensemble with one Diffrax solver.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``.
        It may return a tuple (the unified RHS format); the result is coerced
        to an array to match ``y0``'s pytree structure.
    y0 : array, shape [n_vars] or [N, n_vars]
        Shared initial state (broadcast to all trajectories) or per-trajectory.
    t_span : array-like, shape [n_save]
        Strictly increasing array of save times (len >= 2).
    params : array, shape [N, ...]
        Per-trajectory parameters.
    solver : diffrax.AbstractSolver
        The Diffrax solver instance, e.g. ``diffrax.Tsit5()``.
    controller : diffrax.AbstractStepSizeController
        The step-size controller, e.g. a ``diffrax.PIDController``.
    first_step : float or None
        Initial step; ``None`` uses ``1e-6`` of the integration window.
    max_steps : int
        Diffrax's ``max_steps``.

    Returns
    -------
    array, shape [N, n_save, n_vars]
    """
    y0_arr = jnp.asarray(y0, dtype=jnp.float64)
    params_arr = jnp.asarray(params)
    if y0_arr.ndim == 1:
        y0_arr = jnp.broadcast_to(y0_arr, (params_arr.shape[0], y0_arr.shape[0]))
    save_times = jnp.asarray(t_span, dtype=jnp.float64)
    dt0 = jnp.float64(
        first_step
        if first_step is not None
        else (save_times[-1] - save_times[0]) * 1e-6
    )
    t0 = save_times[0]
    tf = save_times[-1]

    def _solve_one(y0_single, p):
        term = diffrax.ODETerm(lambda t, y, args: jnp.asarray(ode_fn(y, t, p)))
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=tf,
            dt0=dt0,
            y0=y0_single,
            stepsize_controller=controller,
            max_steps=max_steps,
            saveat=diffrax.SaveAt(ts=save_times),
        )
        return sol.ys

    return jax.vmap(_solve_one)(y0_arr, params_arr)
