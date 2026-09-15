"""Kvaerno5 solver via diffrax — ESDIRK method of order 5 for stiff ODEs."""

import functools

import diffrax
import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=("ode_fn", "max_steps"))
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
):
    """Solve a Kvaerno5 ensemble with Diffrax.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``.
    y0 : array, shape [n_vars] or [N, n_vars]
        Shared initial state (broadcast to all trajectories) or per-trajectory.
    t_span : array-like, shape [n_save]
        Strictly increasing array of save times (len >= 2).
    params : array, shape [N, ...]
        Per-trajectory parameters.

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
        # ode_fn may return a tuple (unified RHS format); diffrax's ODETerm
        # requires the vector field to match y0's array pytree structure.
        term = diffrax.ODETerm(lambda t, y, args: jnp.asarray(ode_fn(y, t, p)))
        solver = diffrax.Kvaerno5()
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
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
