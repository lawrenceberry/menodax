"""Kvaerno5 solver via diffrax — ESDIRK method of order 5 for stiff ODEs."""

import functools

import diffrax
import jax

from reference.solvers.python._diffrax_common import solve_ensemble


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
    return solve_ensemble(
        ode_fn,
        y0,
        t_span,
        params,
        diffrax.Kvaerno5(),
        diffrax.PIDController(rtol=rtol, atol=atol),
        first_step=first_step,
        max_steps=max_steps,
    )
