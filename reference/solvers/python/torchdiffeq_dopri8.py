"""Dopri8 solver via torchdiffeq -- explicit Runge--Kutta of order 8 for non-stiff ODEs.

torchdiffeq integrates one tensor-valued state with one adaptive step, so an
ensemble is solved as a single ``(n_vars, N)`` system whose error norm mixes
every trajectory: the step is the one the worst trajectory wants, which is what
the divergence benchmarks measure. The tuple-form ``ode_fn`` is applied to the
transposed batch directly, since indexing ``y[i]`` on an ``(n_vars, N)`` tensor
picks component ``i`` of every trajectory and the arithmetic broadcasts; that
requires a callback built from arithmetic alone, which the non-stiff reference
systems are.
"""

import numpy as np
import torch
from torchdiffeq import odeint


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
    device="cuda",
):
    """Solve an ensemble with torchdiffeq's ``dopri8`` and return a NumPy array.

    Parameters
    ----------
    ode_fn : callable
        ODE right-hand side with signature ``ode_fn(y, t, params) -> dy/dt``,
        returning a tuple of components, each an expression in ``y[i]``,
        ``p[j]`` and ``t`` that broadcasts over a batch.
    y0 : array, shape [n_vars] or [N, n_vars]
        Shared initial state (broadcast to all trajectories) or per-trajectory.
    t_span : array-like, shape [n_save]
        Strictly increasing array of save times (len >= 2).
    params : array, shape [N, n_params]
        Per-trajectory parameters.
    first_step : float or None
        Initial step; ``None`` uses ``1e-6`` of the integration window.
    max_steps : int
        torchdiffeq's ``max_num_steps``.

    Returns
    -------
    ndarray, shape [N, n_save, n_vars]

    The call blocks on the device before returning, so timing it end to end
    measures the solve.
    """
    params_arr = np.asarray(params, dtype=np.float64)
    y0_arr = np.asarray(y0, dtype=np.float64)
    if y0_arr.ndim == 1:
        y0_arr = np.broadcast_to(y0_arr, (params_arr.shape[0], y0_arr.shape[0]))
    times = torch.as_tensor(np.asarray(t_span, dtype=np.float64), device=device)
    # (n_vars, N): y[i] is component i of every trajectory, p[j] parameter j.
    state0 = torch.as_tensor(np.ascontiguousarray(y0_arr.T), device=device)
    p_rows = torch.as_tensor(np.ascontiguousarray(params_arr.T), device=device)
    dt0 = (
        float(first_step)
        if first_step is not None
        else float(times[-1] - times[0]) * 1e-6
    )

    def rhs(t, y):
        return torch.stack(ode_fn(y, t, p_rows))

    with torch.no_grad():
        ys = odeint(
            rhs,
            state0,
            times,
            rtol=rtol,
            atol=atol,
            method="dopri8",
            options={"first_step": dt0, "max_num_steps": max_steps},
        )
        if device == "cuda":
            torch.cuda.synchronize()
    # (n_save, n_vars, N) -> (N, n_save, n_vars)
    return ys.permute(2, 0, 1).cpu().numpy()
