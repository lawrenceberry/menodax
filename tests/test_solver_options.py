"""Rodas5P options that are neither the tableau nor the sparsity: block width,
the ``df/dt`` stage term, and per-trajectory end times.

The linear solve is not one of them any more. It is the default dense LU, or a
sparse direct solver compiled from ``sparsity`` -- there is no third choice and
nothing to pass. ``test_sparse_direct.py`` covers the sparse path.
"""

import numpy as np
import pytest
from numba_cuda_mlir import cuda

from modax.rodas5P import (
    _DEFAULT_TRAJECTORIES_PER_BLOCK,
    trajectories_per_block_or_default,
)
from modax.rodas5P import solve as rodas5P_solve

requires_cuda = pytest.mark.skipif(not cuda.is_available(), reason="CUDA required")


# Non-autonomous: dy/dt = -lam*y + forcing*t, y(0) = 0. Without the
# dt*d_i*df/dt stage correction a Rosenbrock-W method drops below order 5 here,
# so this is what checks df/dt reaches the stages.
LAMBDA = 10.0
FORCING = 5.0


def forced_decay(y, t, p):
    return (-p[0] * y[0] + p[1] * t,)


def _exact_forced(t):
    return FORCING * (LAMBDA * t - 1.0 + np.exp(-LAMBDA * t)) / LAMBDA**2


def test_trajectories_per_block_or_default():
    """A warp by default; nothing on chip bounds an explicit request."""
    assert trajectories_per_block_or_default() == _DEFAULT_TRAJECTORIES_PER_BLOCK
    assert trajectories_per_block_or_default(8) == 8
    assert trajectories_per_block_or_default(64) == 64
    with pytest.raises(ValueError, match="must be positive"):
        trajectories_per_block_or_default(0)


@requires_cuda
def test_the_time_derivative_term_reaches_the_stages():
    t_span = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    y0 = np.zeros((1, 1), dtype=np.float64)
    params = np.array([[LAMBDA, FORCING]], dtype=np.float64)
    result = np.asarray(
        rodas5P_solve(
            forced_decay,
            y0,
            t_span,
            params,
            rtol=1e-10,
            atol=1e-12,
            first_step=1e-4,
            lu_precision="fp64",
        )
    )[0, :, 0]
    assert np.allclose(result, _exact_forced(t_span), rtol=1e-8, atol=1e-10)


@requires_cuda
def test_tf_index_ends_each_trajectory_at_its_own_time():
    """Save times past a trajectory's own end time hold its final state."""
    t_span = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    # The third parameter column is the per-trajectory end time.
    params = np.array(
        [[LAMBDA, FORCING, 1.0], [LAMBDA, FORCING, 0.5]], dtype=np.float64
    )
    y0 = np.zeros((2, 1), dtype=np.float64)
    result = np.asarray(
        rodas5P_solve(
            forced_decay,
            y0,
            t_span,
            params,
            tf_index=2,
            rtol=1e-10,
            atol=1e-12,
            first_step=1e-4,
            lu_precision="fp64",
        )
    )
    exact = _exact_forced(t_span)
    assert np.allclose(result[0, :, 0], exact, rtol=1e-8, atol=1e-10)
    stops_at = int(np.searchsorted(t_span, 0.5, side="right"))
    assert np.allclose(result[1, :stops_at, 0], exact[:stops_at], rtol=1e-8, atol=1e-10)
    held = result[1, stops_at:, 0]
    assert np.allclose(held, result[1, stops_at - 1, 0], rtol=0, atol=0)
