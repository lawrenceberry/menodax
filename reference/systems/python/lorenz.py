"""Lorenz chaotic system (3D) for solver benchmarks and tests.

Summary:
    dx/dt = σ(y − x)
    dy/dt = x(ρ − z) − y
    dz/dt = xy − βz

with σ = 10, β = 8/3 fixed and ρ as the ensemble parameter (centred at 28,
the standard chaotic regime; chaos onset at ρ ≈ 24.74).

The Lorenz attractor is a strange attractor: trajectories orbit it forever but
never repeat.  The maximum Lyapunov exponent λ ≈ 0.9 means that two initially
close trajectories diverge on a timescale of ~1/λ ≈ 1 time unit, so there is
no meaningful point-wise reference solution after t ≈ 5.

Instead the tests verify attractor confinement: a solver that accumulates too
much integration error leaves the attractor manifold and diverges to infinity,
while a correct solver keeps all trajectories within the known attractor
extent (|x|, |y| < 40, 0 ≤ z < 65 for ρ ≈ 28) even after long integration.
These bounds are the signature of a solver that stays on the manifold.

"""

import jax.numpy as jnp
import numpy as np

TIMES = (0.0, 5.0)

Y0 = jnp.array([1000.0, -500.0, 500.0], dtype=np.float64)
N_VARS = 3

PARAMS = jnp.array([0.5], dtype=np.float64)
N_PARAMS = 1


def ode_fn(y, t, p):
    sigma = 10.0
    beta = 8.0 / 3.0
    return (
        sigma * (y[1] - y[0]),
        y[0] * (p[0] - y[2]) - y[1],
        y[0] * y[1] - beta * y[2],
    )


def make_params(size: int, seed: int = 42) -> jnp.ndarray:
    """ρ values centred at 0.5 with ±5% uniform perturbation (all chaotic)."""
    rng = np.random.default_rng(seed)
    return jnp.array(
        0.5 * (1.0 + 0.05 * (2.0 * rng.random((size, 1)) - 1.0)),
        dtype=jnp.float64,
    )


def _validate_divergence(divergence: float) -> float:
    if not np.isfinite(divergence) or divergence < 0.0:
        raise ValueError("divergence must be finite and non-negative")
    return float(divergence)


def make_initial_conditions(
    size: int, seed: int = 42, *, divergence: float = 1.0
) -> jnp.ndarray:
    """The Lorenz system is run with rho=0.5, which places it below the pitchfork
    bifurcation at rho=1. In this regime the origin is the unique globally stable
    fixed point, so every trajectory decays exponentially toward (0, 0, 0) with
    slowest eigenvalue lambda ~ -0.47.

    Step count is therefore strictly monotone in distance from the origin: a
    trajectory that starts far away must cross a large region of phase space with
    significant ODE velocities before the state becomes small enough for Tsit5 to
    take large steps.

    The hard base IC, Y0 = [1000, -500, 500], is fixed analytically as a point
    far from the origin. ``divergence=0`` places every trajectory there, so all
    share the same (maximum) step count and the batch shows zero divergence.
    Larger divergence values scale each trajectory as ``Y0 * (1 - t)`` for
    ``t ~ U(0, divergence)``, moving it radially toward and then, for
    divergence > 1, past the origin. Because distance from the origin is the
    main driver of difficulty, values up to 1 span from near-maximum
    (t -> 0) down to near-zero (t -> 1), while larger divergence values add
    trajectories that start far away on the opposite radial branch.
    """
    divergence = _validate_divergence(divergence)
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, divergence, size=(size, 1))
    return Y0 * (1.0 - t)


def make_scenario(
    size: int, seed: int = 42, *, divergence: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    divergence = _validate_divergence(divergence)
    params = np.asarray(PARAMS) + divergence * (
        np.asarray(make_params(size, seed)) - np.asarray(PARAMS)
    )
    return (
        np.asarray(make_initial_conditions(size, seed, divergence=divergence)),
        params,
    )
