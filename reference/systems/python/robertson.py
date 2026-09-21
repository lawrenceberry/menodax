"""Robertson stiff chemical kinetics system for solver benchmarks and tests."""

import jax.numpy as jnp
import numpy as np

TIMES = jnp.array((0.0, 1e-6, 1e-2, 1e2, 1e5), dtype=jnp.float64)

ALPHA = 0.99
EPS = 0.1
Y0 = jnp.array([(1 - EPS) * ALPHA, EPS, (1 - EPS) * (1 - ALPHA)], dtype=np.float64)
N_VARS = 3

PARAMS = jnp.array([0.04, 1e4, 3e7], dtype=jnp.float64)
N_PARAMS = 3


def ode_fn(y, t, p):
    return (
        -p[0] * y[0] + p[1] * y[1] * y[2],
        p[0] * y[0] - p[1] * y[1] * y[2] - p[2] * y[1] ** 2,
        p[2] * y[1] ** 2,
    )


def make_params(size: int, seed: int = 42) -> jnp.ndarray:
    """Return Robertson rate constants with +/-10% uniform perturbation."""
    rng = np.random.default_rng(seed)
    return jnp.array(
        PARAMS * (1.0 + 0.1 * (2.0 * rng.random((size, N_PARAMS)) - 1.0)),
        dtype=jnp.float64,
    )


def _validate_divergence(divergence: float) -> float:
    if not np.isfinite(divergence) or divergence < 0.0:
        raise ValueError("divergence must be finite and non-negative")
    return float(divergence)


def make_initial_conditions(
    size: int, seed: int = 42, *, divergence: float = 1.0
) -> jnp.ndarray:
    """ICs are parameterised by (alpha, epsilon):

        y(0) = [(1-eps)*alpha,  eps,  (1-eps)*(1-alpha)]

    where eps controls how much of the intermediate species y2 is present at
    t=0 and alpha distributes the remaining mass between fuel (y1) and product
    (y3).

      * eps > 0 forces the solver to resolve the fastest reaction timescale
        immediately: dy3/dt = k3*eps^2 is large at t=0, so the first step must
        be tiny and the Newton iteration may reject steps before settling onto
        the slow manifold.  The standard [1,0,0] IC has eps=0 and a long
        induction period where only the slow k1=0.04 rate is active.

      * alpha controls how long the stiff phase lasts: high alpha means plenty
        of y1 fuel to keep the reactions running, extending the hard region of
        the trajectory.

    ``divergence=0`` returns the identical hard baseline. Larger values draw
    alpha from a broader range and move epsilon log-uniformly from EPS toward
    much smaller intermediate-species concentrations. This keeps concentrations
    non-negative and mass-conserving while spreading Robertson step counts over
    easy induction-dominated trajectories and hard immediately-stiff ones.
    """
    divergence = _validate_divergence(divergence)
    base = np.asarray(Y0, dtype=np.float64)
    if divergence == 0.0:
        return np.broadcast_to(base, (size, N_VARS)).copy()

    rng = np.random.default_rng(seed)
    alpha = ALPHA * rng.uniform(0.0, 1.0, size)
    eps_blend = rng.uniform(0.0, min(divergence / 4.0, 1.0), size)
    eps = 10.0 ** ((1.0 - eps_blend) * np.log10(EPS) + eps_blend * np.log10(1e-8))
    y0 = np.column_stack(
        [
            (1 - eps) * alpha,
            eps,
            (1 - eps) * (1 - alpha),
        ]
    )
    return y0


def make_scenario(
    size: int, seed: int = 42, *, divergence: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return Robertson initial conditions and parameters.

    The parameter spread is multiplicative so the divergence knob can extend
    beyond the original +/-10% perturbation without crossing through negative
    reaction rates. Values up to about 3 are intended for divergence benchmarks;
    much larger values may intentionally create extreme stiffness.
    """
    divergence = _validate_divergence(divergence)
    if divergence == 0.0:
        params = np.broadcast_to(
            np.asarray(PARAMS, dtype=np.float64), (size, N_PARAMS)
        ).copy()
    else:
        rng = np.random.default_rng(seed)
        scales = 10.0 ** rng.uniform(
            -divergence,
            divergence,
            size=(size, N_PARAMS),
        )
        params = np.asarray(PARAMS, dtype=np.float64) * scales
    return make_initial_conditions(size, seed, divergence=divergence), params
