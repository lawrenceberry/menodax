"""Stiff diffusively coupled van der Pol oscillator ring lattice.

The ring couples each oscillator to its two nearest neighbours. ``make_system``
also takes a ``coupling_range`` -- every oscillator within that many places
along the ring -- which changes the Jacobian's density without changing the
dimension; ``make_sparsity`` and ``jacobian_density`` describe the pattern that
results, for a solver that can exploit it.
"""

import jax.numpy as jnp
import numpy as np

from reference.systems.python._tuple_codegen import make_tuple_callback

N_OSC = 35
N_VARS = 2 * N_OSC
N_PARAMS = 1
MU = 100.0
D = 10.0
OMEGA = 1.0
TIMES = jnp.array((0.0, 0.25, 0.5, 0.75, 1.0), dtype=jnp.float64)
Y0 = jnp.array([2.0, 0.0] * N_OSC, dtype=jnp.float64)
PARAMS = jnp.array([1.0], dtype=jnp.float64)


def neighbours(n_osc: int, osc: int, coupling_range: int = 1) -> list[int]:
    """The oscillators within ``coupling_range`` places of ``osc`` along the ring.

    One entry per place, so an oscillator the ring reaches from both sides
    appears twice -- as the nearest-neighbour ring already counts the single
    neighbour of ``n_osc == 2`` -- and ``osc`` itself is left out. The range
    ``n_osc // 2`` couples every oscillator to every other.
    """
    if coupling_range < 1:
        raise ValueError("coupling_range must be at least 1")
    found = []
    for offset in range(1, coupling_range + 1):
        for other in ((osc - offset) % n_osc, (osc + offset) % n_osc):
            if other != osc:
                found.append(other)
    return found


def make_system(
    n_osc: int,
    *,
    mu: float = MU,
    d: float = D,
    omega: float = OMEGA,
    coupling_range: int = 1,
):
    """Return (ode_fn, y0) for a ring of n_osc coupled van der Pol oscillators.

    Defaults reproduce the stiff baseline (mu=100, d=10, omega=1). Pass
    ``mu=1.0`` for the non-stiff variant used by explicit-method benchmarks.

    ``coupling_range`` couples each oscillator diffusively to every oscillator
    within that many places along the ring, with the coupling ``d`` divided by
    the range so that the total coupling strength stays comparable; ``1`` is
    the nearest-neighbour ring the other helpers assume.
    """
    y0 = jnp.array([2.0, 0.0] * n_osc, dtype=jnp.float64)
    strength = d / coupling_range
    components = []
    for osc in range(n_osc):
        base = 2 * osc
        x, v = f"y[{base}]", f"y[{base + 1}]"
        others = neighbours(n_osc, osc, coupling_range)
        laplacian = " + ".join(
            [f"y[{2 * other}]" for other in others] + [f"-{float(len(others))!r} * {x}"]
        )
        components += [
            v,
            f"p[0] * {mu!r} * (1.0 - {x} * {x}) * {v}"
            f" + -{omega * omega!r} * {x}"
            f" + {strength!r} * ({laplacian})",
        ]

    ode_fn = make_tuple_callback("ode_fn", components)

    return ode_fn, y0


def make_sparsity(n_osc: int, coupling_range: int = 1) -> np.ndarray:
    """The ``(n_vars, n_vars)`` Jacobian mask of ``make_system``'s right-hand side.

    Each position row depends on its own velocity; each velocity row on its own
    position and velocity and on the positions of the oscillators it couples to.
    """
    n_vars = 2 * n_osc
    mask = np.zeros((n_vars, n_vars), dtype=bool)
    for osc in range(n_osc):
        x, v = 2 * osc, 2 * osc + 1
        mask[x, v] = True
        mask[v, x] = mask[v, v] = True
        for other in neighbours(n_osc, osc, coupling_range):
            mask[v, 2 * other] = True
    return mask


def jacobian_density(n_osc: int, coupling_range: int = 1) -> float:
    """The fraction of Jacobian entries that are structurally nonzero."""
    mask = make_sparsity(n_osc, coupling_range)
    return float(mask.sum()) / mask.size


ode_fn, _ = make_system(N_OSC)


def make_params(size: int, seed: int = 42) -> np.ndarray:
    """Return damping-scale parameters with ±20% uniform perturbation."""
    rng = np.random.default_rng(seed)
    return np.array(1.0 + 0.2 * (2.0 * rng.random((size, 1)) - 1.0), dtype=np.float64)


def make_scenario(
    n_osc: int,
    size: int,
    seed: int = 42,
    *,
    divergence: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial conditions and parameters for the coupled VDP lattice.

    ``divergence`` controls how far each trajectory is moved away from the
    synchronized baseline. ``0.0`` gives the identical state, ``1.0`` gives
    the original divergent initial-condition distribution, and larger values
    increase velocity spread and damping-scale variation.
    """
    n_vars = 2 * n_osc
    if not np.isfinite(divergence) or divergence < 0.0:
        raise ValueError("divergence must be finite and non-negative")

    rng = np.random.default_rng(seed)
    amplitudes = rng.uniform(0.25, 3.0, size=(size, n_osc))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(size, n_osc))
    target_x = amplitudes * signs
    position_divergence = min(divergence, 1.0)
    x = 2.0 + position_divergence * (target_x - 2.0)
    v = rng.normal(0.0, 2.0 * divergence, size=(size, n_osc))
    y0 = np.empty((size, n_vars), dtype=np.float64)
    y0[:, 0::2] = x
    y0[:, 1::2] = v

    base_params = make_params(size, seed)
    param_center = max(divergence, 1.0)
    params = np.maximum(param_center + divergence * (base_params - 1.0), 1e-6).astype(
        np.float64
    )
    return y0, params


def make_initial_conditions(kind: str, size: int, seed: int = 42) -> np.ndarray:
    """Return baseline or broadly varied initial states.

    State ordering is ``(x0, v0, x1, v1, ..., x34, v34)``.
    """
    if kind == "identical":
        return np.broadcast_to(np.asarray(Y0, dtype=np.float64), (size, N_VARS)).copy()
    if kind != "ic_large":
        raise ValueError(f"unknown initial-condition kind: {kind}")

    rng = np.random.default_rng(seed)
    amplitudes = rng.uniform(0.25, 3.0, size=(size, N_OSC))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(size, N_OSC))
    x = amplitudes * signs
    v = rng.normal(0.0, 2.0, size=(size, N_OSC))
    y0 = np.empty((size, N_VARS), dtype=np.float64)
    y0[:, 0::2] = x
    y0[:, 1::2] = v
    return y0
