"""Coupled Kaps singular-perturbation systems.

Kaps' problem — controllable stiffness with a known exact solution

Each equation pair (y1_i, y2_i) is an independent copy of the original Kaps system
with a different stiffness parameter epsilon_i. The stiffness ratio of pair i is
approximately 1 / epsilon_i, so epsilon_min controls the maximum stiffness across
the system. The initial conditions lie exactly on the slow manifold, so there is
no fast initial transient, only the stable fast modes that implicit solvers must
handle without being forced into tiny steps.

The analytical solution is independent of epsilon_i for all pairs:
    y1_i(t) = exp(-2t),  y2_i(t) = exp(-t)

This allows precise validation at arbitrary stiffness without a reference solver.
"""

import jax.numpy as jnp
import numpy as np

from reference.systems.python._tuple_codegen import (
    add,
    const,
    make_matrix_callback,
    make_tuple_callback,
    mul,
    neg,
    p,
    square,
    sub,
    y,
    zero_matrix,
)

TIMES = jnp.array((0.0, 0.5, 1.0, 2.0), dtype=jnp.float64)


def make_system(n_pairs, epsilon_min):
    """Construct n_pairs independent Kaps singular-perturbation equation pairs."""
    n_vars = 2 * n_pairs
    epsilon = jnp.array(
        [10.0 ** (np.log10(epsilon_min) * i / (n_pairs - 1)) for i in range(n_pairs)],
        dtype=jnp.float64,
    )
    y0 = jnp.array([1.0, 1.0] * n_pairs, dtype=jnp.float64)

    ode_values = []
    jac_rows = zero_matrix(n_vars, n_vars)
    for pair, eps in enumerate(np.asarray(epsilon)):
        base = 2 * pair
        inv_eps = 1.0 / float(eps)
        y_base = y(base)
        y_next = y(base + 1)
        y_next_sq = square(y_next)
        ode_values.extend(
            [
                mul(
                    p(0),
                    add(
                        mul(neg(const(inv_eps + 2.0)), y_base),
                        mul(const(inv_eps), y_next_sq),
                    ),
                ),
                mul(p(0), sub(sub(y_base, y_next), y_next_sq)),
            ]
        )
        jac_rows[base][base] = mul(p(0), neg(const(inv_eps + 2.0)))
        jac_rows[base][base + 1] = mul(p(0), const(2.0 * inv_eps), y_next)
        jac_rows[base + 1][base] = p(0)
        jac_rows[base + 1][base + 1] = mul(
            p(0),
            add(const(-1.0), mul(const(-2.0), y_next)),
        )

    ode_fn = make_tuple_callback("ode_fn", ode_values)
    jac_fn = make_matrix_callback("jac_fn", jac_rows)

    return {
        "n_pairs": n_pairs,
        "epsilon_min": epsilon_min,
        "n_vars": n_vars,
        "ode_fn": ode_fn,
        "jac_fn": jac_fn,
        "y0": y0,
    }


_DEFAULT = make_system(2, 1e-2)
ode_fn = _DEFAULT["ode_fn"]
jac_fn = _DEFAULT["jac_fn"]


def make_params(size, seed=42):
    rng = np.random.default_rng(seed)
    return jnp.array(
        1.0 + 0.1 * (2.0 * rng.random((size, 1)) - 1.0),
        dtype=jnp.float64,
    )


def exact_solution(t_span, params, n_pairs):
    """Exact solution for n_pairs Kaps equation pairs."""
    t = np.asarray(t_span, dtype=np.float64)
    s = np.asarray(params)[:, 0]
    y1 = np.exp(-2.0 * np.outer(s, t))
    y2 = np.exp(-1.0 * np.outer(s, t))
    pair = np.stack([y1, y2], axis=-1)
    return np.tile(pair, (1, 1, n_pairs))
