# Getting started

--8<-- "README.md:install"

A GPU is needed to *run* a solve: every solver is a CUDA kernel. The CPU
install is enough for the tests that exercise the typing and lowering pipeline
without a device (`tests/test_examples.py`), and for building these docs.

## The first solve

```python
import jax.numpy as jnp
from menodax.rodas5P import solve


# A CUDA-device callback: fixed-size tuples of scalars in and out, `math`
# rather than `numpy`. This is the Robertson problem.
def ode_fn(y, t, p):
    y0, y1, y2 = y
    k1, k2, k3 = p
    f0 = -k1 * y0 + k3 * y1 * y2
    f1 = k1 * y0 - k2 * y1 * y1 - k3 * y1 * y2
    f2 = k2 * y1 * y1
    return f0, f1, f2


y0 = jnp.array([1.0, 0.0, 0.0])                  # one initial state ...
params = jnp.tile(                               # ... and 10k parameter rows
    jnp.array([0.04, 3.0e7, 1.0e4]), (10_000, 1)
)
t_span = jnp.logspace(-6, 5, 64)

y = solve(ode_fn, y0, t_span, params)            # (10000, 64, 3)
```

The ensemble dimension comes from whichever of `y0` and `params` is
two-dimensional; both may be, and they must then agree. The result is always
`(N, n_save, n_vars)`.

## Where to go next

- [Calling a solver](guide/api.md) — the full argument list, and the rules the
  callbacks obey.
- [Writing ODE callbacks](guide/callbacks.md) — what `numba_cuda_mlir` allows,
  and how a right-hand side that must also run under JAX is written once.
- [Sparse systems](guide/sparse.md) — what a `sparsity` pattern is worth on a
  structured problem.
- [Gradients](guide/gradients.md) — differentiating a solve with respect to
  `y0` and `params`.
