# Writing ODE callbacks

The right-hand side is compiled with
[`numba_cuda_mlir`](https://github.com/NVIDIA/numba-cuda) and runs as device
code inside the solver kernel, which constrains how it is written.

```python
def ode_fn(y, t, p):
    """(y, t, p) -> tuple of the same length as y."""
    y0, y1 = y
    mu, = p
    return y1, mu * (1.0 - y0 * y0) * y1 - y0
```

A plain Python function is jitted automatically; one already decorated with
`cuda.jit(device=True)` is used as it is.

## The rules

- **Fixed-size tuples of scalars**, in and out — not arrays. The tuple's length
  is the system's dimension and has to be known when the kernel is built.
- **`math`, not `numpy` or `jax.numpy`.** `math.exp`, `math.log`, `math.sqrt`
  and the rest lower to device intrinsics.
- **Device code cannot call a plain Python helper.** Either inline the shared
  work or pre-decorate the helper with `@cuda.jit(device=True)`.
- **Index the tuple with constants only.** A runtime index into a tuple makes
  numba emit a bounds-check `cmpxchg`, and Enzyme refuses the atomic ("cannot
  handle unknown instruction") when it differentiates the callback for
  Rodas5P. Wrapping the loop around a thread-local array instead gets past
  typing but yields IR NVVM will not verify. This is why the reference systems
  under `reference/systems/python/` are generated from one expression string
  per component, and why DISCO-EB's unrolled right-hand side is generated from
  its looped one.
- **Watch what you close over.** Closed-over arrays land in CUDA *constant
  memory*, 64 KiB per module, and numba emits one copy **per reference site**.
  Pack related tables into a single array and bind it to a local before
  indexing it — `make_mode_ode` in `examples/mukhanov_sasaki/main.py` is the
  worked case.

## One body, two backends

Every example here has a Diffrax baseline to compare against, so its right-hand
side has to run both as device code and under `jax`. It is written **once**:
`examples/dual_backend.py` takes the body as a factory over the handful of
names the two backends spell differently (`math.exp`/`jnp.exp`, `max`/
`jnp.maximum`, …), and `build_rhs` returns it three ways:

| Member | What it is | Used for |
|--------|------------|----------|
| `.device` | the tuple form | the modax solve |
| `.jax` | the array form | the Diffrax baseline |
| `.host` | the device arithmetic in plain Python | comparing the two without a GPU, as `tests/test_examples.py` does |

What such a body must avoid is an `if` on a value that varies, since a traced
value cannot be branched on at all. `maximum`/`minimum`, or a branchless
`a + (b - a) * (x > c)`, serve both backends. A helper the body shares with the
rest of the module goes through `build_fn`, whose `.device` member is a
`cuda.jit(device=True)` function — Enzyme inlines straight through it.

## What Rodas5P needs on top

Nothing. There is no `jac_fn`: `df/dy` and the `df/dt` a non-autonomous system
needs to keep fifth-order accuracy are both differentiated out of `ode_fn` with
[numba-enzyme](https://github.com/Qruise-ai/numba-enzyme), which runs Enzyme
over the callback's LLVM IR. The tuple form is exactly what Enzyme reads — a
callback of the documented shape reaches it as a function of flat scalars
returning a struct, because numba-cuda-mlir flattens a tuple argument into one
scalar parameter per element and lowers a tuple return to a struct returned by
value.
