# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated ODE solvers for massive ensembles (1-100k) of low-dimensional
(<200D) trajectories, built on JAX and Numba-CUDA-MLIR. Each solver is a
hand-written CUDA kernel running one CUDA thread per trajectory, exposed to JAX as an
XLA FFI custom call. Applications: Bayesian parameter inference, uncertainty
quantification, and integrating physically uncoupled systems.

## Commands

```bash
# Install dependencies (uses uv)
uv sync                 # CPU
uv sync --extra cuda13  # or --extra cuda12, for GPU

# Run tests
uv run pytest

# Run a specific test file / test
uv run pytest tests/test_solvers.py
uv run pytest tests/test_examples.py -v

# Format and lint
uv run ruff format
uv run ruff check --fix
```

Most solver tests need a GPU and are skipped when `numba_cuda_mlir` is unavailable.
`tests/test_examples.py` runs anywhere: it compiles the examples' device
callbacks with `cuda.compile_ptx`, which exercises the full numba typing and
lowering pipeline without a device.

## Architecture

### Solvers (`solvers/`)

| Method   | Type                   | Use for           | File          |
|----------|------------------------|-------------------|---------------|
| Tsit5    | Explicit RK (order 5)  | Non-stiff systems | `tsit5.py`    |
| Rodas5P  | Rosenbrock-W (order 5) | Stiff systems     | `rodas5P.py`  |

Shared support modules:

- **`_numba_common.py`** — host-side helpers: input normalisation, initial step
  selection, error weights, and the `cuda.jit` coercion for user callbacks.
- **`_enzyme_jacobian.py`** — differentiates `ode_fn` with numba-enzyme so the
  implicit solvers need no hand-written Jacobian. See "Derived Jacobians".
- **`_jax_numba_custom_call.py`** — the XLA FFI shim. Compiles a raw-pointer
  launcher, registers it as an FFI target, and exposes `ffi_call`/`ffi_abi_call`
  so a numba kernel becomes a JAX primitive.
- **`_jax_common.py`** — the JAX-facing glue: ensemble shape normalisation and
  `make_custom_vmap_solver`, whose `custom_vmap` rule lowers an outer
  `jax.vmap` over a single solve into one native ensemble launch.

Because the solvers go through `jax.ffi.ffi_call`, they are `jit`-traceable and
usable inside `lax.scan`/`vmap` — see `examples/bbn_estimation`, which calls one
from a BlackJAX nested-sampling likelihood.

### Solver API

```python
# Explicit
y = solve(ode_fn, y0, t_span, params, rtol=..., atol=..., first_step=...)

# Implicit: same shape; df/dy and df/dt are derived from ode_fn
y = solve(ode_fn, y0, t_span, params, lu_precision="fp32")
```

`y0` is `(n_vars,)` or `(N, n_vars)`; `params` is `(n_params,)` or
`(N, n_params)`; the result is `(N, n_save, n_vars)`. `return_stats=True` adds a
dict of per-trajectory step counters.

### Writing ODE callbacks

Callbacks are compiled with `numba_cuda_mlir`, which constrains them:

- Take and return **fixed-size tuples of scalars**, not arrays.
- Use `math`, not `numpy`/`jax.numpy`.
- **Device code cannot call a plain Python helper.** Either inline the shared
  work or pre-decorate the helper with `@cuda.jit(device=True)`.
- Closed-over arrays land in CUDA **constant memory** (64 KiB per module), and
  numba emits one copy *per reference site*. Pack related tables into a single
  array and bind it to a local before indexing — see `make_mode_ode_device` in
  `examples/mukhanov_sasaki/main.py`.

### Derived Jacobians

`rodas5P` takes no `jac_fn`. `_enzyme_jacobian.py` compiles `ode_fn` once into a
vector-valued primal that writes its outputs through a leading array argument —
MLIR cannot lower a tuple return across an `abi="c"` boundary — and
forward-differentiates it with [numba-enzyme][ne]. Seeding a unit vector gives a
whole Jacobian column per sweep; seeding the time argument gives the whole
`df/dt`, so `n_vars + 1` sweeps supply both matrices the kernel needs.

Forward mode is what makes a sweep worth a whole column: a sweep of a
scalar-output primal yields one Jacobian *entry*. Reverse mode reaches a whole
row per sweep by slicing the right-hand side into scalar components, which
dead-code elimination then shrinks — that was measured to win the solve below
roughly 16 state variables, but it needs `n_vars` primals and `n_vars` Enzyme
differentiations against forward's one apiece, with `O(n_vars ** 2)` generated
device source against `O(n_vars)`. A cold first solve at 96 state variables took
171 s that way against 49 s this way.

Three things to know when touching this:

- Enzyme's CUDA backend takes **flat scalar arguments**, so the state and
  parameter tuples are rebuilt inside the generated primal's body. The output
  array's shadow inherits the primal's offset and stride, so the two arrays
  passed to the derivative must match in layout.
- The column index is a run-time argument and the unit seed is built inside the
  derivative, so this stays one call site and one Enzyme build at any `n_vars`,
  and nothing here materialises a tangent vector.
- Every generated primal has identical source and differs only in what it
  closes over. That is safe because numba-enzyme keys its derivative cache on
  the primal's lowered IR and gives the primal internal linkage; both are local
  changes to that package, so check `wheels/README.md` before upgrading it.
- The derivative is linked as NVVM LTO IR, not PTX, so nvJitLink inlines it
  into the kernel. That is what keeps the per-column buffers in registers —
  linked as PTX they cost `2 * n_vars` doubles of local memory per thread, and
  the solve is 10-20% slower.

The wheel this depends on is not on PyPI — see `wheels/README.md`, which lists
every local change made to numba-enzyme.

[ne]: https://github.com/Qruise-ai/numba-enzyme

`lu_precision` (`"fp32"`/`"fp64"`) selects the LU precision for implicit
solvers. The `"fp32"` default does not lower the method's order — the
Rosenbrock order conditions hold under an approximate Jacobian — while
halving the LU shared-memory footprint.

### Kernel design

- One CUDA thread per trajectory; per-trajectory adaptive stepping, so lanes in
  a warp diverge and the block runs until its slowest trajectory finishes.
- In-kernel LU factorisation with shared-memory workspaces.
- `tsit5` has `backend="shared"` and `backend="global"` paths, selected by
  whether the state fits in shared memory; the two are bit-identical.

### Reference and benchmarks

- `reference/systems/python/` — test systems (Lorenz, VdP, Robertson,
  Brusselator, Bateman, Kaps, heat), each exposing `ode_fn`/`jac_fn` in
  numba-compatible tuple form plus `make_scenario` for ensembles. The solvers
  no longer consume `jac_fn`; it is what `tests/test_enzyme_jacobian.py` checks
  the Enzyme-derived Jacobian against.
  `_tuple_codegen.py` generates these callbacks for parameterised dimensions.
- `reference/solvers/python/` — Diffrax, scipy and Julia (DiffEqGPU) baselines.
- `scripts/` — scaling, dimensionality and divergence benchmarks. Each caches
  timings in `results.json` and writes a per-GPU CSV and plot.
- `examples/` — worked problems, each with its own README.
