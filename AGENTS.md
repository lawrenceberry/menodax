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

- **`_numba_common.py`** — host-side helpers shared by both kernels: input
  normalisation, initial step selection, error weights, the device workspace,
  the shared kernel signature and its two launch paths (`run_kernel` for a
  direct numba launch, `ensemble_ffi_call` for the JAX one), and the `cuda.jit`
  coercion for user callbacks.
- **`_jax_numba_custom_call.py`** — the XLA FFI shim. Compiles a launcher,
  registers it as an FFI target, and exposes `ffi_abi_call` so a numba kernel
  becomes a JAX primitive.
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

`first_step` reaches the kernel as a launch-time scalar, so it cannot be
derived on the host from `t_span` — that argument is traced. Omitting it (or
passing a non-positive value) hands the kernel a sentinel, and it starts from
1e-6 of its own integration window.

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

`rodas5P` takes no `jac_fn`. `_make_kernel` forward-differentiates `ode_fn`
with [numba-enzyme][ne], as it stands and with no adapter around it: a
callback of the documented shape already reaches Enzyme as a function of flat
scalars returning a struct, because numba-cuda-mlir flattens a tuple argument
into one scalar parameter per element and lowers a tuple return to a struct
returned by value. Seeding a unit vector gives a whole Jacobian column per
sweep; seeding the time argument gives the whole `df/dt`, so `n_vars + 1`
sweeps supply both matrices the kernel needs.
numba-enzyme also exposes `jacfwd`, which fills the whole matrix; the kernel
uses `jacfwd_column` because that matrix would put `n_vars ** 2` doubles in
per-thread local memory, where one column at a time keeps the working set at
`O(n_vars)` and lets each column fold straight into the shared LU buffer.

Forward mode is what makes a sweep worth a whole column: a sweep of a
scalar-output primal yields one Jacobian *entry*. Reverse mode reaches a whole
row per sweep by slicing the right-hand side into scalar components, which
dead-code elimination then shrinks — that was measured to win the solve below
roughly 16 state variables, but it needs `n_vars` primals and `n_vars` Enzyme
differentiations against forward's one apiece, with `O(n_vars ** 2)` generated
device source against `O(n_vars)`. A cold first solve at 96 state variables took
171 s that way against 49 s this way.

Things to know when touching this:

- The derivative's call shape **mirrors the primal's argument list**, each
  tuple argument supplied as a contiguous array. So the kernel passes the same
  `y[i]` and `p[i]` rows it already passes to `ode_fn`, and the call is five
  arguments at any `n_vars`; numba-enzyme's entry point loads the scalars out
  of those rows before handing them to Enzyme. There is no generated Python
  here at all, and no module either: the derivative is a signature and one
  `jacfwd_column` call inside `_make_kernel`.
- The column index runs over the primal's **flattened** arguments, which is why
  `df/dt` is free: `t` is the argument after the state. It is a run-time
  argument and the unit seed is built inside the derivative, so this is one
  Enzyme build at any `n_vars`, and nothing materialises a tangent vector.
- numba-enzyme keys its derivative cache on the primal's lowered IR and gives
  the primal internal linkage, so two ODEs that differ only in what they close
  over cannot share a derivative. Both are local changes to that package —
  check `wheels/README.md` before upgrading it.
- The derivative is linked as NVVM LTO IR, not PTX, so nvJitLink inlines it
  into the kernel. That is what keeps the per-column buffers in registers —
  linked as PTX they cost `2 * n_vars` doubles of local memory per thread, and
  the solve is 10-20% slower.
- An explicit signature is **required**, and not only because an array cannot
  say how long the tuple it stands for is. The callbacks are duck-typed on
  indexing, so the kernel's own calls specialise `ode_fn` for array arguments
  (see `make_cuda_striped_vector_writer`); only the signature says the
  derivative wants the tuple form.
- The five-argument call inlines, so the kernel's PTX is byte-identical across
  processes and the CUDA JIT cache hits. Spelling the state out as one scalar
  per component instead makes the call a star call at `n_vars=48`, which
  numba-cuda-mlir will not inline, leaving a wrapper in the kernel named after
  an `id()` that changes every process — that cost 13.0 s a warm compile
  against 10.1 s here.

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
