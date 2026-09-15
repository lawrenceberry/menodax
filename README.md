# modax

GPU-accelerated ODE solvers for **massive ensembles** (1-100k) of low-dimensional (<200D) ODE trajectories, built on
JAX and Numba-CUDA-MLIR. Applications include: Bayesian parameter inference, uncertainty quantification and the integration of physically uncoupled systems.

Every solver is a hand-written **CUDA custom kernel** compiled by
Numba-CUDA-MLIR: one CUDA thread
per trajectory, hand-written step kernels with in-kernel LU factorisation,
exposed to JAX as an XLA FFI custom call. That binding makes each solver an
ordinary JAX primitive — `jit`-traceable, and `vmap` over a single solve lowers
to one native ensemble launch.

## Solvers (`solvers/`)

| Method      | Type                   | Use for           | File           |
|-------------|------------------------|-------------------|----------------|
| **Tsit5**   | Explicit RK (order 5)  | Non-stiff systems | `tsit5.py`     |
| **Rodas5P** | Rosenbrock-W (order 5) | Stiff systems     | `rodas5P.py`   |

Rodas5P supports an `lu_precision` (`"fp32"`/`"fp64"`) knob: the FP32
factorisation halves shared-memory use without lowering method order, since the
Rosenbrock order conditions hold under an approximate Jacobian.

## API

All solvers expose a single `solve(...)` entry point that integrates an
ensemble in one call:

```python
from solvers.rodas5P import solve

# ode_fn is a CUDA-device callable: (y, t, p) -> tuple
y = solve(
    ode_fn,
    y0,          # (n_vars,) or (N, n_vars)     initial state(s)
    t_span,      # (n_save,) output times (shared across the ensemble)
    params,      # (n_params,) or (N, n_params) per-trajectory parameters
    rtol=1e-8,
    atol=1e-10,
    first_step=None,
    max_steps=100_000,
    return_stats=False,                  # also return per-step accept/reject counts
    error_weights=None,                  # optional per-component weights (0 = ignore)
    pcoeff=0.0, icoeff=1.0, dcoeff=0.0,  # PID step-controller gains
)
# y has shape (N, n_save, n_vars)
```

Calling conventions:

- The callbacks are compiled with `numba_cuda_mlir`, so they take and return fixed-size
  tuples of scalars rather than arrays, and use `math` rather than `numpy`/`jax.numpy`.
  Plain Python functions are jitted automatically; pre-`cuda.jit`ed ones are used as-is.
- **Rodas5P** (implicit) needs only `ode_fn`. Its Jacobian ∂f/∂y, and the ∂f/∂t
  a non-autonomous system needs to retain full order, are differentiated out of
  `ode_fn` with [numba-enzyme](https://github.com/Qruise-ai/numba-enzyme),
  which runs Enzyme over the callback's LLVM IR.
- **Tsit5** (explicit) needs no derivatives at all.

Importing `solvers` enables JAX float64.

## Why hand-written CUDA and not a tile DSL

The obvious question about a project that hand-writes CUDA in 2026 is why it
does not use [Triton][triton], [cuTile][cutile] or [Pallas][pallas], all of
which exist to save you from exactly this. The answer is that all three are
*tile-level* languages and modax's unit of work is a thread.

**The parallelism is across trajectories, not within one.** A tile DSL asks you
to describe a block of an array and hands the mapping onto threads to the
compiler. Triton's own documentation puts it as "Blocked Program, Scalar
Threads" against CUDA's "Scalar Program, Blocked Threads": a program instance
owns a whole block, there is no `threadIdx` to write against, and per-lane
conditional work is expressed by *masking* rather than branching. Pallas is the
same model one level down — it lowers to Mosaic GPU, where each operation
occupies a whole warpgroup with its warps in lockstep — and cuTile's entire
premise is that you "focus on data tiles rather than managing individual
threads", with `ct.load`/`ct.store`/`ct.bid` in place of thread indices, shared
memory and barriers.

That is the right abstraction when one problem is big enough to fill a block.
modax's is not: a trajectory is under 200 doubles, and there are 10⁵ of them.
The work per *thread* is a whole ODE solve.

**Adaptive stepping needs per-thread control flow.** Every trajectory carries
its own `t`, `dt`, error history and accept/reject decision, and takes a
different number of steps — that is what adaptivity means, and on a stiff
ensemble the spread across trajectories is the dominant cost term. Under masked
tile semantics a lane that has finished still pays for every step its
neighbours take, for as long as the widest trajectory in the tile runs. In a
SIMT kernel it retires: `rodas5P`'s stage loop is a sequence of `if active:`
regions separated by `cuda.syncthreads()` at hand-picked points, so a finished
lane skips the arithmetic and only meets the barriers. Neither the divergent
branch nor the hand-placed barrier is expressible in a tile language, where
synchronisation is inserted by the compiler.

**We pick the memory layout; a tile compiler picks it for you.** Triton
advertises "shared memory allocation/synchronisation, automatic coalescing,
thread swizzling" as things its compiler does automatically, and cuTile the
same. Those decisions are most of modax's performance:

- `tsit5`'s shared backend stores its nine stage vectors as `(n_vars, block)` —
  transposed, so consecutive lanes are contiguous and the access is
  bank-conflict-free — and falls back to a global-memory kernel only when the
  state will not fit.
- `rodas5P` puts its LU workspace and stage vectors in shared memory but keeps
  the Jacobian in per-thread buffers a column at a time, rather than the full
  `n_vars ** 2` matrix, and maps `threadIdx.x` onto `(batch, lane)` itself.
- Linking the Enzyme-derived Jacobian as NVVM LTO IR rather than PTX is what
  keeps those columns in registers at all; as PTX they become `2 * n_vars`
  doubles of local memory per thread and the solve is 10-20% slower. A single
  placement decision, worth 10-20%, of exactly the kind a tile compiler owns.

**The pieces we build on are numba-cuda device code.** Rodas5P's factorisation
is nvmath-python's cuSOLVERDx `LUPivotSolver`, whose device API documents
support for the numba-cuda and numba-cuda-mlir compilers specifically; the
Jacobian comes from running Enzyme over the user callback's LLVM IR and
nvJitLink-ing the result into the kernel. Triton's only hook for foreign device
code is `extern_elementwise`/libdevice, which is elementwise — an ODE
right-hand side maps `n_vars` scalars to `n_vars` scalars, and an LU solve is
not elementwise at all.

What this costs us is real: tensor cores, TMA, automatic software pipelining
and the coalescing and bank-conflict analysis all have to be done by hand or
done without. But those features are aimed at large dense tiles, and a <200D
ODE has none — there is no matmul here big enough to want a tensor core. The
portability argument also cuts less than it looks: cuTile is new enough that
its hardware coverage is still filling in (Hopper support is listed as future
work), while `cuda.jit` runs anywhere numba-cuda does.

The honest summary is that tile DSLs and modax are solving different problems.
Triton, cuTile and Pallas exist because one large tiled kernel is hard to
schedule well and the compiler can do it better than you. modax's kernel is
trivially parallel and arithmetically small; what is hard is keeping 10⁵
independent, divergent, register-resident state machines fed. That is a
thread-level problem, so it is written in a thread-level language.

[triton]: https://triton-lang.org/
[cutile]: https://github.com/NVIDIA/cutile-python
[pallas]: https://docs.jax.dev/en/latest/pallas/index.html

## Install & run

```bash
uv sync                 # CPU
uv sync --extra cuda13  # or --extra cuda12, for GPU

uv run pytest
uv run ruff format && uv run ruff check --fix
```

`uv sync` resolves `numba-enzyme` from a wheel in `wheels/`, which is too large
to commit. Build it first — see [wheels/README.md](wheels/README.md).

## Examples

Worked end-to-end problems live in `examples/` (each with its own README):

- `bbn_estimation/` — toy Big Bang Nucleosynthesis network with nested-sampling
  parameter estimation and a modax/Diffrax/scipy solver benchmark;
- `21cm_igm_evolution/` — toy global 21cm IGM thermal/ionisation history;
- `mukhanov_sasaki/` — Mukhanov–Sasaki mode evolution.

Scaling, dimensionality and divergence benchmarks are under `scripts/`.
