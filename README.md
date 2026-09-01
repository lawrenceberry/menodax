# modax

GPU-accelerated ODE solvers for **massive ensembles** (1-100k) of low-dimensional (<200D) ODE trajectories, built on
JAX and Numba-CUDA. Applications include: Bayesian parameter inference, uncertainty quantification and the integration of physically uncoupled systems.

Every solver is a **Numba-CUDA custom kernel** (`*numba.py`): one CUDA thread
per trajectory, hand-written step kernels with in-kernel LU factorisation,
exposed to JAX as an XLA FFI custom call. That binding makes each solver an
ordinary JAX primitive — `jit`-traceable, and `vmap` over a single solve lowers
to one native ensemble launch.

## Solvers (`solvers/`)

| Method      | Type                   | Use for           | File           |
|-------------|------------------------|-------------------|----------------|
| **Tsit5**   | Explicit RK (order 5)  | Non-stiff systems | `tsit5.py`     |
| **Rodas5P** | Rosenbrock-W (order 5) | Stiff systems     | `rodas5P.py`   |
| **KenCarp5**| ESDIRK (order 5)       | Stiff systems     | `kencarp5.py`  |

Implicit solvers support an `lu_precision` (`"fp32"`/`"fp64"`) knob: the FP32
factorisation halves shared-memory use without lowering method order, since the
Rosenbrock/SDIRK order conditions hold under an approximate Jacobian.

## API

All solvers expose a single `solve(...)` entry point that integrates an
ensemble in one call:

```python
from solvers.rodas5P import solve

# ode_fn / jac_fn are CUDA-device callables: (y, t, p) -> tuple
y = solve(
    ode_fn,
    jac_fn,
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

- The callbacks are compiled with `numba.cuda`, so they take and return fixed-size
  tuples of scalars rather than arrays, and use `math` rather than `numpy`/`jax.numpy`.
  Plain Python functions are jitted automatically; pre-`cuda.jit`ed ones are used as-is.
- **Implicit solvers** (`rodas5P`, `kencarp5`) take an explicit
  `jac_fn` alongside `ode_fn`, plus an optional `time_jac_fn` (∂f/∂t) that
  non-autonomous systems need to retain full order.
- **Tsit5** (explicit) needs no Jacobian.

Importing `solvers` enables JAX float64.

## Install & run

```bash
uv sync                 # CPU
uv sync --extra cuda13  # or --extra cuda12, for GPU

uv run pytest
uv run ruff format && uv run ruff check --fix
```

## Examples

Worked end-to-end problems live in `examples/` (each with its own README):

- `bbn_estimation/` — toy Big Bang Nucleosynthesis network with nested-sampling
  parameter estimation and a modax/Diffrax/scipy solver benchmark;
- `21cm_igm_evolution/` — toy global 21cm IGM thermal/ionisation history;
- `mukhanov_sasaki/` — Mukhanov–Sasaki mode evolution.

Scaling, dimensionality and divergence benchmarks are under `scripts/`.
