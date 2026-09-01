# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JAX-based stiff ODE solver framework with GPU-accelerated Pallas/Triton custom kernels. Implements Rosenbrock-family solvers (Rodas5, Rosenbrock23) for large ensemble solves (up to 100K+ trajectories) on GPU.

## Commands

```bash
# Install dependencies (uses uv)
uv sync
uv sync --extra gpu  # with CUDA support

# Run tests (skips slow 72D tests by default)
uv run pytest

# Run a specific test file
uv run pytest tests/test_robertson.py
uv run pytest tests/test_30d.py

# Run a specific test
uv run pytest tests/test_robertson.py::test_rodas5_single -v

# Run slow tests (72D Boltzmann)
uv run pytest -m slow

# Format
uv run ruff format
uv run ruff check --fix  # lint + auto-fix
```

## Architecture

### Solvers (`solvers/`)

Progression from reference to optimized implementations:

- **rodas5.py** — Reference Rodas5 (order 5) using `jax.jacobian()` + LU solve. Pure JAX, vmap for ensembles.
- **rodas5_custom_kernel.py** — Pallas/Triton kernel using Cramer's rule. Robertson-specific (3×3 only).
- **rodas5_custom_kernel_v2.py** — General-purpose Pallas/Triton Rodas5. Works with any ODE dimension. Uses JVP-based Jacobian and Pallas Refs for O(1) Triton code size. This is the latest/recommended solver.
- **rosenbrock23_custom_kernel.py** — General Rosenbrock23 (order 2) Pallas kernel. Factory pattern: `make_solver(ode_fn)` returns a solver function.
- **diffrax_kvaerno5.py** / **scipy_bdf.py** — External reference solvers for validation.

### Solver API pattern

```python
# Single solve
y_final = solve(f, y0, t_span, rtol, atol, first_step, max_steps)

# Ensemble (vmap-based)
y_batch = solve_ensemble(f, y0, t_span, params_batch, ...)

# Pallas factory (rosenbrock23, rodas5_v2)
solve = make_solver(ode_fn)
y_batch = solve(y0_batch, t_span, params_batch, ...)
```

### GPU Kernel Design (Pallas/Triton)

- Block size = 32 trajectories (one CUDA warp)
- Input arrays padded to power-of-2 (Triton requirement)
- Pallas Refs used for mutable stage vectors (avoids array slicing)
- `fori_loop` for LU decomposition with dynamic indexing via `pl.ds()`
- Active masking: only update non-finished trajectories

### Jacobian Strategy

- **rodas5.py**: Full `jax.jacobian()` (reverse-mode)
- **v2/Pallas solvers**: Per-column JVP (forward-mode AD), more efficient in kernels

### Test Systems

- **Robertson** (3D): Classical stiff test, conservation law `sum(y)=1`
- **Linear chain** (30D/50D/70D): Stiff decay chain, `sum(y)=1`
- **Boltzmann** (72D): Marked `@pytest.mark.slow`, skipped by default

Tests validate against reference solvers (scipy BDF, rodas5) and check conservation invariants. Benchmarks use `pytest-benchmark` with pedantic mode (1 warmup, 1 round).
