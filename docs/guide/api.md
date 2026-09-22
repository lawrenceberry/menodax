--8<-- "README.md:api"

## Shapes

| Argument | Shape | Notes |
|----------|-------|-------|
| `y0` | `(n_vars,)` or `(N, n_vars)` | broadcast across the ensemble when 1-D |
| `t_span` | `(n_save,)` | output times, shared across the ensemble |
| `params` | `(n_params,)` or `(N, n_params)` | one row per trajectory |
| result | `(N, n_save, n_vars)` | |

`N` is taken from whichever input is batched; if both are, they must agree.
`return_stats=True` adds a dict of per-trajectory step counters alongside the
history.

## Arguments both solvers take

`rtol`, `atol`
:   The tolerances of the weighted RMS error norm the step controller works
    against.

`first_step`
:   Pins the initial step. It reaches the kernel as a launch-time scalar rather
    than something derived on the host from `t_span`, because that argument is
    traced. Omitting it, or passing a non-positive value, hands the kernel a
    sentinel and it starts from 1e-6 of its own integration window.

`max_steps`
:   The per-trajectory step budget.

`error_weights`
:   Per-component weights, `(n_vars,)` or `(N, n_vars)`, applied in the error
    norm. A weight of 0 excludes that component from step-size control.

`pcoeff`, `icoeff`, `dcoeff`
:   PID step-controller gains. The default `(0, 1, 0)` is the classic
    I-controller.

`return_stats`
:   Also return accepted/rejected step counts per trajectory.

`sens_error_control`, `sens_param_columns`
:   Control how a differentiated solve treats the sensitivity block — see
    [Gradients](gradients.md).

## Arguments only Rodas5P takes

`lu_precision`
:   `"fp32"` (the default) or `"fp64"`, the precision of the per-step LU
    factorisation and triangular solves. The state, right-hand side, Jacobian
    and error estimate are always float64. The Rosenbrock--Wanner order
    conditions hold under an approximate Jacobian, so `"fp32"` does not lower
    the method's order; it halves the factorisation's footprint and uses the
    FP32 throughput. `"fp64"` is there for ill-conditioned systems where the
    FP32 factorisation degrades step-size control.

`sparsity`, `ordering`
:   The Jacobian's sparsity pattern and its fill-reducing permutation — see
    [Sparse systems](sparse.md).

`trajectories_per_block`
:   One thread's worth of work each, defaulting to a warp. Nothing on chip
    bounds it, since every per-trajectory buffer is thread-local.

`tf_index`
:   Names a `params` column holding each trajectory's own end time, for an
    ensemble that does not share one.

`max_registers`
:   Caps the per-thread register count.

`array_rhs`
:   An `f(y, t, p, out)` device function equivalent to `ode_fn`. The primal
    stage evaluations do not need the tuple form — that exists for Enzyme — so
    handing this in lets the eight evaluations per step call it directly.

`save_hook`, `hook_size`, `save_history`
:   A device function `hook(save_idx, y, t, p_row, acc)` the kernel calls at
    every save time with the state at that time. `acc` is the trajectory's row
    of an `(N, hook_size)` output, zeroed at the start and persistent across
    saves, so the hook can accumulate — a line-of-sight integral, say — or
    store derived quantities per save. It lets a consumer of the history run
    inside the launch instead of storing the history, and `save_history=False`
    then shrinks the history output to the final state, `(N, 1, n_vars)`. A
    hooked solve returns `(hist, hook_out)` and is a plain ensemble launch: no
    `jax.vmap`, no differentiation.

## Arguments only Tsit5 takes

`backend`
:   `"auto"` (the default), `"shared"` or `"global"`, selecting whether the
    stage workspace lives in shared memory or in global scratch. `"auto"`
    picks on whether the state fits on chip; the two paths are bit-identical.
