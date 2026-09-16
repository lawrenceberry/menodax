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
    sens_error_control=True,             # error-control the sensitivities too
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

## Gradients

Both solvers are differentiable with respect to `y0` and `params`:

```python
import jax
from solvers.rodas5P import solve

def loss(params):
    y = solve(ode_fn, y0, t_span, params)
    return jnp.sum((y[:, -1, :] - observed) ** 2)

value, grad = jax.value_and_grad(loss)(params)   # one joint solve
```

`jax.jvp`, `jax.jacfwd`, `jax.grad`, `jax.jacrev` and `jax.value_and_grad` all
work, inside `jit` and `vmap` as usual. Derivatives are computed only when a
differentiation transform actually asks for them — a plain `solve(...)` runs the
same kernel it always did and pays nothing.

Asking for a derivative integrates the **continuous forward-sensitivity
system** alongside the state. Writing $S = \partial y/\partial\theta$,
differentiating $y' = f(t, y, p)$ with respect to $\theta$ gives the variational
equation

$$\frac{dS}{dt} = J_y(t)\,S(t) + J_p(t), \qquad J_y = \frac{\partial f}{\partial y},\quad J_p = \frac{\partial f}{\partial \theta}$$

which the solver integrates jointly with the state as one larger ODE

$$\frac{\partial}{\partial t}\begin{bmatrix} y \\ S \end{bmatrix} = \begin{bmatrix} f(t, y, p) \\ J_y(t)\,S + J_p(t) \end{bmatrix}$$

so `jax.value_and_grad` costs one solve rather than one for the value and
another for the derivative.

### How the joint system is solved

There are three ways to arrange this, and they are not equally good.

**(a) Two separate solves** — integrate `y` to completion, then integrate `S`
against it. The sensitivity solve needs `y(t)` at *its own* step and stage
points, which the state solve never produces, so this means storing the whole
trajectory: at $10^5$ trajectories and $\sim\!10^3$ adaptive steps that is
hundreds of gigabytes, on a device with tens. It also runs two independent
adaptive loops per trajectory, doubling the warp-divergence penalty that
dominates this kernel's cost. Rejected.

**(b) Staggered** — advance `y` over a step, then advance `S` over the same step
using `y`'s stage values. No trajectory storage, and the sensitivity
subsystem's Jacobian with respect to its own unknown is exactly $J_y$. But for a
*linearly implicit* method this does not avoid anything: treating `y(t)` as a
known function of `t` moves the state dependence into explicit time dependence,
and Rosenbrock's $\partial F/\partial t$ term picks it straight back up by the
chain rule. It costs a second pass through the tableau and the state's stage
values kept alive, for the same derivatives.

**(c) Jointly — what modax does.** One Rosenbrock step on $[y, S]$, exploiting
the fact that the joint Jacobian is *exactly* block lower triangular, because
`f` does not depend on `S`:

$$A = \begin{bmatrix} J_y & 0 \\ L & J_y\end{bmatrix}, \qquad L = \frac{\partial}{\partial y}\left(J_y S + J_p\right)$$

"Joint" therefore does **not** mean factorising an $n_\text{aug} \times
n_\text{aug}$ matrix. The iteration matrix $M = I/(h\gamma) - A$ inherits the
structure, and every diagonal block is the *same* $M_0 = I/(h\gamma) - J_y$, so
one stage is a block forward substitution

$$M_0\,k_y = r_y, \qquad M_0\,k_{S_k} = r_{S_k} + L_k\,k_y$$

against a single factorisation. The LU stays $n_\text{vars}^3$ instead of
$n_\text{vars}^3(1+n_\text{sens})^3$, and shared memory $n_\text{vars}^2$
instead of $n_\text{vars}^2(1+n_\text{sens})^2$.

(c) was chosen because it needs exactly the same derivatives as (b) while
sequencing them in one pass, under one step-size controller with one rejection
decision — and because the triangular structure means sequencing the state
before the sensitivities is not an approximation but the shape of the exact
solve. Within a stage it *is* staggered; it simply does not pretend the
coupling is absent.

### Second derivatives, and why they are unavoidable

The coupling block $L$ is a second derivative of the *original* right-hand side
— with respect to (state, state) and (state, parameter):

$$L_k = \frac{\partial^2 f}{\partial y\,\partial y}\!\left[\cdot,\,S_k\right] + \frac{\partial^2 f}{\partial y\,\partial p_k}$$

They appear because $S' = J_y(y)S + J_p(y)$ is a linear ODE whose *coefficients*
depend on `y`, and an implicit method has to differentiate those coefficients.
There is no arrangement that escapes them: a Newton-iterated method (BDF, SDIRK)
could treat $J_y$ as a mere preconditioner and converge regardless, but Rodas5P
is linearly implicit — its Jacobian is inside the formula, so an approximate one
lands in the answer.

modax gets them from [numba-enzyme](https://github.com/Qruise-ai/numba-enzyme),
whose `jvp` composes with itself: `jvp(jvp(f))` is a forward-over-forward
directional derivative, giving $D^2 f(x)[u,v]$. Seeding $u = (S_k, 0, e_k)$ and
$v = (k_y, 0, 0)$ returns $L_k k_y$ directly — the matrix $L_k$ is never formed.
Seeding $v = (0,1,0)$ instead returns the sensitivity rows'
$\partial F/\partial t$, the other second derivative a Rosenbrock method needs.
The same mechanism supplies the first-order right-hand side: $J_y S_k + J_p$
*is* a directional derivative, so it is one sweep per column rather than a whole
Jacobian.

Composition here is not the trivial thing it is in JAX. `jax.jvp` maps a jaxpr
to a jaxpr, so it is closed under itself; numba-enzyme's maps a Python callable
to a *compiled device symbol*, and differentiating that again would hand Enzyme
an external declaration with no body. So the fork records the chain instead of
applying it, and emits every level as a definition in one module, where a single
Enzyme pass resolves the nested markers.

This matters more than it sounds. Dropping $L$ and using the block diagonal
$\mathrm{diag}(J_y, \ldots, J_y)$ is legitimate for a W method — order 5 survives
— but the error constant does not, and the step-size controller pays for it. On
a two-species right-hand side bilinear in state and parameters:

| joint Jacobian | additive `f` ($L = 0$) | bilinear `f` ($L \neq 0$) |
|---|---|---|
| block diagonal (W approximation) | 1.0× the plain solve's steps | **201×** |
| exact, via `jvp(jvp(f))` | 1.0× | **1.2×** |

and on a forced non-autonomous problem with a closed-form sensitivity, the
gradient error at `rtol=1e-6` improves from $4.6\times10^{-3}$ to
$4.6\times10^{-8}$, converging at the method's proper order instead of crawling.

Details:

- Only the blocks you differentiate are integrated. A gradient with respect to
  `params` alone carries `n_params` sensitivity columns; one with respect to
  `y0` as well carries `n_vars` more.
- The sensitivities take part in step-size control by default (~20% extra steps),
  so the gradient's accuracy is tied to `rtol` rather than left to luck. Pass
  `sens_error_control=False` to drop them from the error norm: the joint solve
  then takes exactly the steps the plain solve takes and returns the same value.
- `t_span` is not differentiable; differentiating through it raises.

### What gradients cost

The joint system is `n_vars * (1 + n_sens)` wide, where `n_sens` is the number
of directions actually differentiated — `n_params`, plus `n_vars` more if you
differentiate `y0` as well. That width, not the arithmetic, is what sets the
price: Rodas5P keeps ten stage vectors of the joint state in shared memory, so
the batch per block shrinks as it grows. The `O(n_vars^3)` LU does *not* grow —
one factorisation of `M0` serves the state and every sensitivity column — so
what each extra column adds is an `O(n_vars^2)` triangular solve, two Enzyme
sweeps per stage, and its share of the occupancy.

**Against parameter count**, at `n_vars = 8`, 1000 trajectories, fp32:

| `n_params` | joint width | solve | `value_and_grad` | overhead |
|---|---|---|---|---|
| 1 | 16 | 6.30 ms | 10.44 ms | **1.66×** |
| 2 | 24 | 6.42 ms | 14.32 ms | **2.23×** |
| 4 | 40 | 6.56 ms | 20.51 ms | **3.13×** |
| 8 | 72 | 6.98 ms | 48.52 ms | **6.95×** |

So cost is roughly **linear in `1 + n_params`**, with a coefficient a little
under one — about `0.7 * (1 + n_params)` here — because the shared factorisation
is amortised over all the columns. Budget accordingly: ten parameters is an
order of magnitude, not a rounding error.

**Against state dimension**, one parameter, on the VdP lattice at 1000
trajectories, fp32:

| `n_vars` | joint width | solve | `value_and_grad` | overhead |
|---|---|---|---|---|
| 8 | 16 | 3.52 ms | 5.85 ms | **1.66×** |
| 16 | 32 | 8.61 ms | 19.84 ms | **2.31×** |
| 32 | 64 | 20.39 ms | 49.45 ms | **2.43×** |
| 48 | 96 | 31.93 ms | 84.85 ms | **2.66×** |

A single sensitivity column costs between 1.7× and 2.7× across that range —
flat enough to plan around, and creeping up rather than down, since the extra
triangular solves and Enzyme sweeps scale with `n_vars` even though the
factorisation they reuse does not. In fp64 the ratio is *lower* (2.21× at
`n_vars = 48`), because the shared LU is twice the work and so a larger share of
the step.

Two things to watch:

- **Differentiating `y0` adds `n_vars` columns**, not one, so it is only
  practical at low dimension. On 3-species Robertson at `N = 20000`, a gradient
  with respect to the three rate parameters costs ~9× the value; adding `y0`
  takes it to six columns and ~34×. At `n_vars = 48` it is not an option at all.
- **Shared memory is the hard limit.** Rodas5P re-fits its LU batch to the
  augmented footprint automatically, and raises a clear error if even one
  trajectory per block will not fit. At `n_vars = 48` that leaves room for about
  one parameter column.

Tsit5 is cheaper per column (it forms no Jacobian and needs no second
derivatives) and is bounded by memory traffic rather than shared memory, so it
scales further in `n_sens` — at the usual cost of needing a non-stiff problem.

### Why continuous forward sensitivities

modax is built for **massive ensembles of low-dimensional systems with few
parameters** — the regime where forward sensitivity analysis is the cheap
option. Its cost scales with the number of directions differentiated, so it wins
whenever there are fewer parameters than state dimensions, which is the case
modax targets (the BBN example fits 2 parameters to a 4-species network). An
adjoint method, whose cost is instead independent of the parameter count, would
only start to pay off well outside that regime — and would need either a
backwards solve, which is unstable for the stiff, dissipative systems Rodas5P
exists to handle, or a checkpointed reverse pass whose gradients are no longer
consistent with the discrete solve the forward pass actually performed.

Forward sensitivities also fit the execution model. The variational equation is
per-trajectory and couples nothing across the ensemble, so the joint system is
still one CUDA thread per trajectory with no cross-trajectory communication.

The alternative — differentiating the solver kernel itself with Enzyme, the way
`ode_fn` is differentiated — is not practical here. The kernels are not ordinary
functions: they are hand-written CUDA with per-trajectory adaptive stepping,
cooperative lane-striped work, `syncthreads` barriers and shared-memory
workspaces, and Rodas5P calls into nvmath's `LUPivotSolver`, a closed device
template. Reverse mode through cross-thread communication and opaque CUDA library
calls is exactly where Enzyme-GPU stops working, and a reverse pass would in any
case need a tape of every stage of every step — at $10^5$ trajectories and
$\sim\!10^3$ adaptive steps that is hundreds of gigabytes, on a device with tens.
Integrating the sensitivity equation instead keeps the whole derivative inside
the same kernel structure, at the same memory footprint, with the same
per-thread independence.

### Why a stiff ODE has a stiff sensitivity ODE

This is why the sensitivity system goes through the *stiff* solver rather than
being handed to an explicit one: it inherits the state's stiffness exactly.

**Claim.** The joint system $z' = F(z)$ has the same Jacobian spectrum as the
state equation, so every spectral measure of stiffness is identical.

**Proof.** With $z = (y, S_1, \ldots, S_m)$ and
$F_{S_k} = J_y(y)S_k + J_{p,k}(y)$, the joint Jacobian is

$$A = \frac{\partial F}{\partial z} = \begin{bmatrix} J_y & 0 & \cdots & 0 \\ L_1 & J_y & & \\ \vdots & & \ddots & \\ L_m & & & J_y \end{bmatrix}$$

since $\partial F_y/\partial S_k = 0$ (the state equation does not involve $S$)
and $\partial F_{S_k}/\partial S_j = J_y\,\delta_{kj}$. $A$ is block lower
triangular, and the determinant of a block triangular matrix is the product of
the determinants of its diagonal blocks, so

$$\det(A - \lambda I) = \prod_{i=0}^{m} \det(J_y - \lambda I) = \big[\det(J_y - \lambda I)\big]^{m+1}.$$

Hence $\mathrm{spec}(A) = \mathrm{spec}(J_y)$, each eigenvalue with its algebraic
multiplicity multiplied by $m+1$. No new eigenvalues appear, and none are lost.
$\blacksquare$

**Consequence.** The stiffness ratio $\max_i|\mathrm{Re}\,\lambda_i| \,/\,
\min_i|\mathrm{Re}\,\lambda_i|$, the linear stability constraint
$h\lambda \in \mathcal{S}$, and any other spectral criterion take the same value
for the joint system as for the original. If the state equation is stiff, the
joint system is stiff to exactly the same degree — no more, no less.

The same fact seen without matrices: the sensitivity equation is linear in $S$
with homogeneous part $S' = J_y(t)S$, which is the *variational equation* of the
original problem. By variation of constants,

$$S(t) = \Phi(t, t_0)\,S(t_0) + \int_{t_0}^{t} \Phi(t, s)\,J_p(s)\,\mathrm{d}s,$$

where $\Phi$ is the state-transition matrix of that variational equation,
$\Phi' = J_y\Phi$, $\Phi(t_0,t_0) = I$. So sensitivities are propagated by
*precisely* the operator that governs how perturbations of the state evolve. The
violently contracting directions that make the state stiff are the same
directions in which $\Phi$ contracts, and an explicit method integrating $S$
would face exactly the step-size restriction it faces on $y$.

One honest caveat: equal spectra do not mean equal transient behaviour. $A$ is
block triangular and generally not normal, so when $L \neq 0$ the joint system
can show larger transient growth than the state alone even though its eigenvalues
are unchanged. Stiffness in the spectral sense is identical; conditioning need
not be.

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
