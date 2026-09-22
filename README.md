# menodax

**[Documentation](https://lawrenceberry.github.io/menodax/)**

<!-- --8<-- [start:overview] -->

GPU-accelerated ODE solvers for **massive ensembles** (1-100k) of low-dimensional (<200D) ODE trajectories, built on
JAX and Numba-CUDA-MLIR. Applications include: Bayesian parameter inference, uncertainty quantification and the integration of physically uncoupled systems.

Every solver is a hand-written **CUDA custom kernel** compiled by
Numba-CUDA-MLIR: one CUDA thread
per trajectory, hand-written step kernels with in-kernel LU factorisation,
exposed to JAX as an XLA FFI custom call. That binding makes each solver an
ordinary JAX primitive — `jit`-traceable, and `vmap` over a single solve lowers
to one native ensemble launch.

<!-- --8<-- [end:overview] -->

<!-- --8<-- [start:solvers] -->

## Solvers (`menodax/`)

| Method      | Type                   | Use for           | File           |
|-------------|------------------------|-------------------|----------------|
| **Tsit5**   | Explicit RK (order 5)  | Non-stiff systems | `tsit5.py`     |
| **Rodas5P** | Rosenbrock-W (order 5) | Stiff systems     | `rodas5P.py`   |

Rodas5P supports an `lu_precision` (`"fp32"`/`"fp64"`) knob: the FP32
factorisation halves shared-memory use without lowering method order, since the
Rosenbrock order conditions hold under an approximate Jacobian.

<!-- --8<-- [end:solvers] -->

<!-- --8<-- [start:sparse] -->

## Sparse systems

Rodas5P takes a `sparsity` pattern, and that one argument is the whole
interface — there is no linear solver to write or to pass:

```python
y = solve(ode_fn, y0, t_span, params,
          sparsity=pattern)   # (n_vars, n_vars) mask, scipy sparse, or (nnz, 2)
```

A pattern buys two separate things. The Jacobian is recovered in one Enzyme
sweep per *colour* of the pattern's column intersection graph rather than one
per column, since columns sharing no row can be seeded together and the pattern
says which output component belongs to which. And the iteration matrix
`M = I/(hγ) − J` gets a **direct sparse solver compiled for that exact
structure**: an in-kernel sparse LU and a pair of sparse triangular solves, one
trajectory per thread, in place of the dense LU. The pattern must be a superset
of the true nonzeros — colouring a superset only costs sweeps, colouring a
subset silently corrupts entries — but it need *not* cover the factorisation's
fill-in, which is worked out from it. With no pattern, every column gets its own
colour and the matrix is factorised densely: the same mechanism at its
uninformative end rather than a second code path.

On DISCO-EB's 50-variable Einstein-Boltzmann system this is **22% faster** than
the hand-written Schur block-LU it replaced, and it asks nothing of the caller
but the pattern.

### The choices behind it

All of the analysis happens once, on the host, when the kernel is built
(`menodax/_sparse_direct.py`).

**AMD for the ordering, not COLAMD.** The obvious alternative, COLAMD, orders
the *columns* so that fill stays bounded whatever row permutation partial
pivoting later chooses. That is the right objective exactly when there will be
pivoting — and there will not be, because the pattern is compiled into the
kernel and cannot depend on the numbers. COLAMD's permutation is also one-sided,
so it moves the diagonal off the diagonal, and this factorisation needs the
diagonal precisely where `I/(hγ)` puts it. AMD instead minimises (approximately)
the fill of the Cholesky factor of `S + Sᵀ`, which is the standard bound on the
fill of an unpivoted `LU` of `S`, and it does so with a *symmetric* permutation
`P S Pᵀ` that leaves every diagonal entry on the diagonal. It is what UMFPACK
and SuperLU use in their "symmetric mode", for these reasons, and an iteration
matrix is about as close to structurally symmetric as an unsymmetric matrix
gets. It comes from SuiteSparse through
[scikit-sparse](https://scikit-sparse.readthedocs.io), which builds against
`libsuitesparse-dev` or equivalent on the machine; `ordering="natural"` skips
the ordering and needs neither.

**No pivoting at all.** The pattern has to be fixed at compile time and the same
in every thread, so rows cannot be swapped on the numbers — which would also
reintroduce the warp divergence one-trajectory-per-thread is there to avoid. Two
things make that sound. The permutation is symmetric, so `M`'s diagonal stays on
the diagonal and `I/(hγ)` guarantees every pivot is structurally present and
grows without bound as the step shrinks. And Rodas5P is a Rosenbrock-**W**
method: order 5 survives an approximate factorisation, so a badly conditioned
pivot costs step-size control rather than correctness, and the controller is
what notices. A pivot that reaches exactly zero leaves an infinity, the error
norm goes to NaN, the step is rejected, and the smaller step puts a larger
`1/(hγ)` on that very diagonal.

**A symbolic factorisation for the footprint, not a trial numeric one.** The
`L + U` pattern comes from pattern-only Gaussian elimination, which is exact: it
is what the numeric factorisation will touch, no more and no less. Factorising a
sample matrix and counting cannot be — a coefficient that happens to vanish for
those particular numbers, or an exact cancellation, drops an entry another
right-hand side needs, and the buffer is then one slot short in a kernel with no
way to say so. It is also cheaper, needing neither a plausible matrix nor a
device. The implementation is bit-per-entry over the whole matrix, `O(n³/64)`
time and `O(n²)` bits, which for the tens-to-a-few-hundred variables these
solvers target analyses in milliseconds and buys nothing back from a sparse
symbolic algorithm.

**CSR, not CSC.** Every one of the three routines that reads the matrix reads it
*by rows*: the up-looking factorisation takes row `i` and subtracts multiples of
the rows above it, the forward substitution is a dot product of row `i` of `L`
with the solution so far, and the back substitution is the same over row `i` of
`U`. One row-major image serves all three. CSC would have to be transposed for
two of them, and a column-oriented factorisation would still leave the solves
wanting rows. `L` and `U` share that one image — `L` strictly left of the
diagonal, `U` from it rightwards — because the factorisation is in place and a
unit diagonal needs no storage, so the buffer is exactly `nnz(L + U)`, which is
per-thread local memory and the thing that bounds occupancy.

**The Jacobian is written straight into the factorisation's buffer.** Colouring
and storage are separate questions, and the AD's colour sweeps deposit `−J` at
the CSR slots the factorisation will read, with the fill-in slots simply cleared
beforehand. Nothing is staged through global memory and read back, and nothing is
expanded to a dense matrix in between. It also means the pattern may be declared
as tightly as it really is: a hand-written solver owning its own buffer had to
declare its fill-in in the pattern to have somewhere to put it, and paid colours
for that.

**Straight-line code where it fits.** Table-driven, a sparse routine spends a
broadcast load on the index of every value before it can issue the load of the
value itself, and that dependent pair is only free when enough other
trajectories are in flight to cover it. DISCO-EB's single-cosmology case is 128
trajectories — four warps on a 46-SM device — and nothing covers it. So below
`MAX_UNROLLED_SUBSTITUTIONS` / `MAX_UNROLLED_UPDATES` the routines are emitted
as straight-line code with every slot a literal, and above them they fall back
to loops over index tables in constant memory. Unrolling costs no registers,
since the kernel indexes both the matrix and the right-hand side with loop
variables of its own and neither can leave local memory whatever this does — it
trades index loads for instruction count and nothing else. Measured on DISCO-EB
at N128: **528 ms** table-driven, **419 ms** with the solves unrolled, **398 ms**
with the factorisation unrolled too, against **509 ms** for the hand-written
Schur solver. The two emissions are checked against each other and required to
agree bit for bit.

**What it finds on a real problem.** DISCO-EB's Einstein-Boltzmann Jacobian is a
densely coupled core bordered by tridiagonal free-streaming hierarchies. AMD
returns a *perfect* elimination order for it — zero fill, `nnz(L + U) = nnz(J)` —
and the order it finds is the hand-written Schur solver's: peel each hierarchy
from its truncated end inwards, where every variable has degree two, then
eliminate the dense core last.

<!-- --8<-- [end:sparse] -->

<!-- --8<-- [start:api] -->

## API

All solvers expose a single `solve(...)` entry point that integrates an
ensemble in one call:

```python
from menodax.rodas5P import solve

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
    sparsity=None,                       # Jacobian pattern; see "Sparse systems"
    ordering="amd",                      # its fill-reducing permutation
)
# y has shape (N, n_save, n_vars)
```

Calling conventions:

- The callbacks are compiled with `numba_cuda_mlir`, so they take and return fixed-size
  tuples of scalars rather than arrays, and use `math` rather than `numpy`/`jax.numpy`.
  Plain Python functions are jitted automatically; pre-`cuda.jit`ed ones are used as-is.
  A right-hand side that must also run under `jax` — to compare against a
  Diffrax baseline, say — need not be written twice: `examples/dual_backend.py`
  builds both forms from one body, parameterised over the names the two
  backends spell differently.
- **Rodas5P** (implicit) needs only `ode_fn`. Its Jacobian ∂f/∂y, and the ∂f/∂t
  a non-autonomous system needs to retain full order, are differentiated out of
  `ode_fn` with [numba-enzyme](https://github.com/Qruise-ai/numba-enzyme),
  which runs Enzyme over the callback's LLVM IR.
- **Tsit5** (explicit) needs no derivatives at all.

Importing `menodax` enables JAX float64.

<!-- --8<-- [end:api] -->

<!-- --8<-- [start:gradients] -->

## Gradients

Both solvers are differentiable with respect to `y0` and `params`:

```python
import jax
from menodax.rodas5P import solve

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

**(c) Jointly — what menodax does.** One Rosenbrock step on $[y, S]$, exploiting
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

menodax gets them from [numba-enzyme](https://github.com/Qruise-ai/numba-enzyme),
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
differentiate `y0` as well.

**Cost is linear in `n_sens`, because the sensitivities are never factorised.**
This is the whole point of the block-triangular structure. The joint iteration
matrix has the same `M0 = I/(h*gamma) - J_y` on every diagonal block, so a step
factorises `M0` exactly **once**, at `O(n_vars^3)`, and every sensitivity column
then reuses that factorisation. What an extra column adds is a forward and back
substitution against factors that already exist — `O(n_vars^2)` — plus two
Enzyme sweeps per stage and its share of the occupancy. Per step:

```
cost  ~  O(n_vars^3)              one LU, however many columns
       + (1 + n_sens) * O(n_vars^2)   one substitution per column per stage
       + (1 + n_sens) * O(n_vars)     right-hand sides and Enzyme sweeps
```

There is no second cubic term anywhere in that. Nothing about differentiating
costs another factorisation, which is exactly why the measured overhead below
tracks `1 + n_sens` and not something steeper.

**Against parameter count**, at `n_vars = 8`, 1000 trajectories, fp32:

| `n_params` | joint width | solve | `value_and_grad` | overhead |
|---|---|---|---|---|
| 1 | 16 | 6.30 ms | 10.44 ms | **1.66×** |
| 2 | 24 | 6.42 ms | 14.32 ms | **2.23×** |
| 4 | 40 | 6.56 ms | 20.51 ms | **3.13×** |
| 8 | 72 | 6.98 ms | 48.52 ms | **6.95×** |

So cost is roughly **linear in `1 + n_params`**, with a coefficient a little
under one — about `0.7 * (1 + n_params)` here — the discount being the
factorisation that all the columns share. Budget accordingly: ten parameters is
an order of magnitude, not a rounding error, but it is an order of magnitude and
not the `n_params`-fold repetition of the cubic that differentiating the
factorisation itself would cost.

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

menodax is built for **massive ensembles of low-dimensional systems with few
parameters**, and that regime picks the method. The three candidates scale
differently in the state dimension `n_vars` and the parameter count
`n_params`:

| approach | work per step | extra memory | grows with |
|---|---|---|---|
| **Continuous forward sensitivity** (menodax) | $O(n_\text{vars}^3 + n_\text{params}\,n_\text{vars}^2)$ | $O(n_\text{vars}\,(1 + n_\text{params}))$ | `n_params` |
| **Continuous adjoint** (backward) | $O(n_\text{vars}^3)$ backward, plus the forward solve and its checkpoint re-solves | $O(n_\text{vars} + n_\text{params})$ plus checkpoints | number of output cotangents — *not* `n_params` |
| **Direct auto-diff through the solver** | $O(n_\text{params}\,n_\text{vars}^3)$ | $O(n_\text{vars}\,(1 + n_\text{params}))$ forward; a full tape in reverse | `n_params`, **on the cubic term** |

The decisive row is the last one. A step's cost is dominated by factorising the
iteration matrix, $O(n_\text{vars}^3)$. Forward sensitivity pays that **once**
and each parameter column then costs a substitution against factors that already
exist, so the cubic term never multiplies:

$$O(n_\text{vars}^3 + n_\text{params}\,n_\text{vars}^2) \quad\text{against}\quad O(n_\text{params}\,n_\text{vars}^3)$$

Direct auto-diff has no way to know that. Handed the kernel's hand-written LU as
ordinary scalar code, Enzyme differentiates the factorisation *itself* —
propagating a tangent through every one of its $O(n_\text{vars}^3)$ operations,
once per direction. That is a factor of `n_params` on the dominant term, and it
is structure no differentiator can recover on its own: what menodax does by hand
is apply the differentiation rule for a linear solve, `M dk = dr - dM k`, which
reuses `M`'s factors. An auto-diff system that treats the solve as a primitive
*with* that rule attached would recover the same scaling; one differentiating
the scalar code beneath it would not.

Against the adjoint, the trade is the usual one: its cost is independent of
`n_params` and instead proportional to the number of outputs differentiated, so
it wins once parameters outnumber state dimensions. menodax targets the opposite
corner — the BBN example fits 2 parameters to a 4-species network — and the
adjoint would additionally need either a backwards solve, which is unstable for
the stiff, dissipative systems Rodas5P exists to handle, or a checkpointed
reverse pass whose gradients are no longer consistent with the discrete solve
the forward pass actually performed.

Differentiating `y0` as well adds `n_vars` columns rather than one, so it enters
the table wherever `n_params` appears, and is only practical at low dimension.

Forward sensitivities also fit the execution model. The variational equation is
per-trajectory and couples nothing across the ensemble, so the joint system is
still one CUDA thread per trajectory with no cross-trajectory communication.

The asymptotics are not the only obstacle to differentiating the solver kernel
itself with Enzyme, the way `ode_fn` is differentiated; it is impractical here
for mechanical reasons too. The kernels are not ordinary functions: they are
hand-written CUDA with per-trajectory adaptive stepping, hand-written linear
algebra over thread-local buffers, and `syncthreads` barriers in Tsit5's shared
backend.
Reverse mode through that, and through the step controller's data-dependent
control flow, is exactly where Enzyme-GPU stops working, and a reverse pass would in any
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

<!-- --8<-- [end:gradients] -->

<!-- --8<-- [start:install] -->

## Install & run

```bash
uv sync                 # CPU
uv sync --extra cuda13  # or --extra cuda12, for GPU

uv run pytest
uv run ruff format && uv run ruff check --fix
```

`uv sync` resolves `numba-enzyme` from a prebuilt wheel published as a GitHub
release asset, hash-pinned in `uv.lock`; nothing has to be built by hand. See
[wheels/README.md](wheels/README.md) for what is in it and why.

<!-- --8<-- [end:install] -->

<!-- --8<-- [start:examples] -->

## Examples

Worked end-to-end problems live in `examples/` (each with its own README):

- `bbn_estimation/` — toy Big Bang Nucleosynthesis network with nested-sampling
  parameter estimation and a menodax/Diffrax/scipy solver benchmark;
- `21cm_igm_evolution/` — toy global 21cm IGM thermal/ionisation history;
- `mukhanov_sasaki/` — Mukhanov–Sasaki mode evolution.

<!-- --8<-- [end:examples] -->
