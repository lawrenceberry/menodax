# AGENTS.md

This file provides guidance to coding agents working in this repository.

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

# Docs (mkdocs-material + mkdocstrings). Build them in their OWN environment:
# `uv sync` with a group writes .venv, so syncing the docs group into the
# default environment replaces the project's packages with the docs toolchain.
# The group installs neither CUDA nor the project, since griffe reads menodax/
# statically.
export UV_PROJECT_ENVIRONMENT=.venv-docs
uv sync --only-group docs --no-install-project
uv run --no-sync mkdocs serve      # or: mkdocs build --strict
```

The site is `docs/` plus `mkdocs.yml`, and `.github/workflows/docs.yml`
publishes it to GitHub Pages on every push to `master`. The long-form prose is
*not* duplicated there: the pages pull sections out of `README.md`, the
examples' READMEs and `wheels/README.md` with `pymdownx.snippets`, so
`<!-- --8<-- [start:name] -->` / `[end:name]` markers in those files are
load-bearing. Docstrings are rendered by mkdocstrings, so a cross-reference in
one is written ``[`menodax._sparsity`][]`` rather than as a Sphinx role.

Most solver tests need a GPU and are skipped when `numba_cuda_mlir` is unavailable.
`tests/test_examples.py` runs anywhere: it compiles the examples' device
callbacks with `cuda.compile_ptx`, which exercises the full numba typing and
lowering pipeline without a device.

## Architecture

### Solvers (`menodax/`)

| Method   | Type                   | Use for           | File          |
|----------|------------------------|-------------------|---------------|
| Tsit5    | Explicit RK (order 5)  | Non-stiff systems | `tsit5.py`    |
| Rodas5P  | Rosenbrock-W (order 5) | Stiff systems     | `rodas5P.py`  |

Rodas5P also takes a **save hook**: a device function
`hook(save_idx, y, t, p_row, acc)` the kernel calls at every save time with the
dense-output state, accumulating into a per-trajectory `(n, hook_size)` output
row (`save_hook=`, `hook_size=`; `save_history=False` then keeps only the final
state). It lets a consumer of the history -- a line-of-sight integral, derived
per-save quantities -- run inside the launch instead of storing the history.
A hooked solve is a plain ensemble launch: no `jax.vmap`, no differentiation.
`tests/test_save_hook.py` pins it against the history.

Shared support modules:

- **`_numba_common.py`** — host-side helpers shared by both kernels: initial
  step selection, error weights, the shared kernel signature and its JAX launch
  (`ensemble_ffi_call`), and the `cuda.jit` coercion for user callbacks. There
  is no direct numba launch path: every solve, the benchmark scripts included,
  goes through `solve` and the XLA custom call.
- **`_codegen.py`** — `compile_device_source`, the one place generated device
  source is exec'd and registered with `linecache`. Both emitters below use it.
- **`_jax_numba_custom_call.py`** — the XLA FFI shim. Compiles a launcher,
  registers it as an FFI target, and exposes `ffi_abi_call` so a numba kernel
  becomes a JAX primitive.
- **`_jax_common.py`** — the JAX-facing glue: ensemble shape normalisation and
  `make_custom_vmap_solver`, whose `custom_vmap` rule lowers an outer
  `jax.vmap` over a single solve into one native ensemble launch.
- **`_sparsity.py`** — colours a sparsity pattern's column intersection graph
  and defines `CompressedJacobian`, the layout the Enzyme sweeps write into.
- **`_sparse_direct.py`** — orders a pattern with AMD, factorises it
  symbolically, and compiles a sparse LU and sparse triangular solves for it as
  `cuda.jit(device=True)` functions. The AMD ordering comes from `cvxopt`,
  which ships SuiteSparse's AMD in a wheel, so nothing here needs a system
  library; `ordering="natural"` skips the ordering entirely and those two are
  the whole of `ORDERINGS`. `scikit-sparse` supplied this until it was
  retired -- no wheels, and it compiles against SuiteSparse's headers -- and
  CHOLMOD's extra orderings went with it, having measured as AMD's own fill
  (`colamd`, `nesdis`, `best`) or worse (`metis`: 660 against 484 on the
  Einstein-Boltzmann-like pattern). `tests/test_sparse_direct.py` pins the fill
  AMD leaves against the values CHOLMOD gave.

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

`trajectories_per_block` is one thread's worth of work each, defaulting to a
warp. Nothing on chip bounds it, since every per-trajectory buffer is
thread-local; `trajectories_per_block_or_default` is where that decision is
made.

`sparsity` takes an `(n_vars, n_vars)` mask, a scipy sparse matrix, or an
`(nnz, 2)` index array. It compresses the Jacobian by colouring *and* selects a
compiled sparse direct linear solve; `ordering` picks its fill-reducing
permutation. See below.

### Sparsity, colouring, and the sparse direct solver

One kernel, one trajectory per CUDA thread: the state, the ten stage vectors,
`df/dt`, the step controller and the iteration matrix are all that thread's own
local memory. Nothing is shared and nothing synchronises inside a step.

**The Jacobian costs one sweep per colour, not one per column.** Forward-mode AD
returns `J v`, never `J`, so a column at a time costs `n_vars + 1` sweeps. But
two columns sharing no row are *structurally orthogonal* — their contributions
to `J v` cannot collide — so seeding both at once returns both intact. Colouring
the column intersection graph (`menodax/_sparsity.py`, NetworkX greedy, best of
five strategies) finds the fewest such groups. DISCO-EB's 50-variable
Einstein-Boltzmann system takes **12 colours**: 13 sweeps where a column at a
time takes 51.

`CompressedJacobian` is the result and the layout: entry `(r, c)` lives at
`r * n_colours + colour[c]`, rows dense and columns compressed. With no pattern
every column gets its own colour, `n_colours == n_vars`, and this *is* the dense
row-major matrix — so the dense path is not a special case, just the
uninformative end of the same mechanism.

**The pattern must be a superset of the true nonzeros.** Colouring a superset
only costs sweeps; colouring a subset silently corrupts the entries where two
columns of a group do overlap after all. `_check_orthogonal` rejects a colouring
that violates this, because it is the one way the scheme can be quietly wrong.
It need *not* cover the factorisation's fill-in, which gets slots of its own.

**`sparsity` is the whole interface.** There is no `linear_solver` argument and
no protocol to satisfy: the kernel builds the solver from the pattern itself.
With none it is `dense_lu_solver` over the dense colour grid; with one it is
`menodax/_sparse_direct.py`, which compiles a direct sparse LU and a pair of
sparse triangular solves for that exact structure — any pattern, no structure
assumed, which is the win a hand-written solver bought without the hand-written
solver. The two are the same `(factorize_local, solve_local)` shape, so the
kernel has no branch. Four host-side steps at kernel-build time:

1. **Order.** The symmetrised pattern goes to SuiteSparse's AMD, through
   cvxopt's wheel. AMD rather than COLAMD because COLAMD orders columns to
   bound fill under whatever row permutation *partial pivoting* chooses,
   and there is
   no pivoting here — the pattern is compiled in and cannot depend on the
   numbers. AMD's permutation is symmetric, so `I/(h*gamma)`'s diagonal stays on
   the diagonal and every pivot exists structurally. (CHOLMOD's `order="colamd"`
   runs AMD anyway on a symmetric analysis, so the two are not even distinct
   here.)
2. **Factorise symbolically.** Pattern-only Gaussian elimination gives the exact
   `L + U`, fill included. Exact, and value-independent: a sample factorisation
   would drop an entry wherever those particular numbers cancelled, and the
   buffer would then be short by one slot in a kernel with no way to say so.
3. **Lay out.** One CSR image of `L + U` — CSR because all three routines read it
   *by rows*: the up-looking factorisation, the forward substitution and the back
   substitution. `L` strictly left of the diagonal and `U` from it rightwards,
   sharing the buffer, since the factorisation is in place and a unit diagonal
   needs no storage. `nnz(L + U)` is the whole footprint.
4. **Compile.** Two `cuda.jit(device=True)` functions. Where the structure fits
   they are emitted as straight-line code with every slot a *literal*; above
   `MAX_UNROLLED_SUBSTITUTIONS` / `MAX_UNROLLED_UPDATES` they fall back to loops
   over index tables in constant memory. Either way the factorisation's inner
   merge — which slot of row `i` each update from row `k` lands in, the part a
   runtime sparse solver spends its time searching for — is resolved on the
   host.

The unrolling is where most of the speed is, and for a reason worth knowing:
table-driven, a sparse routine spends a broadcast load on the index of every
value before it can issue the load of the value itself, and that dependent pair
is only free when there are enough other trajectories in flight to cover it.
DISCO-EB's single-cosmology case is 128 trajectories — four warps on a 46-SM
device — and nothing covers it. Unrolling costs no registers either, since the
kernel indexes both the matrix and the right-hand side with loop variables of
its own and so neither can leave local memory whatever this does. Measured at
N128: **528 ms** table-driven, **419 ms** with the solves unrolled, **398 ms**
with the factorisation unrolled too, against **509 ms** for the hand-written
Schur solver. `tests/test_sparse_direct.py` runs the two emissions against each
other and requires them bit-identical.

No pivoting is sound here for the same reason the Jacobian may be approximate:
Rodas5P is a Rosenbrock-**W** method, so order 5 survives, and a pivot that
reaches zero leaves an infinity, the error norm goes to NaN, the step is
rejected, and the smaller step puts a larger `1/(h*gamma)` on that very diagonal.

It also decouples colouring from storage, which a hand-written solver could not.
Such a solver owned its buffer, so it had to declare its own fill-in in the
pattern to have somewhere to put it, and paid colours for that — DISCO-EB's 12
rather than 11. Here the fill has slots of its own by construction, so the
pattern may be as tight as it really is; `CompressedJacobian.n_slots` is what
lets the buffer be larger than the slots any entry claims.

On DISCO-EB's 50-variable Einstein-Boltzmann Jacobian AMD returns a *perfect*
elimination order — zero fill, `nnz(L + U) == nnz(J) == 238`, one slot per
entry — and the order it finds is the hand-written Schur solver's: peel each
free-streaming hierarchy from its truncated end inwards, where every variable has
degree two, then eliminate the dense core last. It replaced that solver and is
22% faster than it. The README's "Sparse systems" section carries the full
design argument — why AMD and not COLAMD, why no pivoting, why symbolic, why
CSR, why unrolled — and is what to read before touching any of it.

**A negative value in a constant index table is fatal, and silent.** An `int32`
read out of a `cuda.const` array promotes as though it were *unsigned* once it
enters arithmetic: with `t[i] == -1`, `t[i] + 4` evaluates to `2 ** 32 + 3`, so
an index built that way addresses nothing in particular and nothing complains.
It cost an afternoon here, where a destination offset was stored relative to a
source slot and came out negative for the first few rows; the factorisation was
wrong, every step was rejected, and the solve returned its initial state.
`_pack` now rejects a table with any negative entry, and every table the module
builds is an offset, so non-negativity is the natural form anyway. Casting with
`np.int64(...)` first is the other way out.

Forward sensitivities work with either solver: the joint iteration matrix is
block lower triangular with the same `M0` on every diagonal block, so only the
`n_vars` block is ever factorised, and the coupling between blocks is a forward
substitution the kernel does itself.

The colouring and the storage are separate questions, which is what lets the
pattern be declared as tightly as it really is. A solver owning its own buffer
had to declare its fill-in in the pattern to have somewhere to put it, and paid
colours for that — DISCO-EB's 12 rather than 11.

Things to know when touching this:

- **`ode_fn` is always the tuple form.** There is no `jac_fn` any more; the
  Jacobian and `df/dt` both come from Enzyme, and the tuple form is what Enzyme
  reads.
- **A runtime index into the tuple is fatal.** numba emits a bounds-check
  `cmpxchg` for it and Enzyme refuses the atomic ("cannot handle unknown
  instruction"). Wrapping the loops around a thread-local array instead gets
  past typing but yields IR NVVM will not verify. Callbacks must therefore index
  only with constants — which is why the reference systems are AST-generated,
  and why DISCO-EB generates its unrolled right-hand side from the looped one.
- **The seed table's zero row doubles as the null parameter direction**, so it
  is widened to `max(n_vars, n_params)`. It was `n_vars` once, and every `df/dt`
  silently read past the row whenever there were more parameters than states.
- **The sensitivity rows have their own `df/dt`**
  (`write_sensitivity_time_derivative`). Dropping it does not merely lose order:
  those rows of `dT` are thread-local and hold whatever was there before, so
  every step is rejected and the solve silently returns its initial state.
- `tf_index` names a `params` column holding each trajectory's own end time;
  `max_registers` caps the per-thread register count.

### Writing ODE callbacks

Callbacks are compiled with `numba_cuda_mlir`, which constrains them:

- Take and return **fixed-size tuples of scalars**, not arrays.
- Use `math`, not `numpy`/`jax.numpy`.
- **Device code cannot call a plain Python helper.** Either inline the shared
  work or pre-decorate the helper with `@cuda.jit(device=True)`.
- Closed-over arrays land in CUDA **constant memory** (64 KiB per module), and
  numba emits one copy *per reference site*. Pack related tables into a single
  array and bind it to a local before indexing — see `make_mode_ode` in
  `examples/mukhanov_sasaki/main.py`.

A callback that also has to run under `jax` — every example here has a Diffrax
baseline to compare against — is written **once** and built twice, by
`examples/dual_backend.py`: the body is a factory over the handful of names
the two backends spell differently (`math.exp`/`jnp.exp`, `max`/`jnp.maximum`,
…), and `build_rhs` returns it as `.device` (the tuple form), `.jax` (arrays)
and `.host` (the device arithmetic in plain Python, which is how
`tests/test_examples.py` compares the two without a GPU). What such a body
must avoid is an `if` on a value that varies, since a traced one cannot be
branched on at all: `maximum`/`minimum`, or a branchless
`a + (b - a) * (x > c)`, serve both. A helper it shares with the rest of the
module goes through `build_fn`, whose `.device` member is a
`cuda.jit(device=True)` function — Enzyme inlines straight through it.

### Derived Jacobians

`rodas5P` takes no `jac_fn`. `_make_kernel` forward-differentiates `ode_fn`
with [numba-enzyme][ne], as it stands and with no adapter around it: a
callback of the documented shape already reaches Enzyme as a function of flat
scalars returning a struct, because numba-cuda-mlir flattens a tuple argument
into one scalar parameter per element and lowers a tuple return to a struct
returned by value. The kernel uses `jvp` — a directional derivative — because a
seed need not be a unit vector: a whole colour group goes in at once and the
sparsity pattern says which output component belongs to which column (see
"Sparsity, colouring, and custom linear solvers"). Seeding the time argument
instead of the state gives the whole `df/dt`, so
`n_colours + 1` sweeps supply both matrices the kernel needs — `n_vars + 1` when
there is no pattern to exploit. numba-enzyme also exposes `jacfwd`, which fills
the whole matrix; that would put `n_vars ** 2` doubles in per-thread local
memory, where one group at a time keeps the working set at `O(n_vars)` and folds
straight into the LU buffer.

**The seed rows are literals, several to a call.** The derivative links as LTO
IR and nvJitLink inlines it into the kernel before constant propagation, so a
seed the compiler can *see* folds: the zero components kill their tangent
arithmetic and a colour sweep collapses to its own group's columns. The same
seed read through a loop variable arrives in registers and cannot fold, and
that version of this kernel sat at the 168-register cap with a 13 KB spill
frame and cost 114 ms of DISCO-EB's 128 ms regression against a hand-written
Jacobian (see numba-enzyme's
`test_a_compile_time_jvp_direction_folds_and_is_faster`).
`_make_literal_seed_jacobian` therefore generates the writer with the colour
loop unrolled: one `jvp` call per `SEED_BATCH` colours, each seed row a
literal index into the constant-memory table, and one literal store per entry
the colour holds. Batching is what recovers the rest: the sweeps in one call
share one Enzyme entry, so once inlined the primal work they have in common --
for a right-hand side whose coefficients depend on `t` alone, all of it -- is
one computation for LLVM to CSE rather than one per sweep. Measured at N128:
665.9 ms with the loop, 594.7 ms with literal seeds, 508.8 ms with eight seeds
per call, against 552 ms for the hand-written Jacobian the sweeps replaced.
`array_rhs` is the last piece: the primal stage evaluations do not need the
tuple form (that exists for Enzyme), so a caller with an `f(y, t, p, out)`
device function hands it in and the eight evaluations per step call it
directly, worth 15 ms on DISCO-EB.

Enzyme's `opt` pipeline is `-passes=enzyme,adce,globaldce,instnamer`.
`instcombine` used to sit in it and had to come out: on a large kernel it
canonicalises a clamp into `llvm.smax.i64`, and the vendored LLVM's NVPTX path
emits a `.smax` token ptxas rejects. The input IR is clean; the intrinsic is
introduced by the pass. Small systems never hit it, so this only showed up at
DISCO-EB's 50 variables.

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
  `jvp` call inside `_make_kernel`.
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
  (see `make_cuda_local_vector_writer`); only the signature says the
  derivative wants the tuple form.
- The five-argument call inlines, so the kernel's PTX is byte-identical across
  processes and the CUDA JIT cache hits. Spelling the state out as one scalar
  per component instead makes the call a star call at `n_vars=48`, which
  numba-cuda-mlir will not inline, leaving a wrapper in the kernel named after
  an `id()` that changes every process — that cost 13.0 s a warm compile
  against 10.1 s here.

The dependency is `numba-enzyme-cuda`, the fork published under its own name
because upstream's PyPI release has no CUDA backend — see `wheels/README.md`,
which lists every local change made to numba-enzyme and how a new one is cut.

[ne]: https://github.com/Qruise-ai/numba-enzyme

`lu_precision` (`"fp32"`/`"fp64"`) selects the LU precision for implicit
solvers. The `"fp32"` default does not lower the method's order — the
Rosenbrock order conditions hold under an approximate Jacobian — while
halving the LU buffer's footprint.

### Forward sensitivities

Both solvers carry a `jax.custom_jvp` rule (`menodax/_sensitivity.py`), so
`jax.jvp`, `jax.jacfwd`, `jax.grad`, `jax.jacrev` and `jax.value_and_grad` work
with respect to `y0` and `params`. Asking for a derivative integrates the
continuous forward-sensitivity system jointly with the state,

```
d/dt [y, S] = [f(t, y, p), J_y S + J_p]
```

as one system of `n_aug = n_vars * (1 + n_sens)` components, so `value_and_grad`
is one solve rather than two. `t_span` is not differentiable and raises. The
README's "Gradients" section carries the design argument and the proof that a
stiff ODE has an equally stiff sensitivity system; this is what to know before
touching the code.

- **The rule sits outside `custom_vmap`, not inside.** `custom_vmap`'s own JVP
  path traces to a jaxpr, which instantiates every symbolic zero; the rule uses
  `symbolic_zeros=True` to see *which* arguments are being differentiated and
  integrate only those blocks. Nest it the other way and every solve carries
  the `n_vars` initial-state columns whether or not anyone wants them.
- **`jax.grad` needs no adjoint.** The rule materialises `S` and contracts it
  with the input tangents; that contraction is linear in the tangents, so JAX
  transposes it and reverse mode falls out of the same rule.
- **Everything is a directional derivative, never a Jacobian.** `J_y S_k + J_p_k`
  is `jvp` of `ode_fn` seeded with `(S_k, 0, e_k)` — one sweep per column at any
  `n_vars`, where unit columns would cost `n_vars + 1`. The joint Jacobian's
  coupling block applied to a vector, and the sensitivity rows' `dF/dt`, are
  `jvp(jvp(ode_fn))` (forward over forward) with the second direction set to the
  state increment or to the time direction. Composing makes the derivative of
  the whole tangent map, so the call carries a fourth direction for the inner
  one's own variation; the solver passes zero for it, which leaves the plain
  bilinear form. Both the tuple-shaped `jvp` and its composability are
  menodax-driven additions to the numba-enzyme fork, where every endpoint now
  composes over a `jvp`; see `wheels/README.md`.
- **The unit and zero directions are windows into one constant-memory table**,
  `seed_table`: `2L` zeros with a single `1.0` at `L`, so the window starting at
  `L - k` has its `1.0` at `k`, and any window inside `[0, L)` is all zeros. A
  per-thread one-hot buffer instead could not promote to registers, because its
  store index is dynamic.
- **Rodas5P factorises the state block only.** The joint iteration matrix is
  block lower triangular with the same `M0 = I/(h*gamma) - J_y` on every
  diagonal block, so `block_solve` is a forward substitution against one
  factorisation: state row first, then each sensitivity row with its right-hand
  side corrected by `L_k k_y`. The LU is `n_vars`-sized, not `n_aug`-sized.
  Do not be tempted back to the block-diagonal approximation: it is legitimate
  under the W property and order 5 survives, but the error constant does not —
  201x the steps on a bilinear right-hand side, measured, and
  `test_joint_solve_costs_about_what_the_plain_solve_costs` is the guard.
- **The batch no longer has to be re-sized for a joint solve.** The stage
  vectors that hold the augmented state are thread-local, so only the
  `n_sens` costs local memory rather than the block's trajectory count.
- **Step control includes the sensitivities by default.** `sens_error_control`
  flips it. With it off, the sensitivity components get zero error weight and
  `n_error` keeps the norm dividing by `n_vars`, so the joint solve takes the
  plain solve's steps and returns its value — Tsit5 bit for bit, Rodas5P to
  round-off.
- **The writers are where the augmentation lives**, not a generated augmented
  callback: the kernels are parameterised on a `SensitivitySpec` and otherwise
  integrate the larger system unchanged. Both writers now own a whole trajectory
  per thread, so each writes its own local arrays with no barrier inside a
  device function its callers invoke under a divergent `if running`.

### Kernel design

- One CUDA thread per trajectory; per-trajectory adaptive stepping, so lanes in
  a warp diverge and the block runs until its slowest trajectory finishes.
  Rodas5P needs no barrier at all: a finished thread simply returns.
- In-kernel LU factorisation, entirely in thread-local memory.
- `tsit5` has `backend="shared"` and `backend="global"` paths, selected by
  whether the state fits in shared memory; the two are bit-identical.

### Reference and benchmarks

- `reference/systems/python/` — test systems (Lorenz, VdP, Robertson,
  Brusselator, Bateman, Kaps, heat), each exposing `ode_fn` in
  numba-compatible tuple form plus `make_scenario` for ensembles. Nothing
  consumes a hand-written Jacobian any more, so the systems no longer carry
  one; `tests/test_enzyme_jacobian.py` checks the Enzyme-derived Jacobian
  against JAX's forward-mode AD of the same `ode_fn`.
  `_tuple_codegen.py` builds these callbacks for parameterised dimensions from
  one expression string per component, so every index is a literal.
- `reference/solvers/python/` — Diffrax, scipy and Julia (DiffEqGPU) baselines.
- `examples/` — worked problems, each with its own README.
