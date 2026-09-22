# Kernel design

One CUDA thread per trajectory, and per-trajectory adaptive stepping. Lanes in
a warp diverge as their step sizes do, and a block runs until its slowest
trajectory finishes; Rodas5P needs no barrier at all, since a finished thread
simply returns. The state, the stage vectors, `df/dt`, the step controller and
the iteration matrix are all that thread's own local memory. Nothing is shared
and nothing synchronises inside a step.

That is why `trajectories_per_block` is free to be anything — nothing on chip
bounds it — and why `trajectories_per_block_or_default` defaults it to a warp.

`tsit5` keeps two paths, `backend="shared"` and `backend="global"`, selected by
whether the stage workspace fits in shared memory. They are bit-identical.

## How a solve reaches the GPU

There is no direct numba launch path: every solve goes through `solve` and the
XLA custom call.

1. `solve` normalises the ensemble shapes and builds (or fetches from cache)
   the kernel for this `ode_fn`, dimension and set of options.
2. `solvers._jax_numba_custom_call` compiles a launcher for that kernel,
   registers it as an XLA FFI target, and exposes it through `ffi_abi_call`.
3. `solvers._numba_common.ensemble_ffi_call` makes the call, with the state and
   stage buffers transposed to `(n_vars, N)` so a warp's accesses are
   coalesced. XLA materialises that transpose.
4. `solvers._jax_common.make_custom_vmap_solver` wraps the result, so an outer
   `jax.vmap` over a single solve lowers to one native ensemble launch rather
   than to a batched trace.

Because the whole thing is a `jax.ffi.ffi_call`, a solve is `jit`-traceable and
usable inside `lax.scan` and `vmap` — `examples/bbn_estimation` calls one from
a BlackJAX nested-sampling likelihood.

## The derived Jacobian

Rodas5P takes no `jac_fn`. `_make_kernel` forward-differentiates `ode_fn` with
[numba-enzyme](https://github.com/Qruise-ai/numba-enzyme) as it stands, with no
adapter around it, and uses `jvp` — a directional derivative — rather than a
full `jacfwd`: a seed need not be a unit vector, so a whole colour group goes in
at once and the sparsity pattern says which output component belongs to which
column. Seeding the time argument instead of the state gives the whole `df/dt`,
so `n_colours + 1` sweeps supply both matrices the kernel needs, and
`n_vars + 1` when there is no pattern to exploit. A full `jacfwd` would put
`n_vars ** 2` doubles in per-thread local memory, where one group at a time
keeps the working set at `O(n_vars)` and folds straight into the LU buffer.

Two details carry most of the performance:

- **The seed rows are literals.** The derivative links as NVVM LTO IR and
  nvJitLink inlines it into the kernel before constant propagation, so a seed
  the compiler can *see* folds: the zero components kill their tangent
  arithmetic and a colour sweep collapses to its own group's columns. The same
  seed read through a loop variable arrives in registers and cannot fold — that
  version sat at the 168-register cap with a 13 KB spill frame.
- **Several seeds share a call.** The sweeps in one `jvp` call share one Enzyme
  entry, so once inlined the primal work they have in common is one computation
  for LLVM to CSE rather than one per sweep. Measured on DISCO-EB at N128:
  665.9 ms with the colour loop, 594.7 ms with literal seeds, 508.8 ms with
  eight seeds per call, against 552 ms for the hand-written Jacobian the sweeps
  replaced.

Forward mode is what makes a sweep worth a whole column; a sweep of a
scalar-output primal yields one Jacobian *entry*. Reverse mode reaches a whole
row per sweep by slicing the right-hand side into scalar components, which was
measured to win below roughly 16 state variables — but it needs `n_vars`
primals and `n_vars` Enzyme differentiations against forward's one apiece, and
`O(n_vars ** 2)` generated device source against `O(n_vars)`. A cold first
solve at 96 state variables took 171 s that way against 49 s this way.

## Caches

Kernel construction is `functools.cache`d on the callback and its options, and
numba-enzyme keys its derivative cache on the primal's lowered IR. Because the
five-argument derivative call inlines, the kernel's PTX is byte-identical
across processes and the CUDA JIT cache hits. `clear_caches()` in each solver
module drops the host-side caches.
