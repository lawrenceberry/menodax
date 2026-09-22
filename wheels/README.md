# The numba-enzyme-cuda wheel

`rodas5P` derives its Jacobian with [numba-enzyme][ne]. The release upstream
publishes on PyPI has no CUDA backend, so the dependency is
[**numba-enzyme-cuda**][cuda], the [`cuda` branch of a fork][fork] published
under its own distribution name:

```toml
dependencies = ["numba-enzyme-cuda>=0.2.1"]
```

An ordinary dependency, with no `[tool.uv.sources]` entry behind it. It used to
be one: the fork was a wheel attached to a GitHub release, and
`pyproject.toml` pinned that URL. That could never work for anyone installing
*this* package, because PyPI rejects a distribution whose metadata carries a
direct URL dependency — a uv source is the consuming project's own mechanism
and is not inherited — so every consumer would have had to write the same pin
by hand, and `pip install modax-solvers` would have resolved upstream's CUDA-less
wheel and failed at the first differentiation.

It provides the `numba_enzyme` import package, so it is a drop-in replacement
and **upstream must not be installed alongside it** — the two would fight over
the same directory.

The wheel is self-contained. The derivative pipeline shells out to `clang`,
`llvm-link` and `opt` from LLVM 15 and loads the standalone Enzyme plugin, none
of which are in numba-enzyme's source tree, so all of them ship inside the
wheel under `numba_enzyme/_vendor/` — 237 MB installed, 73.5 MB compressed, and
PyPI's per-file limit is 100 MB. No system LLVM is involved, and
`toolchain.py` resolves `_vendor/` ahead of `PATH`. It is tagged
`py3-none-manylinux_2_38_x86_64`: no CPython tag, because the package has no
extension modules and so installs on any Python ≥ 3.11, and 2.38 because that
is the highest `GLIBC_` symbol version any of the nineteen ELF files under
`_vendor/` references, which is what auditwheel would derive from the contents.
A bare `linux_x86_64`, which the release asset carried, PyPI refuses outright.

[ne]: https://github.com/Qruise-ai/numba-enzyme
[cuda]: https://pypi.org/project/numba-enzyme-cuda/
[fork]: https://github.com/lawrenceberry/numba-enzyme/tree/cuda

## Installing from the branch instead

```toml
[tool.uv.sources]
numba-enzyme-cuda = { git = "https://github.com/lawrenceberry/numba-enzyme", branch = "cuda" }
```

Worth it when tracking unreleased fork changes matters more than resolve time.
It is equally self-contained: when `src/numba_enzyme/_vendor/` is absent —
which it is for any build that is not a cibuildwheel run — the fork's
`hatch_build.py` downloads upstream's released PyPI wheel and restages the LLVM
and Enzyme binaries it already carries, so the branch builds into the same
wheel the release does. Before that hook existed, a git install produced a
196 KB package that imported cleanly and then failed at the first
differentiation. A git source re-downloads and re-stages ~73 MB on every fresh
resolve, which is the reason it is not the default; set
`NUMBA_ENZYME_VENDOR_FROM_PYPI=0` to suppress the staging deliberately.

> **Do not symlink `site-packages/numba_enzyme` at the fork's working tree.**
> It is a tempting way to iterate on the fork without reinstalling, and the
> next `uv sync` that replaces the package deletes *through* the link: it
> empties the real `src/numba_enzyme/`, taking the untracked 237 MB
> `_vendor/` with it, and only then fails on `rmdir` with "Not a directory".
> Tracked files come back with `git checkout`, and `_vendor/` can be
> unzipped out of any built wheel, but neither is a step you want to
> discover mid-sync. Point `[tool.uv.sources]` at a local path instead and
> let uv own the directory.

## Cutting a new release

Bump `version` in the fork's `pyproject.toml`, then tag it:

```bash
cd ../numba-enzyme
git tag -a v0.2.2 -m "numba-enzyme-cuda 0.2.2" && git push origin cuda v0.2.2
```

`.github/workflows/release.yml` does the rest: `uv build` (whose hook stages
`_vendor/` from PyPI if absent), `twine check --strict`, an install of the
built wheel into a throwaway environment that imports it and the five
endpoints, a check that the vendored `clang` and `opt` still run out of it, and
a trusted-publishing upload — no API token, and no Docker, since the binaries
come from upstream's wheel either way. `wheels.yml` still builds the toolchain
from source in the manylinux image, but is `workflow_dispatch`-only: two
workflows uploading one version would race.

That install-and-import step exists because 0.2.0 shipped **unimportable**.
`__init__` read its own version from `importlib.metadata` under
`"numba-enzyme"`, the name the fork had just stopped using, so the first line
of `import numba_enzyme` raised `PackageNotFoundError`. Unpacking the wheel and
running the binaries — which was the whole check — never touches Python's view
of the package.

Then point this repository at it:

```bash
uv lock --refresh    # --refresh-package is not enough; the index listing is cached too
uv pip install --no-deps "numba-enzyme-cuda==0.2.2"   # or a full uv sync
```

The build leaves `src/numba_enzyme/_vendor/` behind in the fork, 237 MB that
its `.gitignore` covers, so `git status` stays clean. Leave it: the next build
reuses it instead of re-downloading. Ignoring it is safe because
`hatch_build.py` force-includes each file, which bypasses VCS ignore rules,
and restages the toolchain whenever the directory is absent — a wheel built
from an sdist that excludes `_vendor/` still comes out complete.

## Local changes to numba-enzyme

The fork carries changes that are not upstream:

- **tuple-returning primals** — a CUDA primal with several outputs returns a
  homogeneous tuple. numba-cuda-mlir lowers that to an LLVM struct returned by
  value; forward modes take Enzyme's tangent struct directly, and reverse modes
  differentiate an internal `sum_k w_k * f_k(x)` with the weights inactive,
  since Enzyme rejects an aggregate differential return. Nothing is staged
  through an output array, so the solver hands the derivative its own callback.
- **tuple *arguments*** — a primal may also take homogeneous tuples, and every
  derivative call then mirrors its argument list, each tuple supplied as a
  contiguous array whose elements the entry point loads before the Enzyme
  marker. That is what lets the solver differentiate `ode_fn` as written, with
  no adapter, and call the result with a fixed five arguments at any `n_vars`.
  Such a primal needs an explicit `signature`, since an array cannot say how
  long the tuple it stands for is.
- **qualname-mangled primal symbol** — `lower_cuda` derives the primal's symbol
  the way numba-cuda-mlir does, from `__qualname__` rather than `__name__`.
  They coincide only for module-level functions, so a nested or generated
  callback was previously looked up under a symbol the module never defined.
- **`jacfwd`** — forward-mode Jacobian of a tuple-returning primal. `jvp`
  differentiates a scalar-output primal, so a sweep yields a single Jacobian
  entry; a sweep of a multi-output one yields a whole column, and `jacfwd`
  fills the whole matrix one sweep per column. modax does not use it: the
  matrix would have to live in per-thread local memory. See `_make_kernel` in
  `modax/rodas5P.py`.
- **`jvp` for tuple-returning primals** — `jvp` used to be scalar-output only,
  so a directional derivative of a vector field had to be assembled from
  `n_vars + 1` unit columns. It now also takes the array call shape,
  `(tangent, *args, *directions)`, with each direction mirroring the primal's
  own argument list: an array where the primal takes a tuple, a scalar where it
  takes a scalar. One sweep returns the whole `J @ d`, which is what makes the
  forward-sensitivity right-hand side cost one sweep per column at any
  `n_vars`. See "Forward sensitivities" in `AGENTS.md`.
- **`jvp` and `vjp` take several directions at once** — a tuple-returning
  primal's `jvp` accepts one mirrored direction set per sweep and writes a
  matrix when there is more than one; `vjp` takes a matrix of cotangents and
  loops over its rows. `jacfwd` and `jacrev` are then those same loops with the
  identity supplied internally rather than read from the caller, which is what
  removed their bespoke index arithmetic.
- **every endpoint composes** — a CUDA derivative is a valid primal, so
  `jvp(jvp(f))`, `jacrev(jvp(f))`, `vjp(jvp(f))` and the rest all work. It
  cannot work by re-differentiating the result of the first call: that is a
  `cuda.declare_device` handle to a separately compiled LTO IR blob, and Enzyme
  differentiates definitions, not declarations. So the chain is *recorded* --
  `_differentiate` walks back to the base primal and raises the depth -- and
  every level is emitted as a definition. Forward markers nest happily in one
  Enzyme pass; a *reverse* endpoint over a forward level does not, because
  Enzyme preprocesses a callee before resolving a marker inside it, so those
  builds run Enzyme once per stage, feeding each output into the next link.
- **reverse-mode multi-output APIs** — `vjp` and `jacrev`, the reverse
  counterparts of `jvp` and `jacfwd`. modax does not use them; see "Derived
  Jacobians" in `AGENTS.md` for why the solver is forward-mode.
- **a compile-time direction folds, and the single-column and single-row
  endpoints are gone with it** — the fork briefly carried `jacfwd_column` and
  `jacrev_row`, which took a run-time index and built the unit seed inside the
  derivative. They were removed: the derivative links as LTO IR, so nvJitLink
  inlines it before constant propagation, and a unit direction written out at
  the call site folds to exactly the column that index would have selected, at
  the same cost. A unit seed through `jvp` is therefore the column and a unit
  cotangent through `vjp` is the row. This is the same folding the kernel's
  colour seeds rely on; numba-enzyme's
  `test_a_compile_time_jvp_direction_folds_and_is_faster` is the guard.
- **`CUDADifferentiable.externals`** — the `cuda.declare_device` handle behind
  each tuple implementation, reached through `differentiate_cuda`.
- **optional `signature`** — CUDA derivatives now specialise lazily at each
  call site; passing `signature` only constrains that. A call of more than 30
  positional arguments, which CPython compiles as a star call that numba's
  inliner rejects, is compiled as a separate function instead of inlined, and
  nvJitLink's LTO recovers the cost: modax's derivative call at `n_vars=48`
  carries 53 arguments and solves no slower than the earlier raw extern.
- **LTO IR instead of PTX** — the derivative is emitted as NVVM LTO IR, which
  makes numba-cuda-mlir compile the calling kernel to LTO IR too, so nvJitLink
  inlines the derivative rather than leaving an opaque call carrying a
  parameter per primal argument. That inlining is what lets the caller's column
  buffers live in registers instead of local memory.
- **selective entry points** — `synthesise_cuda`/`build_cuda`/`differentiate_cuda`
  take a `modes` argument, and each public entry point requests only its own.
  Every emitted entry point carries its own Enzyme marker call, so building
  `grad` for an n-argument primal costs n reverse differentiations whether or
  not anything calls them.
- **closure-aware derivative cache** — the cache key now includes the primal's
  lowered IR. It previously keyed on source text and qualified name, which are
  blind to what a device function closes over: two identically-written
  components wrapping different callees silently shared one derivative.
- **internalised primal** — the primal is given internal linkage after linking,
  so the derivative PTX no longer exports its mangled name. Two derivatives of
  same-shaped primals would otherwise define the same symbol and collide in a
  kernel that links both.
- **libNVVM sanitiser fixes** — Enzyme emits `fneg`, and fast-math flags on
  `select` and `phi`; all three postdate the LLVM 7 textual IR reader libNVVM
  uses, which stops at the opcode. Without this, any ODE whose derivative
  involves a negation or a guarded singularity fails to compile.
- **relaxed `llvmlite`/`numba` pins**, and a guard around the removed
  `llvmlite.binding.initialize()`. Upstream pins `llvmlite==0.44.0` and
  `numba==0.61.2` for its CPU driver; that would force numpy below 2.3 here.
- **self-contained git installs** — `hatch_build.py` stages the LLVM/Enzyme
  binaries from the released PyPI wheel when `_vendor/` is absent, and tags the
  wheel `py3-none-linux_x86_64`. See "Installing from the branch instead"
  above. `_vendor/` is gitignored again now that the hook, rather than the
  un-ignore, is what keeps the plugin in the wheel.
