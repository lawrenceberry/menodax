# Local numba-enzyme wheel

`rodas5P` derives its Jacobian with [numba-enzyme][ne]. The release on PyPI has
no CUDA backend, so `pyproject.toml` points `[tool.uv.sources]` at a wheel built
from a local checkout instead. The wheel is ~73 MB and is not committed
(`wheels/` is in `.gitignore`), so a fresh clone has to rebuild it before
`uv sync` will succeed.

[ne]: https://github.com/Qruise-ai/numba-enzyme

## Rebuilding

The wheel must be self-contained: the derivative pipeline shells out to
`clang`, `llvm-link` and `opt` from LLVM 15 plus the standalone Enzyme plugin,
none of which are in the source tree. numba-enzyme's own build populates
`src/numba_enzyme/_vendor/` from a staging directory under cibuildwheel; the
released PyPI wheel already carries those binaries, and
`packaging/bootstrap_dev_toolchain.py` downloads them.

From a checkout of numba-enzyme at `../numba-enzyme`:

```bash
cd ../numba-enzyme
uv run python packaging/bootstrap_dev_toolchain.py   # once; fills .dev-toolchain/

# Stage the binaries where the wheel build looks for them. The shared libraries
# go under _vendor/lib because that is the first entry in the binaries' RPATH
# ($ORIGIN/../lib); the released wheel instead resolves them through the second
# entry, a top-level numba_enzyme.libs/ that hatchling would not include.
mkdir -p src/numba_enzyme/_vendor
cp -r .dev-toolchain/wheel/numba_enzyme/_vendor/. src/numba_enzyme/_vendor/
mkdir -p src/numba_enzyme/_vendor/lib
cp -r .dev-toolchain/wheel/numba_enzyme.libs/. src/numba_enzyme/_vendor/lib/

uv build --wheel
cp dist/numba_enzyme-*-linux_x86_64.whl ../modax/wheels/
```

Then, in this repository:

```bash
uv lock --upgrade-package numba-enzyme   # the lock pins the wheel's sha256
uv sync --extra cuda13
```

`_vendor/` is 237 MB and is un-ignored in numba-enzyme's `.gitignore`, so delete
it afterwards if you do not want it in `git status` there.

## Local changes to numba-enzyme

The checkout this wheel is built from carries changes that are not upstream:

- **`jacfwd`** — forward-mode Jacobian of a *vector-valued* primal, one that
  writes its outputs through a leading array argument. `jvp` differentiates a
  scalar-output primal, so a sweep yields a single Jacobian entry; a sweep of a
  vector-valued one yields a whole column. It emits the whole matrix by
  default, or a single column chosen by a run-time index with `column=True`.
  The solver uses the column shape: the whole matrix would have to live in
  per-thread local memory. See `solvers/_enzyme_jacobian.py`.
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
