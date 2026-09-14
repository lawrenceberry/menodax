"""Enzyme-derived Jacobians for the implicit solvers.

The implicit kernels need ``df/dy`` (and, for non-autonomous systems,
``df/dt``) of a user ``ode_fn`` written in the numba tuple form described in
``CLAUDE.md``. Rather than making the caller hand-write those, this module
differentiates ``ode_fn`` with `numba-enzyme
<https://github.com/Qruise-ai/numba-enzyme>`_, which runs Enzyme over the
device function's LLVM IR and hands back a device-callable derivative.

The right-hand side is compiled once into a flat-argument adapter that returns
the callback's tuple unchanged. Enzyme's CUDA backend differentiates functions
of flat scalar arguments, and Numba-CUDA-MLIR lowers a tuple return to an LLVM
struct returned by value, which Enzyme differentiates directly — so the adapter
needs no output array. Forward-differentiating it and seeding a unit vector
gives a whole Jacobian column per sweep; seeding the time argument instead
gives the whole ``df/dt`` vector, so ``n_vars + 1`` sweeps supply both matrices
the kernel needs. The seed is built inside the derivative from a column index,
so nothing here materialises a tangent vector.

numba-enzyme's ``jacfwd`` would return the whole ``n_vars`` by ``n_vars + 1 +
n_params`` matrix at once. The kernel uses ``jacfwd_column`` instead, because
that matrix would have to live in per-thread local memory: free below about 16
state variables, then ``n ** 2`` doubles a thread above it. One column at a
time keeps the working set at ``O(n_vars)`` by construction, and lets each
column fold into the shared LU buffer as it arrives rather than being staged.

Forward mode is what makes one sweep worth a whole column. A sweep of a
*scalar*-output primal — the shape Enzyme's CUDA backend otherwise requires —
yields a single directional derivative, so it would take one sweep per Jacobian
entry. Reverse mode reaches a whole row per sweep instead, by slicing the
right-hand side into scalar components, and for a sparse right-hand side those
components shrink under dead-code elimination to something much cheaper than a
full evaluation. That was measured to win the solve below roughly 16 state
variables, but it needs ``n_vars`` primals and ``n_vars`` Enzyme
differentiations against forward's one apiece, and ``O(n_vars ** 2)`` generated
device source against ``O(n_vars)``: a cold first solve at 96 state variables
took 171 s that way against 49 s this way, and the gap widens with dimension.

numba-enzyme emits the derivative as NVVM LTO IR, so nvJitLink inlines it into
the kernel rather than leaving a call with a parameter per primal argument.
That is why ``dout`` costs no local memory: inlined, it promotes to registers.

The derivative is built eagerly, through ``differentiate_cuda``, and the kernel
calls its external declaration rather than the lazy ``jacfwd_column``
placeholder. The public transform would resolve the same derivative at call-site
typing time, but it resolves to a wrapper, and this call carries more than 30
arguments — a star call, which Numba-CUDA-MLIR's inliner refuses, so the wrapper
survives into the kernel as its own device function named after an ``id()`` that
changes every process. That makes the kernel's device code differ in every run
and miss the CUDA JIT cache; calling the external directly deletes the wrapper,
and its symbol comes from the derivative's cache key. Warm compile at
``n_vars=48`` is 10.1 s that way against 13.0 s through the placeholder. Nothing
is lost by resolving early: everything the placeholder exists to infer — the
shape, the mode, the primal signature — is fixed here already.

The parameter partials are never seeded, so they cost nothing. Seeding
direction ``n_vars + 1 + j`` instead would give ``df/dp_j`` as a whole column,
which is what a forward sensitivity analysis of the trajectory would need.
"""

from __future__ import annotations

import functools

from numba_cuda_mlir import cuda, types
from numba_enzyme.cuda import differentiate_cuda

from solvers._numba_common import as_cuda_device


def _adapter_source(n_vars: int, n_params: int) -> str:
    """Source for the flat-argument form of the ODE right-hand side.

    Enzyme's arguments have to be flat scalars, so the state and parameter
    tuples the callback expects are rebuilt inside the body. Its tuple is
    returned unchanged.
    """
    args = [f"y{j}" for j in range(n_vars)] + ["t"] + [f"p{j}" for j in range(n_params)]
    y_tuple = "({},)".format(", ".join(f"y{j}" for j in range(n_vars)))
    p_tuple = (
        "({},)".format(", ".join(f"p{j}" for j in range(n_params)))
        if n_params
        else "()"
    )
    return (
        f"def _ode_flat({', '.join(args)}):\n    return _ode({y_tuple}, t, {p_tuple})\n"
    )


@functools.cache
def make_jacobian_column(ode_fn, n_vars: int, n_params: int):
    """Return a device callable writing one column of the ODE's derivatives.

    The result is called as ``jacobian_column(dout, y, t, p, col)`` — with
    ``y`` and ``p`` indexable by component. For ``col < n_vars`` it fills
    ``dout`` with column ``col`` of ``df/dy``; at ``col == n_vars`` it fills
    ``dout`` with ``df/dt``. ``dout`` must hold ``n_vars`` float64s.

    ``col`` is passed straight through to the derivative, which builds the unit
    seed itself, so this is one call site and one Enzyme build whatever
    ``n_vars`` is. Specialising the seed to a literal per column would let
    constant folding drop the zero tangents — a genuine column-wise sparsity
    compression — but at the cost of one derivative per column, which is the
    build cost this arrangement exists to avoid.
    """
    ode_device = as_cuda_device(ode_fn)
    n_args = n_vars + 1 + n_params
    signature = types.UniTuple(types.float64, n_vars)(*([types.float64] * n_args))

    namespace = {"_ode": ode_device}
    exec(  # noqa: S102
        compile(_adapter_source(n_vars, n_params), "<enzyme ode adapter>", "exec"),
        namespace,
    )
    primal = cuda.jit(device=True)(namespace["_ode_flat"])

    built = differentiate_cuda(primal, signature=signature, modes=("jacfwd_column",))
    column_namespace = {"_jacfwd": built.externals["jacfwd_column"]}
    call_args = ", ".join(
        ["dout"]
        + [f"y[{j}]" for j in range(n_vars)]
        + ["t"]
        + [f"p[{j}]" for j in range(n_params)]
        + ["col"]
    )
    exec(  # noqa: S102
        compile(
            f"def _jacobian_column(dout, y, t, p, col):\n    _jacfwd({call_args})\n",
            "<enzyme jacobian column>",
            "exec",
        ),
        column_namespace,
    )
    return cuda.jit(device=True)(column_namespace["_jacobian_column"])
