"""Enzyme-derived Jacobians for the implicit solvers.

The implicit kernels need ``df/dy`` (and, for non-autonomous systems,
``df/dt``) of a user ``ode_fn`` written in the numba tuple form described in
``CLAUDE.md``. Rather than making the caller hand-write those, this module
differentiates ``ode_fn`` with `numba-enzyme
<https://github.com/Qruise-ai/numba-enzyme>`_, which runs Enzyme over the
device function's LLVM IR and hands back a device-callable derivative.

The right-hand side is compiled once into a **vector-valued primal** that
writes its outputs through a leading array argument — Numba-CUDA-MLIR cannot
lower a tuple return across an ``abi="c"`` boundary, but it does lower a
leading array. Forward-differentiating that primal and seeding a unit vector
gives a whole Jacobian column per sweep; seeding the time argument instead
gives the whole ``df/dt`` vector, so ``n_vars + 1`` sweeps supply both matrices
the kernel needs. The seed is built inside the derivative from a column index,
so nothing here materialises a tangent vector.

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
That is why ``out`` and ``dout`` cost no local memory: inlined, they promote to
registers.

The parameter partials are never seeded, so they cost nothing. Seeding
direction ``n_vars + 1 + j`` instead would give ``df/dp_j`` as a whole column,
which is what a forward sensitivity analysis of the trajectory would need.
"""

from __future__ import annotations

import functools

from numba_cuda_mlir import cuda, types
from numba_enzyme import jacfwd_column

from solvers._numba_common import as_cuda_device


def _vector_source(n_vars: int, n_params: int) -> str:
    """Source for the vector-valued device form of the ODE right-hand side.

    Enzyme needs the outputs in memory rather than in a tuple, and its
    arguments have to be flat scalars, so the state and parameter tuples the
    callback expects are rebuilt inside the body.
    """
    args = [f"y{j}" for j in range(n_vars)] + ["t"] + [f"p{j}" for j in range(n_params)]
    y_tuple = "({},)".format(", ".join(f"y{j}" for j in range(n_vars)))
    p_tuple = (
        "({},)".format(", ".join(f"p{j}" for j in range(n_params)))
        if n_params
        else "()"
    )
    stores = "\n".join(f"    out[{j}] = values[{j}]" for j in range(n_vars))
    return (
        f"def _ode_vector(out, {', '.join(args)}):\n"
        f"    values = _ode({y_tuple}, t, {p_tuple})\n"
        f"{stores}\n"
    )


@functools.cache
def make_jacobian_column(ode_fn, n_vars: int, n_params: int):
    """Return a device callable writing one column of the ODE's derivatives.

    The result is called as ``jacobian_column(out, dout, y, t, p, col)`` — with
    ``y`` and ``p`` indexable by component. For ``col < n_vars`` it fills
    ``dout`` with column ``col`` of ``df/dy``; at ``col == n_vars`` it fills
    ``dout`` with ``df/dt``. ``out`` receives the right-hand side itself and is
    otherwise unused. Both arrays must hold ``n_vars`` float64s and share a
    layout, since Enzyme is handed the array's offset and stride as inactive
    and the shadow inherits them.

    ``col`` is passed straight through to the derivative, which builds the unit
    seed itself, so this is one call site and one Enzyme build whatever
    ``n_vars`` is. Specialising the seed to a literal per column would let
    constant folding drop the zero tangents — a genuine column-wise sparsity
    compression — but at the cost of one derivative per column, which is the
    build cost this arrangement exists to avoid.
    """
    ode_device = as_cuda_device(ode_fn)
    n_args = n_vars + 1 + n_params
    array = types.float64[::1]
    signature = types.void(array, *([types.float64] * n_args))

    namespace = {"_ode": ode_device}
    exec(  # noqa: S102
        compile(_vector_source(n_vars, n_params), "<enzyme ode vector>", "exec"),
        namespace,
    )
    primal = cuda.jit(device=True)(namespace["_ode_vector"])

    column_namespace = {"_jacfwd": jacfwd_column(primal, signature=signature)}
    call_args = ", ".join(
        ["out", "dout"]
        + [f"y[{j}]" for j in range(n_vars)]
        + ["t"]
        + [f"p[{j}]" for j in range(n_params)]
        + ["col"]
    )
    exec(  # noqa: S102
        compile(
            f"def _jacobian_column(out, dout, y, t, p, col):\n    _jacfwd({call_args})\n",
            "<enzyme jacobian column>",
            "exec",
        ),
        column_namespace,
    )
    return cuda.jit(device=True)(column_namespace["_jacobian_column"])
