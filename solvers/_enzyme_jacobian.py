"""Enzyme-derived Jacobians for the implicit solvers.

The implicit kernels need ``df/dy`` (and, for non-autonomous systems,
``df/dt``) of a user ``ode_fn`` written in the numba tuple form described in
``CLAUDE.md``. Rather than making the caller hand-write those, this module
differentiates ``ode_fn`` with `numba-enzyme
<https://github.com/Qruise-ai/numba-enzyme>`_, which runs Enzyme over the
device function's LLVM IR and hands back a device-callable derivative.

``ode_fn`` is differentiated as it stands, with no adapter around it.
Numba-CUDA-MLIR flattens a tuple argument into one scalar parameter per element
under the C ABI and lowers a tuple return to an LLVM struct returned by value,
so a callback of the documented shape already reaches Enzyme as a function of
flat scalars returning a struct — which is exactly what Enzyme differentiates.
Forward-differentiating it and seeding a unit vector gives a whole Jacobian
column per sweep; seeding the time argument instead gives the whole ``df/dt``
vector, so ``n_vars + 1`` sweeps supply both matrices the kernel needs.

The derivative's call shape mirrors the primal's argument list, with each
tuple argument supplied as a contiguous array — so the kernel passes the same
``y[i]`` and ``p[i]`` rows it already passes to ``ode_fn`` itself, and the call
is five arguments at any ``n_vars``. numba-enzyme's entry point loads the
scalars out of those rows before handing them to Enzyme.

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
That is why ``dout`` costs no local memory: inlined, it promotes to registers,
and so do the loads the derivative makes out of ``y`` and ``p``.

An explicit signature is required, because an array cannot say how long the
tuple it stands for is. It also pins the specialisation: the callbacks are
duck-typed on indexing, so the kernel's own calls specialise ``ode_fn`` for
array arguments, and only the signature says the derivative wants the tuple
form.

The parameter partials are never seeded, so they cost nothing. Seeding
direction ``n_vars + 1 + j`` instead would give ``df/dp_j`` as a whole column,
which is what a forward sensitivity analysis of the trajectory would need.
"""

from __future__ import annotations

import functools

from numba_cuda_mlir import types
from numba_enzyme import jacfwd_column

from solvers._numba_common import as_cuda_device


@functools.cache
def make_jacobian_column(ode_fn, n_vars: int, n_params: int):
    """Return a device callable writing one column of the ODE's derivatives.

    The result is called as ``jacobian_column(dout, y, t, p, col)``, where
    ``y`` and ``p`` are the trajectory's state and parameter rows — the same
    arrays the kernel passes to ``ode_fn``. For ``col < n_vars`` it fills
    ``dout`` with column ``col`` of ``df/dy``; at ``col == n_vars`` it fills
    ``dout`` with ``df/dt``. ``dout`` must hold ``n_vars`` float64s.

    ``col`` indexes the primal's flattened argument list, which is why the time
    derivative comes free: ``t`` is the argument after the state. It is a
    run-time argument and the unit seed is built inside the derivative, so this
    is one Enzyme build whatever ``n_vars`` is. Specialising the seed to a
    literal per column would let constant folding drop the zero tangents — a
    genuine column-wise sparsity compression — but at the cost of one
    derivative per column, which is the build cost this arrangement avoids.
    """
    signature = types.UniTuple(types.float64, n_vars)(
        types.UniTuple(types.float64, n_vars),
        types.float64,
        types.UniTuple(types.float64, n_params),
    )
    return jacfwd_column(as_cuda_device(ode_fn), signature=signature)
