"""Write an ODE right-hand side once, run it on both backends.

The modax kernel solvers compile their callback with ``numba_cuda_mlir``,
which wants ``math`` scalars and a fixed-size tuple; the Diffrax reference
backends trace the same equations as ``jnp`` arrays. Each example here needs
both, and writing the two by hand is how they drift apart.

The way out is to write the body *once*, as a factory that takes the handful
of names the two backends spell differently, and to call that factory twice::

    def _make_rhs(*, exp, sqrt):
        def rhs(y, t, p):
            return (-exp(-y[0]) * p[0], sqrt(y[1]))

        return rhs

    RHS = build_rhs(_make_rhs)

``RHS.device`` closes over ``math.exp``/``math.sqrt`` and returns the tuple the
kernel solvers want; ``RHS.jax`` closes over ``jnp.exp``/``jnp.sqrt`` and
returns the array Diffrax and the rest of the example want. Numba resolves a
closed-over function object at compile time exactly as it resolves a global,
so the device form lowers to the code a hand-written one would --
``tests/test_examples.py`` compiles every device callback to PTX and checks
its values against the traced twin.

What such a body may use:

* the operations listed in :data:`MATH_OPS`, taken as keyword arguments;
* arithmetic, comparisons, indexing and literals;
* another :class:`Forms` pair passed in as a keyword argument -- a helper
  function (see :func:`build_fn`) or a per-backend value such as a lookup
  table. Device code cannot call a plain Python function, so a helper the
  body shares with the rest of the module has to arrive this way;
* a helper defined *inside* the body, which numba inlines;
* no ``if`` on a value that varies: a traced one cannot be branched on at
  all, so use ``maximum``/``minimum`` or a branchless
  ``a + (b - a) * (x > c)`` select.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, NamedTuple

import jax.numpy as jnp
from numba_cuda_mlir import cuda


class Forms(NamedTuple):
    """One body in its three forms.

    ``device`` is what a modax solver takes: ``math`` scalars, and any helper
    it calls compiled as a CUDA device function. ``jax`` is the traced form.
    ``host`` is the device body with those helpers left as plain Python, which
    is the same arithmetic runnable without a GPU -- how the tests compare the
    two backends on a machine that has none. With no helpers to compile the
    two are the same function.
    """

    device: Any
    jax: Any
    host: Any


# The names a body takes as keyword arguments, in each backend's spelling.
# ``maximum``/``minimum`` are the two-argument forms (``jnp.maximum`` and the
# builtins agree on that shape) and ``index_of`` floors to a table index.
MATH_OPS: dict[str, Callable] = {
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "maximum": max,
    "minimum": min,
    "index_of": int,
}

JAX_OPS: dict[str, Callable] = {
    "exp": jnp.exp,
    "log": jnp.log,
    "log10": jnp.log10,
    "sqrt": jnp.sqrt,
    "sin": jnp.sin,
    "cos": jnp.cos,
    "maximum": jnp.maximum,
    "minimum": jnp.minimum,
    "index_of": lambda value: jnp.floor(value).astype(jnp.int32),
}


def _call(factory, side: str, kwargs):
    """Call ``factory`` for one form, ``side`` naming a field of :class:`Forms`.

    Only the ops the factory names are passed, so a body that needs nothing
    but arithmetic declares nothing. A :class:`Forms` argument is resolved to
    this form's member, which is how a body reaches a helper -- or a table --
    that the rest of the module shares with it.
    """
    ops = JAX_OPS if side == "jax" else MATH_OPS
    wanted = set(inspect.signature(factory).parameters)
    unknown = set(kwargs) - wanted
    if unknown:
        raise TypeError(f"{factory.__name__} takes no {sorted(unknown)} argument")
    arguments = {name: ops[name] for name in wanted & set(ops)}
    for name, value in kwargs.items():
        arguments[name] = getattr(value, side) if isinstance(value, Forms) else value
    return factory(**arguments)


def build_fn(factory, **kwargs) -> Forms:
    """Build a scalar helper that a right-hand side and the module can share.

    The device member is a ``cuda.jit`` device function, which is what makes
    it callable from a device callback -- and inlinable, so Enzyme
    differentiates straight through it.
    """
    host = _call(factory, "host", kwargs)
    device = cuda.jit(device=True, inline="always")(_call(factory, "device", kwargs))
    return Forms(device, _call(factory, "jax", kwargs), host)


def build_rhs(factory, **kwargs) -> Forms:
    """Build an ODE right-hand side in all three forms.

    The body returns a tuple of scalars, which is what the kernel solvers
    take; the ``jax`` member stacks that tuple into the array the Diffrax
    backends and the rest of the example expect.
    """
    traced = _call(factory, "jax", kwargs)

    def rhs(y, t, p):
        return jnp.stack(jnp.broadcast_arrays(*traced(y, t, p)))

    host = _call(factory, "host", kwargs)
    # With nothing to compile the two forms are one function, so what the
    # tests exercise through ``host`` is the callback the solver is handed.
    device = (
        host if not _has_device_helper(kwargs) else _call(factory, "device", kwargs)
    )
    return Forms(device, rhs, host)


def _has_device_helper(kwargs) -> bool:
    """Does any argument differ between the device and host forms?"""
    return any(
        isinstance(value, Forms) and value.device is not value.host
        for value in kwargs.values()
    )
