"""Build a numba-compatible tuple callback from one expression per component.

The kernel solvers differentiate ``ode_fn`` with Enzyme, which needs every
index into ``y`` and ``p`` to be a literal (see "Writing ODE callbacks" in
AGENTS.md). A system whose dimension is a parameter therefore cannot loop over
its components inside the callback; it spells them out instead. Each reference
system writes its components as plain Python expression strings over ``y[i]``,
``p[j]`` and ``t`` -- ``f"{coeff!r} * y[{j}]"`` and the like -- and this module
turns the list into ``ode_fn(y, t, p)`` returning them as a tuple.
"""

from __future__ import annotations

from collections.abc import Sequence


def make_tuple_callback(name: str, components: Sequence[str]):
    """``def name(y, t, p): return (c0, c1, ...)`` from expression strings.

    Each component is wrapped as ``(expr) + 0.0`` so an entry that is a bare
    ``y[i]`` still lowers to a float and never aliases its input.
    """
    body = ",\n        ".join(f"({expr}) + 0.0" for expr in components)
    source = f"def {name}(y, t, p):\n    return (\n        {body},\n    )\n"
    namespace: dict[str, object] = {}
    exec(compile(source, f"<generated {name}>", "exec"), namespace)  # noqa: S102
    fn = namespace[name]
    fn._generated_source = source
    return fn


def linear_combination(coefficients: Sequence[float], indices: Sequence[int]) -> str:
    """``c0 * y[i0] + c1 * y[i1] + ...`` with the zero coefficients dropped."""
    terms = [f"{float(c)!r} * y[{int(i)}]" for c, i in zip(coefficients, indices) if c]
    return " + ".join(terms) if terms else "0.0"
