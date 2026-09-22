"""Compile generated straight-line device source into a ``cuda.jit`` function.

Two parts of the Rodas5P kernel are emitted as source rather than written: the
sparse factorisation and triangular solves, with every slot a literal
([`solvers._sparse_direct`][]), and the Jacobian writer, with every colour's
seed row a literal (``solvers.rodas5P``). Both are load-bearing -- see the
measurements in AGENTS.md -- and both need the same three things done to the
text they produce, which is what this module does once.
"""

from __future__ import annotations

import itertools
import linecache

from numba_cuda_mlir import cuda

_SOURCES = itertools.count(1)


def compile_device_source(name: str, lines: list[str], namespace: dict | None = None):
    """Exec ``lines`` defining ``name`` and return it as a device function.

    The source is registered with ``linecache`` under a filename of its own, so
    a numba typing error inside it points at the offending line rather than at
    nothing, and the text stays readable from a debugger as
    ``fn._generated_source``. ``namespace`` supplies whatever the generated body
    refers to by name.
    """
    source = "\n".join(lines) + "\n"
    filename = f"<modax generated {name} {next(_SOURCES)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    scope: dict = {}
    exec(compile(source, filename, "exec"), dict(namespace or {}), scope)  # noqa: S102
    generated = scope[name]
    generated._generated_source = source
    return cuda.jit(device=True)(generated)
