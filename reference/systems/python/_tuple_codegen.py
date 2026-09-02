"""Helpers for building Numba-compatible fixed-size tuple callbacks."""

from __future__ import annotations

import ast
import copy
from collections.abc import Sequence

Expr = ast.expr


def const(value: float) -> Expr:
    return ast.Constant(float(value))


def y(index: int) -> Expr:
    return _subscript("y", index)


def p(index: int) -> Expr:
    return _subscript("p", index)


def add(*terms: Expr) -> Expr:
    if not terms:
        return const(0.0)
    expr = _clone(terms[0])
    for term in terms[1:]:
        expr = ast.BinOp(left=expr, op=ast.Add(), right=_clone(term))
    return expr


def sub(left: Expr, right: Expr) -> Expr:
    return ast.BinOp(left=_clone(left), op=ast.Sub(), right=_clone(right))


def mul(*factors: Expr) -> Expr:
    if not factors:
        return const(1.0)
    expr = _clone(factors[0])
    for factor in factors[1:]:
        expr = ast.BinOp(left=expr, op=ast.Mult(), right=_clone(factor))
    return expr


def neg(expr: Expr) -> Expr:
    return ast.UnaryOp(op=ast.USub(), operand=_clone(expr))


def square(expr: Expr) -> Expr:
    return mul(expr, expr)


def make_tuple_callback(name: str, values: Sequence[Expr]):
    return _compile_callback(
        name,
        ast.Tuple(elts=[_float_expr(value) for value in values], ctx=ast.Load()),
    )


def make_matrix_callback(name: str, rows: Sequence[Sequence[Expr]]):
    return _compile_callback(
        name,
        ast.Tuple(
            elts=[
                ast.Tuple(
                    elts=[_float_expr(value) for value in row],
                    ctx=ast.Load(),
                )
                for row in rows
            ],
            ctx=ast.Load(),
        ),
    )


def zero_matrix(n_rows: int, n_cols: int) -> list[list[Expr]]:
    return [[const(0.0) for _ in range(n_cols)] for _ in range(n_rows)]


def _compile_callback(name: str, return_value: Expr):
    module = ast.Module(
        body=[
            ast.FunctionDef(
                name=name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[ast.arg(arg="y"), ast.arg(arg="t"), ast.arg(arg="p")],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=[ast.Return(value=return_value)],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, filename=f"<generated {name}>", mode="exec"), namespace)
    return namespace[name]


def _subscript(name: str, index: int) -> Expr:
    return ast.Subscript(
        value=ast.Name(id=name, ctx=ast.Load()),
        slice=ast.Constant(index),
        ctx=ast.Load(),
    )


def _float_expr(expr: Expr) -> Expr:
    return add(expr, const(0.0))


def _clone(expr: Expr) -> Expr:
    return copy.deepcopy(expr)
