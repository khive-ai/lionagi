# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""AST nodes for LNDL symbolic phase."""

from __future__ import annotations

from dataclasses import dataclass, field


class SNode:
    __slots__ = ()


class SExpr(SNode):
    __slots__ = ()


class SStmt(SNode):
    __slots__ = ()


# ── Expressions ──────────────────────────────────────────────


@dataclass(slots=True)
class SLiteral(SExpr):
    value: str | int | float | bool | None


@dataclass(slots=True)
class SIdentifier(SExpr):
    name: str


@dataclass(slots=True)
class SDeref(SExpr):
    """*name — dereference / force a thunk."""

    name: str


@dataclass(slots=True)
class SBinaryOp(SExpr):
    left: SExpr
    op: str
    right: SExpr


@dataclass(slots=True)
class SUnaryOp(SExpr):
    op: str
    operand: SExpr


@dataclass(slots=True)
class SCall(SExpr):
    """Function call: name(arg1, arg2, kwarg=val)."""

    name: str
    args: list[SExpr] = field(default_factory=list)
    kwargs: dict[str, SExpr] = field(default_factory=dict)


@dataclass(slots=True)
class SDo(SExpr):
    """DO expr — execute env tool or defined function. Returns value."""

    call: SCall


@dataclass(slots=True)
class SIfExpr(SExpr):
    """Inline if-expression: if cond: expr_a; else: expr_b"""

    condition: SExpr
    then_expr: SExpr
    else_expr: SExpr | None = None


# ── Statements ───────────────────────────────────────────────


@dataclass(slots=True)
class SAssign(SStmt):
    """name = expr"""

    name: str
    value: SExpr


@dataclass(slots=True)
class SIfStmt(SStmt):
    """if/elif/else block."""

    condition: SExpr
    body: list[SStmt]
    elif_clauses: list[tuple[SExpr, list[SStmt]]] = field(default_factory=list)
    else_body: list[SStmt] | None = None


@dataclass(slots=True)
class SExprStmt(SStmt):
    """Expression used as statement (e.g., *lact_name or DO call)."""

    expr: SExpr


@dataclass(slots=True)
class SOut(SStmt):
    """OUT{Model: [field1, field2], scalar: literal}"""

    fields: dict[str, list[str] | str | int | float | bool]


@dataclass(slots=True)
class SReturn(SStmt):
    """return expr — exit current scope."""

    value: SExpr


@dataclass(slots=True)
class SNoteSet(SStmt):
    """note["key"] = expr"""

    key: str
    value: SExpr


@dataclass(slots=True)
class SNoteGet(SExpr):
    """note["key"] — retrieve persisted value."""

    key: str


# ── Program ──────────────────────────────────────────────────


@dataclass(slots=True)
class SBlock:
    """A single ```lndl code block — a list of statements."""

    stmts: list[SStmt]


__all__ = (
    "SAssign",
    "SBinaryOp",
    "SBlock",
    "SCall",
    "SDeref",
    "SDo",
    "SExpr",
    "SExprStmt",
    "SIdentifier",
    "SIfExpr",
    "SIfStmt",
    "SLiteral",
    "SNode",
    "SNoteGet",
    "SNoteSet",
    "SOut",
    "SReturn",
    "SStmt",
    "SUnaryOp",
)
