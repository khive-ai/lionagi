"""
LNDL AST Nodes - Clean cognitive programming structures
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any


class Node(ABC):
    """Base AST node"""

    pass


# Expressions
class Expr(Node):
    """Base expression"""

    pass


@dataclass
class Literal(Expr):
    """Literal values"""

    value: str | int | float | bool


@dataclass
class Identifier(Expr):
    """Variable references"""

    name: str


@dataclass
class SemanticOp(Expr):
    """Semantic operations: similar(), synthesize(), etc."""

    op: str
    args: list[Expr]


@dataclass
class BinaryOp(Expr):
    """Binary operations: >, <, =="""

    left: Expr
    operator: str
    right: Expr


@dataclass
class FuncCall(Expr):
    """Function calls"""

    name: str
    args: list[Expr]


# Statements
class Stmt(Node):
    """Base statement"""

    pass


@dataclass
class CogVar(Stmt):
    """Cognitive variable: <let-lvar, field_X as var>content<let-lvar/>"""

    field_type: str
    name: str
    content: str


@dataclass
class Let(Stmt):
    """Variable assignment: let var = expr;"""

    name: str
    expr: Expr


@dataclass
class If(Stmt):
    """Conditional: IF condition; then_body; ELSE; else_body;"""

    condition: Expr
    then_body: list[Stmt]
    else_body: list[Stmt] | None = None


@dataclass
class Do(Stmt):
    """Action: DO action WITH params(*context);"""

    action: str
    params: list[Expr]


@dataclass
class ExprStmt(Stmt):
    """Expression as statement"""

    expr: Expr


# Program
@dataclass
class Program(Node):
    """Root program node"""

    stmts: list[Stmt]


# Context Types
@dataclass
class Context:
    """Execution context"""

    vars: dict[str, Any]
    confidence_threshold: float = 0.8


@dataclass
class Result:
    """Operation result"""

    value: Any
    confidence: float
    reasoning: str | None = None
