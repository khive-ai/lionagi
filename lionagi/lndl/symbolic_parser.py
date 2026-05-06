# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Parser for LNDL symbolic phase — Pratt-style expression parser
with Python-like indentation-based blocks."""

from __future__ import annotations

from .symbolic_ast import (
    SAssign,
    SBinaryOp,
    SBlock,
    SCall,
    SDeref,
    SDo,
    SExpr,
    SExprStmt,
    SIdentifier,
    SIfExpr,
    SIfStmt,
    SLiteral,
    SNoteGet,
    SNoteSet,
    SOut,
    SReturn,
    SStmt,
    SUnaryOp,
)
from .symbolic_lexer import STok, SToken


class SParseError(Exception):
    pass


class SymbolicParser:
    """Recursive descent parser for LNDL symbolic code blocks."""

    def __init__(self, tokens: list[SToken]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _cur(self) -> SToken:
        return self.tokens[self.pos]

    def _peek_type(self) -> STok:
        return self.tokens[self.pos].type

    def _at(self, *types: STok) -> bool:
        return self._peek_type() in types

    def _eat(self, tok_type: STok) -> SToken:
        tok = self._cur()
        if tok.type != tok_type:
            raise SParseError(
                f"Expected {tok_type.name}, got {tok.type.name} ({tok.value!r}) at line {tok.line}"
            )
        self.pos += 1
        return tok

    def _skip_newlines(self) -> None:
        while self._at(STok.NEWLINE):
            self.pos += 1

    def _eat_newline_or_semi(self) -> None:
        if self._at(STok.SEMI):
            self.pos += 1
        while self._at(STok.NEWLINE):
            self.pos += 1

    # ── Top-level ────────────────────────────────────────────

    def parse(self) -> SBlock:
        self._skip_newlines()
        stmts = self._parse_stmt_list(until={STok.EOF})
        return SBlock(stmts)

    def _parse_stmt_list(self, until: set[STok]) -> list[SStmt]:
        stmts: list[SStmt] = []
        while not self._at(*until):
            self._skip_newlines()
            if self._at(*until):
                break
            stmts.append(self._parse_stmt())
            self._eat_newline_or_semi()
        return stmts

    # ── Statements ───────────────────────────────────────────

    def _parse_stmt(self) -> SStmt:
        self._skip_newlines()
        t = self._peek_type()

        if t == STok.IF:
            return self._parse_if_stmt()

        if t == STok.RETURN:
            return self._parse_return()

        if t == STok.OUT:
            return self._parse_out()

        if t == STok.NOTE:
            return self._parse_note_stmt()

        # Assignment: ident = expr
        if t == STok.IDENT and self.pos + 1 < len(self.tokens):
            next_tok = self.tokens[self.pos + 1]
            if next_tok.type == STok.ASSIGN:
                return self._parse_assign()

        # Expression statement (DO, *deref, function call, etc.)
        expr = self._parse_expr()
        return SExprStmt(expr)

    def _parse_assign(self) -> SAssign:
        name = self._eat(STok.IDENT).value
        self._eat(STok.ASSIGN)
        value = self._parse_expr()
        return SAssign(name, value)

    def _parse_if_stmt(self) -> SIfStmt:
        self._eat(STok.IF)
        condition = self._parse_expr()
        self._eat(STok.COLON)
        body = self._parse_block()

        elif_clauses: list[tuple[SExpr, list[SStmt]]] = []
        while self._at(STok.ELIF):
            self._eat(STok.ELIF)
            elif_cond = self._parse_expr()
            self._eat(STok.COLON)
            elif_body = self._parse_block()
            elif_clauses.append((elif_cond, elif_body))

        else_body: list[SStmt] | None = None
        if self._at(STok.ELSE):
            self._eat(STok.ELSE)
            self._eat(STok.COLON)
            else_body = self._parse_block()

        return SIfStmt(condition, body, elif_clauses, else_body)

    def _parse_block(self) -> list[SStmt]:
        self._skip_newlines()
        if self._at(STok.INDENT):
            self._eat(STok.INDENT)
            stmts = self._parse_stmt_list(until={STok.DEDENT, STok.EOF})
            if self._at(STok.DEDENT):
                self._eat(STok.DEDENT)
            return stmts
        # Single-line block (no indent): parse one statement
        stmt = self._parse_stmt()
        return [stmt]

    def _parse_return(self) -> SReturn:
        self._eat(STok.RETURN)
        value = self._parse_expr()
        return SReturn(value)

    def _parse_out(self) -> SOut:
        """OUT{Model: [field1, field2], scalar: literal, alias: ident}.

        Three field shapes:
          - [name, ...]  list of bound-variable refs
          - "..."/N/bool literal
          - ident        sugar for [ident]
        """
        self._eat(STok.OUT)
        self._eat(STok.LBRACE)
        fields: dict[str, list[str] | str | int | float | bool] = {}
        while not self._at(STok.RBRACE):
            key = self._eat(STok.IDENT).value
            self._eat(STok.COLON)
            if self._at(STok.LBRACKET):
                self._eat(STok.LBRACKET)
                refs: list[str] = []
                while not self._at(STok.RBRACKET):
                    refs.append(self._eat(STok.IDENT).value)
                    if self._at(STok.COMMA):
                        self._eat(STok.COMMA)
                self._eat(STok.RBRACKET)
                fields[key] = refs
            elif self._at(STok.IDENT):
                fields[key] = [self._eat(STok.IDENT).value]
            else:
                fields[key] = self._parse_scalar_literal()
            if self._at(STok.COMMA):
                self._eat(STok.COMMA)
            self._skip_newlines()
        self._eat(STok.RBRACE)
        return SOut(fields)

    def _parse_scalar_literal(self) -> str | int | float | bool:
        t = self._cur()
        if t.type == STok.STRING:
            self.pos += 1
            return t.value
        if t.type == STok.INT:
            self.pos += 1
            return int(t.value)
        if t.type == STok.FLOAT:
            self.pos += 1
            return float(t.value)
        if t.type == STok.TRUE:
            self.pos += 1
            return True
        if t.type == STok.FALSE:
            self.pos += 1
            return False
        raise SParseError(f"Expected scalar literal, got {t.type.name} at line {t.line}")

    def _parse_note_stmt(self) -> SNoteSet:
        self._eat(STok.NOTE)
        self._eat(STok.LBRACKET)
        key = self._eat(STok.STRING).value
        self._eat(STok.RBRACKET)
        self._eat(STok.ASSIGN)
        value = self._parse_expr()
        return SNoteSet(key, value)

    # ── Expressions (Pratt precedence) ───────────────────────

    def _parse_expr(self) -> SExpr:
        # Inline if-expression: if cond: expr_a; else: expr_b
        if self._at(STok.IF):
            return self._parse_if_expr()
        return self._parse_or()

    def _parse_if_expr(self) -> SIfExpr:
        self._eat(STok.IF)
        condition = self._parse_or()
        self._eat(STok.COLON)
        then_expr = self._parse_or()
        else_expr = None
        if self._at(STok.SEMI):
            self._eat(STok.SEMI)
        if self._at(STok.ELSE):
            self._eat(STok.ELSE)
            self._eat(STok.COLON)
            else_expr = self._parse_or()
        return SIfExpr(condition, then_expr, else_expr)

    def _parse_or(self) -> SExpr:
        left = self._parse_and()
        while self._at(STok.OR):
            self.pos += 1
            right = self._parse_and()
            left = SBinaryOp(left, "or", right)
        return left

    def _parse_and(self) -> SExpr:
        left = self._parse_not()
        while self._at(STok.AND):
            self.pos += 1
            right = self._parse_not()
            left = SBinaryOp(left, "and", right)
        return left

    def _parse_not(self) -> SExpr:
        if self._at(STok.NOT):
            self.pos += 1
            operand = self._parse_not()
            return SUnaryOp("not", operand)
        return self._parse_comparison()

    def _parse_comparison(self) -> SExpr:
        left = self._parse_addition()
        cmp_ops = {STok.EQ, STok.NEQ, STok.LT, STok.GT, STok.LTE, STok.GTE}
        if self._at(*cmp_ops):
            op = self._cur().value
            self.pos += 1
            right = self._parse_addition()
            left = SBinaryOp(left, op, right)
        return left

    def _parse_addition(self) -> SExpr:
        left = self._parse_multiplication()
        while self._at(STok.PLUS, STok.MINUS):
            op = self._cur().value
            self.pos += 1
            right = self._parse_multiplication()
            left = SBinaryOp(left, op, right)
        return left

    def _parse_multiplication(self) -> SExpr:
        left = self._parse_unary()
        while self._at(STok.STAR, STok.SLASH):
            op = self._cur().value
            self.pos += 1
            right = self._parse_unary()
            left = SBinaryOp(left, op, right)
        return left

    def _parse_unary(self) -> SExpr:
        if self._at(STok.MINUS):
            self.pos += 1
            operand = self._parse_unary()
            return SUnaryOp("-", operand)
        return self._parse_postfix()

    def _parse_postfix(self) -> SExpr:
        expr = self._parse_primary()
        while self._at(STok.LPAREN, STok.LBRACKET, STok.DOT):
            if self._at(STok.LPAREN) and isinstance(expr, SIdentifier):
                args, kwargs = self._parse_call_args()
                expr = SCall(expr.name, args, kwargs)
            elif self._at(STok.LBRACKET):
                self._eat(STok.LBRACKET)
                idx = self._parse_expr()
                self._eat(STok.RBRACKET)
                if (
                    isinstance(expr, SIdentifier)
                    and expr.name == "note"
                    and isinstance(idx, SLiteral)
                ):
                    expr = SNoteGet(str(idx.value))
                else:
                    expr = SBinaryOp(expr, "[]", idx)
            else:
                break
        return expr

    def _parse_call_args(self) -> tuple[list[SExpr], dict[str, SExpr]]:
        self._eat(STok.LPAREN)
        args: list[SExpr] = []
        kwargs: dict[str, SExpr] = {}
        while not self._at(STok.RPAREN):
            # Check for kwarg: name=expr
            if (
                self._at(STok.IDENT)
                and self.pos + 1 < len(self.tokens)
                and self.tokens[self.pos + 1].type == STok.ASSIGN
            ):
                name = self._eat(STok.IDENT).value
                self._eat(STok.ASSIGN)
                val = self._parse_expr()
                kwargs[name] = val
            else:
                args.append(self._parse_expr())
            if self._at(STok.COMMA):
                self._eat(STok.COMMA)
        self._eat(STok.RPAREN)
        return args, kwargs

    def _parse_primary(self) -> SExpr:
        t = self._cur()

        if t.type == STok.INT:
            self.pos += 1
            return SLiteral(int(t.value))

        if t.type == STok.FLOAT:
            self.pos += 1
            return SLiteral(float(t.value))

        if t.type == STok.STRING:
            self.pos += 1
            return SLiteral(t.value)

        if t.type == STok.TRUE:
            self.pos += 1
            return SLiteral(True)

        if t.type == STok.FALSE:
            self.pos += 1
            return SLiteral(False)

        if t.type == STok.NONE:
            self.pos += 1
            return SLiteral(None)

        if t.type == STok.DEREF:
            self.pos += 1
            return SDeref(t.value)

        if t.type == STok.DO:
            return self._parse_do()

        if t.type == STok.IDENT:
            self.pos += 1
            return SIdentifier(t.value)

        if t.type == STok.NOTE:
            self.pos += 1
            if self._at(STok.LBRACKET):
                self._eat(STok.LBRACKET)
                key = self._eat(STok.STRING).value
                self._eat(STok.RBRACKET)
                return SNoteGet(key)
            return SIdentifier("note")

        if t.type == STok.LPAREN:
            self._eat(STok.LPAREN)
            expr = self._parse_expr()
            self._eat(STok.RPAREN)
            return expr

        raise SParseError(f"Unexpected token {t.type.name} ({t.value!r}) at line {t.line}")

    def _parse_do(self) -> SDo:
        self._eat(STok.DO)
        # DO func_name(args) or DO *lact_name
        if self._at(STok.DEREF):
            deref = self._cur()
            self.pos += 1
            if self._at(STok.LPAREN):
                args, kwargs = self._parse_call_args()
                return SDo(SCall(deref.value, args, kwargs))
            return SDo(SCall(deref.value, [], {}))

        name = self._eat(STok.IDENT).value
        if self._at(STok.LPAREN):
            args, kwargs = self._parse_call_args()
            return SDo(SCall(name, args, kwargs))
        return SDo(SCall(name, [], {}))


def parse_symbolic_block(text: str) -> SBlock:
    """Parse a single ```lndl code block into an AST."""
    from .symbolic_lexer import SymbolicLexer

    lexer = SymbolicLexer(text)
    tokens = lexer.tokenize()
    parser = SymbolicParser(tokens)
    return parser.parse()


__all__ = ("SParseError", "SymbolicParser", "parse_symbolic_block")
