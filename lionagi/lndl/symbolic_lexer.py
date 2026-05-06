# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Lexer for LNDL symbolic phase (```lndl code blocks)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class STok(Enum):
    # Literals
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()

    # Identifiers & refs
    IDENT = auto()
    DEREF = auto()  # *name

    # Keywords
    DO = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    RETURN = auto()
    OUT = auto()
    NOTE = auto()
    DEF = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    EQ = auto()  # ==
    NEQ = auto()  # !=
    LT = auto()  # <
    GT = auto()  # >
    LTE = auto()  # <=
    GTE = auto()  # >=
    ASSIGN = auto()  # =

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    COMMA = auto()
    COLON = auto()
    SEMI = auto()
    DOT = auto()

    # Special
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()
    EOF = auto()


_KEYWORDS = {
    "DO": STok.DO,
    "if": STok.IF,
    "elif": STok.ELIF,
    "else": STok.ELSE,
    "and": STok.AND,
    "or": STok.OR,
    "not": STok.NOT,
    "return": STok.RETURN,
    "OUT": STok.OUT,
    "note": STok.NOTE,
    "DEF": STok.DEF,
    "True": STok.TRUE,
    "true": STok.TRUE,
    "False": STok.FALSE,
    "false": STok.FALSE,
    "None": STok.NONE,
}


@dataclass(slots=True)
class SToken:
    type: STok
    value: str
    line: int
    col: int


class SymbolicLexer:
    """Tokenizer for LNDL symbolic code blocks.

    Produces INDENT/DEDENT tokens from leading whitespace changes,
    similar to Python's tokenizer but simpler (spaces only, no tabs).
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[SToken] = []
        self._indent_stack: list[int] = [0]
        self._at_line_start = True

    def _ch(self) -> str | None:
        return self.text[self.pos] if self.pos < len(self.text) else None

    def _peek(self, offset: int = 1) -> str | None:
        p = self.pos + offset
        return self.text[p] if p < len(self.text) else None

    def _advance(self) -> str:
        ch = self.text[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _emit(self, tok_type: STok, value: str) -> None:
        self.tokens.append(SToken(tok_type, value, self.line, self.col))

    def _read_string(self) -> str:
        quote = self._advance()
        result: list[str] = []
        while (ch := self._ch()) is not None and ch != quote:
            if ch == "\\":
                self._advance()
                esc = self._ch()
                if esc is not None:
                    result.append(
                        {"n": "\n", "t": "\t", "\\": "\\", '"': '"', "'": "'"}.get(esc, esc)
                    )
                    self._advance()
            else:
                result.append(ch)
                self._advance()
        if self._ch() == quote:
            self._advance()
        return "".join(result)

    def _read_number(self) -> SToken:
        start = self.pos
        line, col = self.line, self.col
        is_float = False
        while (ch := self._ch()) is not None and (ch.isdigit() or ch == "."):
            if ch == ".":
                if is_float:
                    break
                is_float = True
            self._advance()
        val = self.text[start : self.pos]
        return SToken(STok.FLOAT if is_float else STok.INT, val, line, col)

    def _read_ident(self) -> str:
        start = self.pos
        while (ch := self._ch()) is not None and (ch.isalnum() or ch == "_"):
            self._advance()
        return self.text[start : self.pos]

    def _handle_indent(self) -> None:
        spaces = 0
        while self._ch() == " ":
            spaces += 1
            self._advance()
        if self._ch() == "\n" or self._ch() is None:
            return
        if self._ch() == "#":
            while self._ch() is not None and self._ch() != "\n":
                self._advance()
            return

        current = self._indent_stack[-1]
        if spaces > current:
            self._indent_stack.append(spaces)
            self._emit(STok.INDENT, "")
        elif spaces < current:
            while self._indent_stack[-1] > spaces:
                self._indent_stack.pop()
                self._emit(STok.DEDENT, "")
        self._at_line_start = False

    def tokenize(self) -> list[SToken]:
        while self.pos < len(self.text):
            if self._at_line_start:
                self._handle_indent()
                if self.pos >= len(self.text):
                    break
                continue

            ch = self._ch()
            if ch is None:
                break

            if ch == "\n":
                self._emit(STok.NEWLINE, "\n")
                self._advance()
                self._at_line_start = True
                continue

            if ch in " \t\r":
                self._advance()
                continue

            if ch == "#":
                while self._ch() is not None and self._ch() != "\n":
                    self._advance()
                continue

            line, col = self.line, self.col

            if ch in "\"'":
                val = self._read_string()
                self.tokens.append(SToken(STok.STRING, val, line, col))
                continue

            if ch.isdigit():
                self.tokens.append(self._read_number())
                continue

            if ch == "-" and self._peek() is not None and self._peek().isdigit():
                self._advance()
                tok = self._read_number()
                tok.value = "-" + tok.value
                tok.line = line
                tok.col = col
                self.tokens.append(tok)
                continue

            if (
                ch == "*"
                and self._peek() is not None
                and (self._peek().isalpha() or self._peek() == "_")
            ):
                self._advance()
                name = self._read_ident()
                self.tokens.append(SToken(STok.DEREF, name, line, col))
                continue

            if ch.isalpha() or ch == "_":
                ident = self._read_ident()
                tok_type = _KEYWORDS.get(ident, STok.IDENT)
                self.tokens.append(SToken(tok_type, ident, line, col))
                continue

            # Two-char operators
            two = self.text[self.pos : self.pos + 2] if self.pos + 1 < len(self.text) else ""
            if two == "==":
                self._advance()
                self._advance()
                self.tokens.append(SToken(STok.EQ, "==", line, col))
                continue
            if two == "!=":
                self._advance()
                self._advance()
                self.tokens.append(SToken(STok.NEQ, "!=", line, col))
                continue
            if two == "<=":
                self._advance()
                self._advance()
                self.tokens.append(SToken(STok.LTE, "<=", line, col))
                continue
            if two == ">=":
                self._advance()
                self._advance()
                self.tokens.append(SToken(STok.GTE, ">=", line, col))
                continue

            # Single-char
            singles = {
                "+": STok.PLUS,
                "-": STok.MINUS,
                "/": STok.SLASH,
                "*": STok.STAR,
                "<": STok.LT,
                ">": STok.GT,
                "=": STok.ASSIGN,
                "(": STok.LPAREN,
                ")": STok.RPAREN,
                "[": STok.LBRACKET,
                "]": STok.RBRACKET,
                "{": STok.LBRACE,
                "}": STok.RBRACE,
                ",": STok.COMMA,
                ":": STok.COLON,
                ";": STok.SEMI,
                ".": STok.DOT,
            }
            if ch in singles:
                self._advance()
                self.tokens.append(SToken(singles[ch], ch, line, col))
                continue

            self._advance()

        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            self._emit(STok.DEDENT, "")
        self._emit(STok.EOF, "")
        return self.tokens


__all__ = ("STok", "SToken", "SymbolicLexer")
