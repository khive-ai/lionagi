# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL Parser — recursive descent parser for structured output tags."""

import ast
import re
import warnings
from typing import Any

from .ast import Lact, Lvar, OutBlock, Program, RLvar
from .lexer import Token, TokenType

_warned_action_names: set[str] = set()

PYTHON_RESERVED = {
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
    "print",
    "input",
    "open",
    "len",
    "range",
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "int",
    "float",
    "bool",
    "type",
}


class ParseError(Exception):
    def __init__(self, message: str, token: Token):
        self.message = message
        self.token = token
        super().__init__(
            f"Parse error at line {token.line}, column {token.column}: {message}"
        )


class Parser:
    """Recursive descent parser for LNDL. Not thread-safe."""

    def __init__(self, tokens: list[Token], source_text: str | None = None):
        self.tokens = tokens
        self.pos = 0
        self.source_text = source_text

    def current_token(self) -> Token:
        if self.pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[self.pos]

    def peek_token(self, offset: int = 1) -> Token:
        peek_pos = self.pos + offset
        if peek_pos >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[peek_pos]

    def advance(self) -> None:
        if self.pos < len(self.tokens) - 1:
            self.pos += 1

    def expect(self, token_type: TokenType) -> Token:
        token = self.current_token()
        if token.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {token.type.name}", token
            )
        self.advance()
        return token

    def match(self, *token_types: TokenType) -> bool:
        return self.current_token().type in token_types

    def skip_newlines(self) -> None:
        while self.match(TokenType.NEWLINE):
            self.advance()

    def parse(self) -> Program:
        if self.source_text is None:
            raise ParseError(
                "Parser requires source_text for content extraction",
                self.current_token(),
            )

        lvars: list[Lvar] = []
        lacts: list[Lact] = []
        out_block: OutBlock | None = None
        aliases: set[str] = set()

        while not self.match(TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.EOF):
                break

            if self.match(TokenType.LVAR_OPEN):
                lvar = self.parse_lvar()
                if lvar.alias in aliases:
                    raise ParseError(
                        f"Duplicate alias '{lvar.alias}' - aliases must be unique across lvars and lacts",
                        self.current_token(),
                    )
                aliases.add(lvar.alias)
                lvars.append(lvar)
                continue

            if self.match(TokenType.LACT_OPEN):
                lact = self.parse_lact()
                if lact.alias in aliases:
                    raise ParseError(
                        f"Duplicate alias '{lact.alias}' - aliases must be unique across lvars and lacts",
                        self.current_token(),
                    )
                aliases.add(lact.alias)
                lacts.append(lact)
                continue

            if self.match(TokenType.OUT_OPEN):
                out_block = self.parse_out_block()
                break

            self.advance()

        return Program(lvars=lvars, lacts=lacts, out_block=out_block)

    def parse_lvar(self) -> Lvar | RLvar:
        self.expect(TokenType.LVAR_OPEN)
        self.skip_newlines()

        first_id = self.expect(TokenType.ID).value
        extra_id: str | None = None

        if self.match(TokenType.DOT):
            self.advance()
            field = self.expect(TokenType.ID).value
            model = first_id

            if self.match(TokenType.ID):
                alias = self.current_token().value
                self.advance()
                has_explicit_alias = True
            else:
                alias = field
                has_explicit_alias = False

            is_raw = False
        elif self.match(TokenType.ID):
            # Forgiving 2-ID no-dot pattern: <lvar name alias>...</lvar>
            # First ID is treated as a redundant hint, second as the alias.
            alias = self.current_token().value
            self.advance()
            extra_id = first_id
            model = None
            field = None
            has_explicit_alias = True
            is_raw = True
        else:
            alias = first_id
            model = None
            field = None
            has_explicit_alias = False
            is_raw = True

        self.expect(TokenType.GT)
        self.skip_newlines()

        if not self.source_text:
            raise ParseError(
                "Parser requires source_text for content extraction",
                self.current_token(),
            )

        if is_raw:
            if extra_id:
                pattern = rf"<lvar\s+{re.escape(extra_id)}\s+{re.escape(alias)}\s*>(.*?)</lvar>"
            else:
                pattern = rf"<lvar\s+{re.escape(alias)}\s*>(.*?)</lvar>"
        else:
            if has_explicit_alias:
                pattern = rf"<lvar\s+{re.escape(model)}\.{re.escape(field)}\s+{re.escape(alias)}\s*>(.*?)</lvar>"
            else:
                pattern = (
                    rf"<lvar\s+{re.escape(model)}\.{re.escape(field)}\s*>(.*?)</lvar>"
                )

        match = re.search(pattern, self.source_text, re.DOTALL)
        if not match:
            if "</lvar>" not in self.source_text:
                raise ParseError(
                    "Unclosed lvar tag - missing </lvar>", self.current_token()
                )
            raise ParseError(
                f"Could not extract lvar content with pattern: {pattern}",
                self.current_token(),
            )

        content = match.group(1).strip()

        while not self.match(TokenType.LVAR_CLOSE):
            if self.match(TokenType.EOF):
                raise ParseError(
                    "Unclosed lvar tag - missing </lvar>", self.current_token()
                )
            self.advance()

        self.expect(TokenType.LVAR_CLOSE)

        if is_raw:
            return RLvar(alias=alias, content=content)
        return Lvar(model=model, field=field, alias=alias, content=content)

    def parse_lact(self) -> Lact:
        self.expect(TokenType.LACT_OPEN)
        self.skip_newlines()

        first_id = self.expect(TokenType.ID).value
        has_explicit_alias = False
        extra_id: str | None = None

        if self.match(TokenType.DOT):
            self.advance()
            field = self.expect(TokenType.ID).value
            model = first_id

            if self.match(TokenType.ID):
                alias = self.current_token().value
                self.advance()
                has_explicit_alias = True
            else:
                alias = field
                has_explicit_alias = False
        elif self.match(TokenType.ID):
            # Forgiving 2-ID no-dot pattern: <lact name alias>...</lact>
            # First ID is a redundant hint (often a function name); second is the alias.
            alias = self.current_token().value
            self.advance()
            extra_id = first_id
            model = None
            field = None
            has_explicit_alias = True
        else:
            model = None
            field = None
            alias = first_id
            has_explicit_alias = True

        self.expect(TokenType.GT)
        self.skip_newlines()

        if not self.source_text:
            raise ParseError(
                "Parser requires source_text for call extraction", self.current_token()
            )

        if model:
            if has_explicit_alias:
                pattern = rf"<lact\s+{re.escape(model)}\.{re.escape(field)}\s+{re.escape(alias)}\s*>(.*?)</lact>"
            else:
                pattern = (
                    rf"<lact\s+{re.escape(model)}\.{re.escape(field)}\s*>(.*?)</lact>"
                )
        elif extra_id:
            pattern = (
                rf"<lact\s+{re.escape(extra_id)}\s+{re.escape(alias)}\s*>(.*?)</lact>"
            )
        else:
            pattern = rf"<lact\s+{re.escape(alias)}\s*>(.*?)</lact>"

        match = re.search(pattern, self.source_text, re.DOTALL)
        if not match:
            if "</lact>" not in self.source_text:
                raise ParseError(
                    "Unclosed lact tag - missing </lact>", self.current_token()
                )
            raise ParseError(
                f"Could not extract lact call with pattern: {pattern}",
                self.current_token(),
            )

        call = match.group(1).strip()

        while not self.match(TokenType.LACT_CLOSE):
            if self.match(TokenType.EOF):
                raise ParseError(
                    "Unclosed lact tag - missing </lact>", self.current_token()
                )
            self.advance()

        self.expect(TokenType.LACT_CLOSE)

        if alias in PYTHON_RESERVED and alias not in _warned_action_names:
            _warned_action_names.add(alias)
            warnings.warn(
                f"Action name '{alias}' is a Python reserved keyword or builtin.",
                UserWarning,
                stacklevel=2,
            )

        return Lact(model=model, field=field, alias=alias, call=call)

    def parse_out_block(self) -> OutBlock:
        self.expect(TokenType.OUT_OPEN)
        self.skip_newlines()

        fields: dict[str, list[str] | str | int | float | bool] = {}

        while not self.match(TokenType.OUT_CLOSE, TokenType.EOF):
            self.skip_newlines()
            if self.match(TokenType.OUT_CLOSE, TokenType.EOF):
                break

            if not self.match(TokenType.ID):
                self.advance()
                continue

            field_name = self.current_token().value
            self.advance()
            self.skip_newlines()
            self.expect(TokenType.COLON)
            self.skip_newlines()

            if self.match(TokenType.LBRACKET):
                self.advance()
                self.skip_newlines()
                refs: list[str] = []
                while not self.match(TokenType.RBRACKET, TokenType.EOF):
                    self.skip_newlines()
                    if self.match(TokenType.RBRACKET, TokenType.EOF):
                        break
                    if self.match(TokenType.ID):
                        name = self.current_token().value
                        self.advance()
                        while self.match(TokenType.DOT):
                            self.advance()
                            if not self.match(TokenType.ID):
                                break
                            name = f"{name}.{self.current_token().value}"
                            self.advance()
                        refs.append(name)
                    elif self.match(TokenType.STR) or self.match(TokenType.NUM):
                        refs.append(self.current_token().value)
                        self.advance()
                    else:
                        self.advance()
                    self.skip_newlines()
                    if self.match(TokenType.COMMA):
                        self.advance()
                if self.match(TokenType.RBRACKET):
                    self.advance()
                fields[field_name] = refs

            elif self.match(TokenType.STR):
                fields[field_name] = self.current_token().value
                self.advance()

            elif self.match(TokenType.NUM):
                num_str = self.current_token().value
                self.advance()
                fields[field_name] = float(num_str) if "." in num_str else int(num_str)

            elif self.match(TokenType.ID):
                value = self.current_token().value
                self.advance()
                if value.lower() == "true":
                    fields[field_name] = True
                elif value.lower() == "false":
                    fields[field_name] = False
                else:
                    while self.match(TokenType.DOT):
                        self.advance()
                        if not self.match(TokenType.ID):
                            break
                        value = f"{value}.{self.current_token().value}"
                        self.advance()
                    fields[field_name] = [value]
            else:
                self.advance()

            self.skip_newlines()
            if self.match(TokenType.COMMA):
                self.advance()

        if self.match(TokenType.OUT_CLOSE):
            self.advance()

        return OutBlock(fields=fields)


def parse_value(value_str: Any) -> Any:
    if not isinstance(value_str, str):
        return value_str
    value_str = value_str.strip()
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null":
        return None
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str


__all__ = ("ParseError", "Parser")
