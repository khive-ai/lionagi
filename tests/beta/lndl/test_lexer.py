# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from lionagi.beta.lndl.lexer import Lexer, Token, TokenType


def tokenize(text: str) -> list[Token]:
    return Lexer(text).tokenize()


def token_types(text: str) -> list[TokenType]:
    return [t.type for t in tokenize(text)]


def token_values(text: str) -> list[str]:
    return [t.value for t in tokenize(text)]


class TestTokenType:
    def test_all_token_types_exist(self):
        expected = [
            "LVAR_OPEN",
            "LVAR_CLOSE",
            "LACT_OPEN",
            "LACT_CLOSE",
            "OUT_OPEN",
            "OUT_CLOSE",
            "ID",
            "NUM",
            "STR",
            "DOT",
            "COMMA",
            "COLON",
            "LBRACKET",
            "RBRACKET",
            "LPAREN",
            "RPAREN",
            "GT",
            "NEWLINE",
            "EOF",
        ]
        for name in expected:
            assert hasattr(TokenType, name)


class TestLexerBasics:
    def test_empty_string(self):
        tokens = tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_eof_always_at_end(self):
        tokens = tokenize("abc")
        assert tokens[-1].type == TokenType.EOF

    def test_identifier(self):
        tokens = tokenize("hello")
        assert tokens[0].type == TokenType.ID
        assert tokens[0].value == "hello"

    def test_identifier_with_underscore(self):
        tokens = tokenize("my_var")
        assert tokens[0].type == TokenType.ID
        assert tokens[0].value == "my_var"

    def test_number_integer(self):
        tokens = tokenize("42")
        assert tokens[0].type == TokenType.NUM
        assert tokens[0].value == "42"

    def test_number_float(self):
        tokens = tokenize("3.14")
        assert tokens[0].type == TokenType.NUM
        assert tokens[0].value == "3.14"

    def test_dot(self):
        tokens = tokenize(".")
        assert tokens[0].type == TokenType.DOT

    def test_comma(self):
        tokens = tokenize(",")
        assert tokens[0].type == TokenType.COMMA

    def test_colon(self):
        tokens = tokenize(":")
        assert tokens[0].type == TokenType.COLON

    def test_lbracket(self):
        tokens = tokenize("[")
        assert tokens[0].type == TokenType.LBRACKET

    def test_rbracket(self):
        tokens = tokenize("]")
        assert tokens[0].type == TokenType.RBRACKET

    def test_lparen(self):
        tokens = tokenize("(")
        assert tokens[0].type == TokenType.LPAREN

    def test_rparen(self):
        tokens = tokenize(")")
        assert tokens[0].type == TokenType.RPAREN

    def test_gt(self):
        tokens = tokenize(">")
        assert tokens[0].type == TokenType.GT

    def test_newline(self):
        tokens = tokenize("\n")
        assert tokens[0].type == TokenType.NEWLINE

    def test_whitespace_skipped(self):
        tokens = tokenize("   \t  abc")
        assert tokens[0].type == TokenType.ID
        assert tokens[0].value == "abc"


class TestLvarTokens:
    def test_lvar_open(self):
        tokens = tokenize("<lvar")
        assert tokens[0].type == TokenType.LVAR_OPEN
        assert tokens[0].value == "<lvar"

    def test_lvar_close(self):
        tokens = tokenize("</lvar>")
        assert tokens[0].type == TokenType.LVAR_CLOSE
        assert tokens[0].value == "</lvar>"

    def test_lact_open(self):
        tokens = tokenize("<lact")
        assert tokens[0].type == TokenType.LACT_OPEN

    def test_lact_close(self):
        tokens = tokenize("</lact>")
        assert tokens[0].type == TokenType.LACT_CLOSE

    def test_full_lvar_tag(self):
        tokens = tokenize("<lvar Report.title t>")
        types = [t.type for t in tokens[:-1]]  # exclude EOF
        assert TokenType.LVAR_OPEN in types
        assert TokenType.ID in types
        assert TokenType.DOT in types
        assert TokenType.GT in types

    def test_lvar_simple_alias(self):
        tokens = tokenize("<lvar x>")
        assert tokens[0].type == TokenType.LVAR_OPEN
        assert tokens[1].type == TokenType.ID
        assert tokens[1].value == "x"
        assert tokens[2].type == TokenType.GT


class TestOutBlock:
    def test_out_open(self):
        tokens = tokenize("OUT{")
        assert tokens[0].type == TokenType.OUT_OPEN
        assert tokens[0].value == "OUT{"

    def test_out_close(self):
        tokens = tokenize("}")
        # OUT_CLOSE requires being inside OUT block
        # outside of OUT block, } may be skipped or OUT_CLOSE
        # The lexer only sets in_out_block flag on OUT_OPEN
        # So standalone } outside gets OUT_CLOSE token
        assert tokens[0].type == TokenType.OUT_CLOSE

    def test_out_block_sequence(self):
        text = "OUT{score: 0.9}"
        tokens = tokenize(text)
        types = [t.type for t in tokens]
        assert TokenType.OUT_OPEN in types
        assert TokenType.ID in types
        assert TokenType.COLON in types
        assert TokenType.NUM in types
        assert TokenType.OUT_CLOSE in types

    def test_string_in_out_block(self):
        text = 'OUT{name: "hello"}'
        tokens = tokenize(text)
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "hello"

    def test_negative_number_in_out_block(self):
        text = "OUT{score: -0.5}"
        tokens = tokenize(text)
        num_tokens = [t for t in tokens if t.type == TokenType.NUM]
        assert len(num_tokens) == 1
        assert num_tokens[0].value == "-0.5"

    def test_boolean_true_in_out_block(self):
        text = "OUT{flag: true}"
        tokens = tokenize(text)
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        # 'flag' and 'true' are both IDs at lexer level
        assert any(t.value == "true" for t in id_tokens)

    def test_array_in_out_block(self):
        text = "OUT{report: [t, s]}"
        tokens = tokenize(text)
        types = [t.type for t in tokens]
        assert TokenType.LBRACKET in types
        assert TokenType.RBRACKET in types
        assert TokenType.COMMA in types

    def test_single_quote_string_in_out_block(self):
        text = "OUT{name: 'world'}"
        tokens = tokenize(text)
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert str_tokens[0].value == "world"


class TestLineTracking:
    def test_line_starts_at_1(self):
        tokens = tokenize("abc")
        assert tokens[0].line == 1

    def test_column_starts_at_1(self):
        tokens = tokenize("abc")
        assert tokens[0].column == 1

    def test_newline_increments_line(self):
        tokens = tokenize("a\nb")
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert id_tokens[0].line == 1
        assert id_tokens[1].line == 2


class TestReadString:
    def test_escape_newline(self):
        text = 'OUT{x: "a\\nb"}'
        tokens = tokenize(text)
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert len(str_tokens) == 1
        assert "\n" in str_tokens[0].value

    def test_escape_tab(self):
        text = 'OUT{x: "a\\tb"}'
        tokens = tokenize(text)
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert "\t" in str_tokens[0].value

    def test_escape_backslash(self):
        text = r'OUT{x: "a\\b"}'
        tokens = tokenize(text)
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert "\\" in str_tokens[0].value

    def test_escape_double_quote(self):
        text = r'OUT{x: "say \"hi\""}'
        tokens = tokenize(text)
        str_tokens = [t for t in tokens if t.type == TokenType.STR]
        assert '"' in str_tokens[0].value


class TestComplexInput:
    def test_full_lndl_response(self):
        text = (
            "<lvar Report.title t>My Title</lvar>\n"
            "<lvar Report.summary s>Summary text</lvar>\n"
            "OUT{report: [t, s]}"
        )
        tokens = tokenize(text)
        assert tokens[-1].type == TokenType.EOF
        lvar_opens = [t for t in tokens if t.type == TokenType.LVAR_OPEN]
        assert len(lvar_opens) == 2

    def test_lact_tag_sequence(self):
        text = "<lact Report.summary s>summarize(text='hi')</lact>"
        tokens = tokenize(text)
        assert any(t.type == TokenType.LACT_OPEN for t in tokens)
        assert any(t.type == TokenType.LACT_CLOSE for t in tokens)

    def test_unknown_chars_skipped(self):
        # Characters not matching any rule are silently skipped (e.g. '@')
        text = "@ abc"
        tokens = tokenize(text)
        id_tokens = [t for t in tokens if t.type == TokenType.ID]
        assert any(t.value == "abc" for t in id_tokens)

    def test_token_has_required_fields(self):
        tokens = tokenize("abc 123")
        for t in tokens:
            assert hasattr(t, "type")
            assert hasattr(t, "value")
            assert hasattr(t, "line")
            assert hasattr(t, "column")
