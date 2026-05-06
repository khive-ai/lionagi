# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import warnings

import pytest

from lionagi.lndl.ast import Lact, Lvar, OutBlock, Program, RLvar
from lionagi.lndl.lexer import Lexer
from lionagi.lndl.parser import ParseError, Parser, parse_value


def parse(text: str) -> Program:
    tokens = Lexer(text).tokenize()
    return Parser(tokens, source_text=text).parse()


class TestParseValue:
    def test_true(self):
        assert parse_value("true") is True
        assert parse_value("True") is True

    def test_false(self):
        assert parse_value("false") is False
        assert parse_value("False") is False

    def test_null(self):
        assert parse_value("null") is None

    def test_integer(self):
        assert parse_value("42") == 42
        assert isinstance(parse_value("42"), int)

    def test_float(self):
        assert parse_value("3.14") == pytest.approx(3.14)

    def test_string_passthrough(self):
        assert parse_value("hello world") == "hello world"

    def test_list_literal(self):
        result = parse_value("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_dict_literal(self):
        result = parse_value("{'a': 1}")
        assert result == {"a": 1}

    def test_non_string_passthrough(self):
        assert parse_value(42) == 42
        assert parse_value(3.14) == pytest.approx(3.14)
        assert parse_value(True) is True
        assert parse_value(None) is None


class TestParserRequiresSourceText:
    def test_parse_raises_without_source_text(self):
        tokens = Lexer("<lvar x>val</lvar>").tokenize()
        parser = Parser(tokens, source_text=None)
        with pytest.raises(ParseError):
            parser.parse()


class TestParseLvar:
    def test_simple_raw_lvar(self):
        text = "<lvar x>hello world</lvar>"
        prog = parse(text)
        assert len(prog.lvars) == 1
        lvar = prog.lvars[0]
        assert isinstance(lvar, RLvar)
        assert lvar.alias == "x"
        assert lvar.content == "hello world"

    def test_namespaced_lvar(self):
        text = "<lvar Report.title t>My Title</lvar>"
        prog = parse(text)
        assert len(prog.lvars) == 1
        lvar = prog.lvars[0]
        assert isinstance(lvar, Lvar)
        assert lvar.model == "Report"
        assert lvar.field == "title"
        assert lvar.alias == "t"
        assert lvar.content == "My Title"

    def test_namespaced_lvar_no_alias(self):
        text = "<lvar Report.title>My Title</lvar>"
        prog = parse(text)
        lvar = prog.lvars[0]
        assert isinstance(lvar, Lvar)
        assert lvar.field == "title"
        assert lvar.alias == "title"  # alias defaults to field name

    def test_two_id_no_dot_lvar(self):
        text = "<lvar hint alias>some value</lvar>"
        prog = parse(text)
        lvar = prog.lvars[0]
        assert isinstance(lvar, RLvar)
        assert lvar.alias == "alias"
        assert lvar.content == "some value"

    def test_lvar_multiline_content(self):
        text = "<lvar x>line one\nline two\nline three</lvar>"
        prog = parse(text)
        assert "line one" in prog.lvars[0].content
        assert "line three" in prog.lvars[0].content

    def test_multiple_lvars(self):
        text = (
            "<lvar Report.title t>Title</lvar>\n"
            "<lvar Report.summary s>Summary</lvar>"
        )
        prog = parse(text)
        assert len(prog.lvars) == 2

    def test_lvar_duplicate_alias_raises(self):
        text = (
            "<lvar Report.title t>Title</lvar>\n"
            "<lvar Report.summary t>Summary</lvar>"
        )
        with pytest.raises(ParseError, match="Duplicate alias"):
            parse(text)

    def test_lvar_unclosed_tag_raises(self):
        # No </lvar> present
        text = "<lvar x>value without close"
        with pytest.raises(ParseError):
            parse(text)


class TestParseLact:
    def test_direct_lact(self):
        text = "<lact data>fetch(url='http://x')</lact>"
        prog = parse(text)
        assert len(prog.lacts) == 1
        lact = prog.lacts[0]
        assert isinstance(lact, Lact)
        assert lact.alias == "data"
        assert lact.model is None
        assert lact.field is None
        assert "fetch" in lact.call

    def test_namespaced_lact(self):
        text = "<lact Report.summary s>summarize(text='hello')</lact>"
        prog = parse(text)
        lact = prog.lacts[0]
        assert lact.model == "Report"
        assert lact.field == "summary"
        assert lact.alias == "s"

    def test_namespaced_lact_no_alias(self):
        text = "<lact Report.summary>summarize(text='hi')</lact>"
        prog = parse(text)
        lact = prog.lacts[0]
        assert lact.field == "summary"
        assert lact.alias == "summary"  # alias defaults to field

    def test_two_id_lact(self):
        text = "<lact fn alias>fetch(url='x')</lact>"
        prog = parse(text)
        lact = prog.lacts[0]
        assert lact.alias == "alias"
        assert lact.model is None

    def test_lact_python_reserved_warns(self):
        text = "<lact print>fetch(url='x')</lact>"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prog = parse(text)
            assert any("reserved" in str(warning.message).lower() for warning in w)

    def test_lact_duplicate_alias_raises(self):
        text = "<lvar x>val</lvar>\n" "<lact x>fn(a='b')</lact>"
        with pytest.raises(ParseError, match="Duplicate alias"):
            parse(text)

    def test_lact_unclosed_raises(self):
        text = "<lact data>fetch()"
        with pytest.raises(ParseError):
            parse(text)


class TestParseOutBlock:
    def test_empty_out_block(self):
        text = "OUT{}"
        prog = parse(text)
        assert prog.out_block is not None
        assert prog.out_block.fields == {}

    def test_out_block_with_array(self):
        text = "OUT{report: [t, s]}"
        prog = parse(text)
        assert prog.out_block.fields["report"] == ["t", "s"]

    def test_out_block_with_int(self):
        text = "OUT{count: 5}"
        prog = parse(text)
        assert prog.out_block.fields["count"] == 5
        assert isinstance(prog.out_block.fields["count"], int)

    def test_out_block_with_float(self):
        text = "OUT{score: 0.85}"
        prog = parse(text)
        assert prog.out_block.fields["score"] == pytest.approx(0.85)

    def test_out_block_with_string(self):
        text = 'OUT{name: "hello"}'
        prog = parse(text)
        assert prog.out_block.fields["name"] == "hello"

    def test_out_block_with_bool_true(self):
        text = "OUT{flag: true}"
        prog = parse(text)
        assert prog.out_block.fields["flag"] is True

    def test_out_block_with_bool_false(self):
        text = "OUT{flag: false}"
        prog = parse(text)
        assert prog.out_block.fields["flag"] is False

    def test_out_block_multiple_fields(self):
        text = "OUT{report: [t, s], score: 0.9}"
        prog = parse(text)
        assert "report" in prog.out_block.fields
        assert "score" in prog.out_block.fields

    def test_out_block_single_ref_as_list(self):
        text = "OUT{x: y}"
        prog = parse(text)
        # Single ID value becomes a list with one item
        assert prog.out_block.fields["x"] == ["y"]

    def test_out_block_dotted_ref(self):
        text = "OUT{x: [note.draft]}"
        prog = parse(text)
        assert prog.out_block.fields["x"] == ["note.draft"]

    def test_no_out_block_returns_none(self):
        text = "<lvar x>val</lvar>"
        prog = parse(text)
        assert prog.out_block is None


class TestFullProgram:
    def test_full_program_with_lvars_and_out(self):
        text = (
            "<lvar Report.title t>Title Text</lvar>\n"
            "<lvar Report.summary s>Summary Text</lvar>\n"
            "OUT{report: [t, s]}"
        )
        prog = parse(text)
        assert len(prog.lvars) == 2
        assert len(prog.lacts) == 0
        assert prog.out_block is not None
        assert prog.out_block.fields["report"] == ["t", "s"]

    def test_full_program_with_lact(self):
        text = (
            "<lvar Report.title t>Title</lvar>\n"
            "<lact Report.summary s>summarize(text='hi')</lact>\n"
            "OUT{report: [t, s]}"
        )
        prog = parse(text)
        assert len(prog.lvars) == 1
        assert len(prog.lacts) == 1
        assert prog.out_block is not None

    def test_program_ignores_prose(self):
        text = (
            "Let me think about this...\n"
            "<lvar x>value</lvar>\n"
            "More thinking here.\n"
            "OUT{x: [x]}"
        )
        prog = parse(text)
        assert len(prog.lvars) == 1
        assert prog.out_block is not None

    def test_parse_error_has_message_and_token(self):
        tokens = Lexer("<lvar x>val</lvar>").tokenize()
        parser = Parser(tokens, source_text=None)
        try:
            parser.parse()
            assert False, "Should have raised"
        except ParseError as e:
            assert e.message is not None
            assert e.token is not None
