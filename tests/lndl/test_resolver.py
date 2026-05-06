# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel

from lionagi.lndl.errors import MissingFieldError, MissingOutBlockError
from lionagi.lndl.resolver import (
    NOTE_NAMESPACE,
    _is_note_ref,
    _normalize_note_lvars,
    _read_note,
    parse_lndl,
    resolve_references_prefixed,
)
from lionagi.lndl.types import (
    ActionCall,
    LactMetadata,
    LNDLOutput,
    LvarMetadata,
    RLvarMetadata,
)
from lionagi.ln.types import Operable, Spec


class Report(BaseModel):
    title: str
    summary: str = ""


class Analysis(BaseModel):
    file_name: str
    line_count: int = 0


def make_operable(*specs):
    return Operable(list(specs))


def make_report_operable():
    return make_operable(Spec(Report, name="report"))


class TestNoteRefHelpers:
    def test_is_note_ref_true(self):
        assert _is_note_ref("note.draft") is True
        assert _is_note_ref("note.x.y") is True

    def test_is_note_ref_false(self):
        assert _is_note_ref("note") is False
        assert _is_note_ref("x") is False
        assert _is_note_ref("notepad") is False

    def test_read_note_simple(self):
        scratchpad = {"draft": "my draft text"}
        result = _read_note("note.draft", scratchpad)
        assert result == "my draft text"

    def test_read_note_no_scratchpad_raises(self):
        with pytest.raises(ValueError, match="no scratchpad"):
            _read_note("note.x", None)

    def test_read_note_missing_key_raises(self):
        with pytest.raises(ValueError, match="Scratchpad miss"):
            _read_note("note.missing", {"other": "val"})

    def test_read_note_dict_returns_yaml_like(self):
        scratchpad = {"data": {"key": "val"}}
        result = _read_note("note.data", scratchpad)
        assert isinstance(result, str)

    def test_read_note_non_string_converts(self):
        scratchpad = {"count": 42}
        result = _read_note("note.count", scratchpad)
        assert "42" in result


class TestNormalizeNoteLvars:
    def test_converts_note_namespace_lvar_to_rlvar(self):
        lvars = {
            "x": LvarMetadata(
                model="note", field="draft", local_name="x", value="text"
            ),
        }
        result = _normalize_note_lvars(lvars)
        assert isinstance(result["x"], RLvarMetadata)
        assert result["x"].local_name == "x"
        assert result["x"].value == "text"

    def test_preserves_non_note_lvars(self):
        lvars = {
            "t": LvarMetadata(
                model="Report", field="title", local_name="t", value="My Title"
            ),
        }
        result = _normalize_note_lvars(lvars)
        assert isinstance(result["t"], LvarMetadata)

    def test_preserves_rlvars(self):
        lvars = {
            "r": RLvarMetadata(local_name="r", value="raw"),
        }
        result = _normalize_note_lvars(lvars)
        assert isinstance(result["r"], RLvarMetadata)


class TestResolveReferencesPrefixed:
    def test_scalar_string_literal(self):
        operable = make_operable(Spec(str, name="label"))
        out_fields = {"label": "hello"}
        lvars = {}
        lacts = {}
        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)
        assert isinstance(result, LNDLOutput)
        assert result["label"] == "hello"

    def test_scalar_from_lvar(self):
        operable = make_operable(Spec(str, name="label"))
        lvars = {"x": RLvarMetadata(local_name="x", value="hello")}
        out_fields = {"label": ["x"]}
        result = resolve_references_prefixed(out_fields, lvars, {}, operable)
        assert result["label"] == "hello"

    def test_scalar_float(self):
        operable = make_operable(Spec(float, name="score"))
        out_fields = {"score": "0.85"}
        result = resolve_references_prefixed(out_fields, {}, {}, operable)
        assert result["score"] == pytest.approx(0.85)

    def test_scalar_int(self):
        operable = make_operable(Spec(int, name="count"))
        out_fields = {"count": 5}
        result = resolve_references_prefixed(out_fields, {}, {}, operable)
        assert result["count"] == 5

    def test_pydantic_model_from_lvars(self):
        operable = make_report_operable()
        lvars = {
            "t": LvarMetadata(
                model="Report", field="title", local_name="t", value="My Title"
            ),
            "s": LvarMetadata(
                model="Report", field="summary", local_name="s", value="A summary"
            ),
        }
        out_fields = {"report": ["t", "s"]}
        result = resolve_references_prefixed(out_fields, lvars, {}, operable)
        report = result["report"]
        assert isinstance(report, Report)
        assert report.title == "My Title"
        assert report.summary == "A summary"

    def test_name_collision_raises(self):
        operable = make_operable(Spec(str, name="label"))
        lvars = {"x": RLvarMetadata(local_name="x", value="val")}
        lacts = {
            "x": LactMetadata(model=None, field=None, local_name="x", call="fn(a='b')")
        }
        with pytest.raises(ValueError, match="Name collision"):
            resolve_references_prefixed({"label": ["x"]}, lvars, lacts, operable)

    def test_action_call_in_scalar(self):
        operable = make_operable(Spec(str, name="data"))
        lacts = {
            "d": LactMetadata(
                model=None, field=None, local_name="d", call="fetch(url='http://x')"
            )
        }
        out_fields = {"data": ["d"]}
        result = resolve_references_prefixed(out_fields, {}, lacts, operable)
        assert isinstance(result["data"], ActionCall)
        assert result["data"].function == "fetch"

    def test_action_call_in_model(self):
        operable = make_report_operable()
        lvars = {
            "t": LvarMetadata(
                model="Report", field="title", local_name="t", value="Title"
            ),
        }
        lacts = {
            "s": LactMetadata(
                model="Report",
                field="summary",
                local_name="s",
                call="summarize(text='hi')",
            )
        }
        out_fields = {"report": ["t", "s"]}
        result = resolve_references_prefixed(out_fields, lvars, lacts, operable)
        report = result["report"]
        assert report.title == "Title"
        assert isinstance(report.summary, ActionCall)

    def test_missing_required_spec_raises(self):
        operable = make_report_operable()
        # report is required by default, but not provided
        with pytest.raises(Exception):
            resolve_references_prefixed({}, {}, {}, operable)

    def test_scalar_with_inline_literal(self):
        operable = make_operable(Spec(str, name="label"))
        # OUT{label: hello} where hello is not a var/lact → inline literal
        out_fields = {"label": ["hello"]}
        result = resolve_references_prefixed(out_fields, {}, {}, operable)
        assert result["label"] == "hello"

    def test_lndl_output_actions_populated(self):
        operable = make_operable(Spec(str, name="data"))
        lacts = {
            "d": LactMetadata(
                model=None, field=None, local_name="d", call="fetch(url='http://x')"
            )
        }
        result = resolve_references_prefixed({"data": ["d"]}, {}, lacts, operable)
        assert "d" in result.actions


class TestParseLndl:
    def test_simple_scalar(self):
        text = "<lvar x>hello world</lvar>\nOUT{label: [x]}"
        operable = make_operable(Spec(str, name="label"))
        result = parse_lndl(text, operable)
        assert result["label"] == "hello world"

    def test_pydantic_model_from_parse(self):
        text = (
            "<lvar Report.title t>My Title</lvar>\n"
            "<lvar Report.summary s>My Summary</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        result = parse_lndl(text, operable)
        report = result["report"]
        assert isinstance(report, Report)
        assert report.title == "My Title"
        assert report.summary == "My Summary"

    def test_missing_out_block_raises(self):
        text = "<lvar x>val</lvar>"
        operable = make_operable(Spec(str, name="x"))
        with pytest.raises(MissingOutBlockError):
            parse_lndl(text, operable)

    def test_multiple_scalars(self):
        text = (
            "<lvar score s>0.9</lvar>\n"
            "<lvar label l>good</lvar>\n"
            "OUT{score: [s], label: [l]}"
        )
        operable = make_operable(Spec(float, name="score"), Spec(str, name="label"))
        result = parse_lndl(text, operable)
        assert result["score"] == pytest.approx(0.9)
        assert result["label"] == "good"

    def test_lact_in_parse_lndl(self):
        text = "<lact data>fetch(url='http://x')</lact>\n" "OUT{label: [data]}"
        operable = make_operable(Spec(str, name="label"))
        result = parse_lndl(text, operable)
        assert isinstance(result["label"], ActionCall)
        assert result["label"].function == "fetch"
