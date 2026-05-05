# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel

from lionagi.beta.lndl.errors import (
    AmbiguousMatchError,
    MissingFieldError,
    MissingOutBlockError,
)
from lionagi.beta.lndl.fuzzy import normalize_lndl_text, parse_lndl_fuzzy
from lionagi.beta.lndl.types import ActionCall, LNDLOutput
from lionagi.ln.types import Operable, Spec


class Report(BaseModel):
    title: str
    summary: str = ""
    score: float = 0.0


class Analysis(BaseModel):
    file_name: str
    line_count: int = 0


def make_operable(*specs):
    return Operable(list(specs))


def make_report_operable():
    return make_operable(Spec(Report, name="report"))


class TestNormalizeLndlText:
    def test_curly_brace_lvar(self):
        text = "{lvar x}value{/lvar}"
        normalized = normalize_lndl_text(text)
        assert "<lvar" in normalized
        assert "{lvar" not in normalized

    def test_curly_brace_lact(self):
        text = "{lact x}fn(a='b'){/lact}"
        normalized = normalize_lndl_text(text)
        assert "<lact" in normalized

    def test_xml_attributes_cleaned(self):
        text = '<lvar name="title" type="str" t>'
        normalized = normalize_lndl_text(text)
        # XML attrs should be stripped; 't' alias should remain
        assert 'name="title"' not in normalized
        assert "t" in normalized

    def test_note_namespace_lowercased(self):
        text = "<lvar Note.draft d>some text</lvar>"
        normalized = normalize_lndl_text(text)
        assert "note.draft" in normalized or "<lvar note." in normalized

    def test_passthrough_normal_text(self):
        text = "<lvar Report.title t>Title</lvar>"
        normalized = normalize_lndl_text(text)
        assert "Report.title" in normalized
        assert "<lvar" in normalized

    def test_empty_string(self):
        result = normalize_lndl_text("")
        assert result == ""


class TestParseLndlFuzzy:
    def test_exact_match(self):
        text = (
            "<lvar Report.title t>My Title</lvar>\n"
            "<lvar Report.summary s>My Summary</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        result = parse_lndl_fuzzy(text, operable)
        assert isinstance(result, LNDLOutput)
        report = result["report"]
        assert isinstance(report, Report)
        assert report.title == "My Title"
        assert report.summary == "My Summary"

    def test_typo_correction_field(self):
        # 'titlE' → 'title' via fuzzy
        text = (
            "<lvar Report.titlE t>My Title</lvar>\n"
            "<lvar Report.summary s>Summary</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        result = parse_lndl_fuzzy(text, operable, threshold=0.80)
        report = result["report"]
        assert isinstance(report, Report)
        assert report.title == "My Title"

    def test_missing_out_block_raises(self):
        text = "<lvar Report.title t>Title</lvar>"
        operable = make_report_operable()
        with pytest.raises(MissingOutBlockError):
            parse_lndl_fuzzy(text, operable)

    def test_scalar_spec(self):
        text = "<lvar score s>0.9</lvar>\nOUT{score: [s]}"
        operable = make_operable(Spec(float, name="score"))
        result = parse_lndl_fuzzy(text, operable)
        assert result["score"] == pytest.approx(0.9)

    def test_strict_mode_rejects_unknown_model(self):
        text = (
            "<lvar WrongModel.title t>Title</lvar>\n"
            "<lvar Report.summary s>Summary</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        with pytest.raises(Exception):
            parse_lndl_fuzzy(text, operable, threshold=1.0)

    def test_strict_mode_rejects_unknown_field(self):
        text = "<lvar Report.nonexistent t>Title</lvar>\n" "OUT{report: [t]}"
        operable = make_report_operable()
        with pytest.raises(Exception):
            parse_lndl_fuzzy(text, operable, threshold=1.0)

    def test_strict_mode_exact_match_passes(self):
        text = (
            "<lvar Report.title t>Title</lvar>\n"
            "<lvar Report.summary s>Summary</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        result = parse_lndl_fuzzy(text, operable, threshold=1.0)
        assert result["report"].title == "Title"

    def test_capabilities_filter(self):
        text = (
            "<lvar Report.title t>Title</lvar>\n"
            "<lvar Report.summary s>Summary</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_operable(
            Spec(Report, name="report"),
            Spec(float, name="score"),
        )
        result = parse_lndl_fuzzy(text, operable, capabilities=["report"])
        assert "report" in result.fields

    def test_capabilities_rejects_non_allowed(self):
        text = "OUT{score: 0.9}"
        operable = make_operable(Spec(Report, name="report"), Spec(float, name="score"))
        with pytest.raises(Exception):
            parse_lndl_fuzzy(text, operable, capabilities=["report"])

    def test_action_call_in_model_field(self):
        text = (
            "<lvar Report.title t>Title</lvar>\n"
            "<lact Report.summary s>summarize(text='hi')</lact>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        result = parse_lndl_fuzzy(text, operable)
        report = result["report"]
        assert report.title == "Title"
        assert isinstance(report.summary, ActionCall)

    def test_per_threshold_overrides(self):
        text = (
            "<lvar Report.title t>Title</lvar>\n"
            "<lvar Report.summary s>Summary</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        result = parse_lndl_fuzzy(
            text,
            operable,
            threshold_field=0.8,
            threshold_model=0.9,
            threshold_spec=0.85,
            threshold_lvar=0.8,
        )
        assert result["report"].title == "Title"

    def test_rlvar_in_model_positional(self):
        # Raw lvar (no model.field) assigned positionally to first unfilled field
        text = (
            "<lvar t>Title Text</lvar>\n"
            "<lvar Report.summary s>Summary</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        result = parse_lndl_fuzzy(text, operable)
        report = result["report"]
        assert isinstance(report, Report)
        # t assigned positionally to 'title' (first unfilled)
        assert report.title == "Title Text"

    def test_result_is_lndl_output(self):
        text = (
            "<lvar Report.title t>T</lvar>\n"
            "<lvar Report.summary s>S</lvar>\n"
            "OUT{report: [t, s]}"
        )
        operable = make_report_operable()
        result = parse_lndl_fuzzy(text, operable)
        assert isinstance(result, LNDLOutput)
        assert result.raw_out_block is not None
