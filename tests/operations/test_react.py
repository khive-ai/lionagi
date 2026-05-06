# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/react.py — ReActParams, ReActAnalysis, helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lionagi.operations.generate import GenerateParams
from lionagi.operations.react import (
    Analysis,
    PlannedAction,
    ReActAnalysis,
    ReActParams,
    _needs_extension,
)
from lionagi.ln.types import Operable, Spec

# ---------------------------------------------------------------------------
# PlannedAction
# ---------------------------------------------------------------------------


class TestPlannedAction:
    def test_defaults(self):
        p = PlannedAction()
        assert p.action_type is None
        assert p.description is None

    def test_with_values(self):
        p = PlannedAction(action_type="search", description="Search the web")
        assert p.action_type == "search"


# ---------------------------------------------------------------------------
# ReActAnalysis
# ---------------------------------------------------------------------------


class TestReActAnalysis:
    def test_basic_construction(self):
        a = ReActAnalysis(analysis="thinking...")
        assert a.analysis == "thinking..."
        assert a.extension_needed is False

    def test_extension_needed_true(self):
        a = ReActAnalysis(analysis="not done yet", extension_needed=True)
        assert a.extension_needed is True

    def test_class_vars_present(self):
        assert "{max_rounds}" in ReActAnalysis.FIRST_ROUND_PROMPT
        assert "{remaining}" in ReActAnalysis.CONTINUE_PROMPT
        assert "{instruction}" in ReActAnalysis.ANSWER_PROMPT

    def test_first_round_prompt_format(self):
        prompt = ReActAnalysis.FIRST_ROUND_PROMPT.format(max_rounds=5)
        assert "5" in prompt

    def test_continue_prompt_format(self):
        prompt = ReActAnalysis.CONTINUE_PROMPT.format(remaining=2)
        assert "2" in prompt

    def test_answer_prompt_format(self):
        prompt = ReActAnalysis.ANSWER_PROMPT.format(instruction="tell me a joke")
        assert "tell me a joke" in prompt


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_default_answer_none(self):
        a = Analysis()
        assert a.answer is None

    def test_with_answer(self):
        a = Analysis(answer="42")
        assert a.answer == "42"

    def test_whitespace_only_becomes_none(self):
        a = Analysis(answer="   ")
        assert a.answer is None

    def test_empty_string_becomes_none(self):
        a = Analysis(answer="")
        assert a.answer is None

    def test_non_string_raises(self):
        with pytest.raises(Exception):
            Analysis(answer=12345)

    def test_answer_stripped(self):
        a = Analysis(answer="  hello  ")
        assert a.answer == "hello"


# ---------------------------------------------------------------------------
# _needs_extension
# ---------------------------------------------------------------------------


class TestNeedsExtension:
    def test_true_when_extension_needed_true(self):
        analysis = ReActAnalysis(analysis="...", extension_needed=True)
        assert _needs_extension(analysis) is True

    def test_false_when_extension_needed_false(self):
        analysis = ReActAnalysis(analysis="...", extension_needed=False)
        assert _needs_extension(analysis) is False

    def test_dict_with_extension_needed(self):
        d = {"extension_needed": True}
        assert _needs_extension(d) is True

    def test_dict_without_extension_needed(self):
        d = {"analysis": "done"}
        assert _needs_extension(d) is False

    def test_object_without_attribute(self):
        assert _needs_extension("plain string") is False

    def test_none_returns_false(self):
        assert _needs_extension(None) is False


# ---------------------------------------------------------------------------
# ReActParams
# ---------------------------------------------------------------------------


class TestReActParams:
    def setup_method(self):
        self.gen = GenerateParams(primary="test")
        self.op = Operable([Spec(str, name="answer")])

    def test_basic_construction(self):
        p = ReActParams(
            instruction="What is 2+2?",
            operable=self.op,
            generate_params=self.gen,
        )
        assert p.instruction == "What is 2+2?"
        assert p.max_rounds == 3
        assert p.invoke_actions is True

    def test_custom_max_rounds(self):
        p = ReActParams(
            instruction="x",
            operable=self.op,
            generate_params=self.gen,
            max_rounds=5,
        )
        assert p.max_rounds == 5

    def test_persist_default(self):
        p = ReActParams(
            instruction="x",
            operable=self.op,
            generate_params=self.gen,
        )
        assert p.persist is True

    def test_request_model_default_none(self):
        p = ReActParams(
            instruction="x",
            operable=self.op,
            generate_params=self.gen,
        )
        assert p.request_model is None

    def test_auto_fix_default(self):
        p = ReActParams(
            instruction="x",
            operable=self.op,
            generate_params=self.gen,
        )
        assert p.auto_fix is True

    def test_toolkits_default_none(self):
        p = ReActParams(
            instruction="x",
            operable=self.op,
            generate_params=self.gen,
        )
        assert p.toolkits is None
