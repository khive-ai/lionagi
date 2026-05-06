# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.lndl.orchestrator helper functions (no LLM calls)."""

from __future__ import annotations

from unittest.mock import MagicMock

from lionagi.lndl.orchestrator import (
    _last_assistant_text,
    build_continuation_prompt,
    extract_lvars,
)


class TestBuildContinuationPrompt:
    def test_first_round_message(self):
        result = build_continuation_prompt(0, max_rounds=3)
        assert "Round 1" in result
        assert "OUT{}" in result

    def test_with_last_error(self):
        result = build_continuation_prompt(1, max_rounds=3, last_error="parse failed")
        assert "parse failed" in result

    def test_final_round_message(self):
        result = build_continuation_prompt(2, max_rounds=3)
        assert "FINAL" in result

    def test_low_rounds_message(self):
        result = build_continuation_prompt(1, max_rounds=3)
        assert "Running low" in result or "FINAL" in result or "Continue" in result

    def test_no_error_continue(self):
        result = build_continuation_prompt(0, max_rounds=5)
        assert "Continue" in result or "Round" in result


class TestExtractLvars:
    def test_extracts_lvar(self):
        text = "<lvar greeting>Hello world</lvar>"
        lvars, lacts = extract_lvars(text)
        assert "greeting" in lvars

    def test_empty_on_invalid_text(self):
        lvars, lacts = extract_lvars("no lvar here")
        assert lvars == {}
        assert lacts == {}

    def test_extracts_multiple_lvars(self):
        text = "<lvar a>alpha</lvar>\n<lvar b>beta</lvar>"
        lvars, lacts = extract_lvars(text)
        assert "a" in lvars
        assert "b" in lvars

    def test_returns_tuple_of_dicts(self):
        lvars, lacts = extract_lvars("")
        assert isinstance(lvars, dict)
        assert isinstance(lacts, dict)


class TestLastAssistantText:
    def test_returns_none_if_branch_none(self):
        session = MagicMock()
        assert _last_assistant_text(session, None) is None

    def test_returns_none_if_empty_order(self):
        branch = MagicMock()
        branch.order = []
        session = MagicMock()
        assert _last_assistant_text(session, branch) is None

    def test_finds_response_text(self):
        branch = MagicMock()
        msg_id = "abc"
        branch.order = [msg_id]

        content = MagicMock()
        content.response = "assistant said this"
        msg = MagicMock()
        msg.content = content

        session = MagicMock()
        session.messages = {msg_id: msg}

        assert _last_assistant_text(session, branch) == "assistant said this"

    def test_skips_non_assistant_messages(self):
        branch = MagicMock()
        branch.order = ["id1", "id2"]

        no_response = MagicMock(spec=[])

        with_response = MagicMock()
        with_response.response = "found it"

        m1 = MagicMock()
        m1.content = no_response

        m2 = MagicMock()
        m2.content = with_response

        session = MagicMock()
        session.messages = {"id1": m1, "id2": m2}

        # Iteration is reversed — most recent first; id2 has the response.
        assert _last_assistant_text(session, branch) == "found it"

    def test_returns_none_if_no_response_found(self):
        branch = MagicMock()
        branch.order = ["id1"]

        content = MagicMock(spec=[])
        msg = MagicMock()
        msg.content = content

        session = MagicMock()
        session.messages = {"id1": msg}

        assert _last_assistant_text(session, branch) is None
