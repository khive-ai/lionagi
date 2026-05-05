# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/operate.py — helper functions (no LLM calls)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lionagi.beta.operations.generate import GenerateParams
from lionagi.beta.operations.operate import (
    OperateParams,
    _actions_to_messages,
    _build_lndl_continuation,
    _extract_lvars_from_text,
    _get_last_assistant_text,
    _responses_to_results,
)
from lionagi.beta.operations.specs import Action, ActionResult
from lionagi.protocols.messages import Message

# ---------------------------------------------------------------------------
# _build_lndl_continuation
# ---------------------------------------------------------------------------


class TestBuildLndlContinuation:
    def test_first_round_message(self):
        branch = MagicMock()
        result = _build_lndl_continuation(branch, 0, max_rounds=3)
        assert "Round 1" in result
        assert "OUT{}" in result

    def test_with_last_error(self):
        branch = MagicMock()
        result = _build_lndl_continuation(
            branch, 1, last_error="parse failed", max_rounds=3
        )
        assert "parse failed" in result

    def test_final_round_message(self):
        branch = MagicMock()
        result = _build_lndl_continuation(branch, 2, max_rounds=3)
        assert "FINAL" in result

    def test_low_rounds_message(self):
        branch = MagicMock()
        result = _build_lndl_continuation(branch, 1, max_rounds=3)
        assert "Running low" in result or "FINAL" in result or "Continue" in result

    def test_no_error_continue(self):
        branch = MagicMock()
        result = _build_lndl_continuation(branch, 0, max_rounds=5)
        assert "Continue" in result or "Round" in result


# ---------------------------------------------------------------------------
# _extract_lvars_from_text
# ---------------------------------------------------------------------------


class TestExtractLvarsFromText:
    def test_extracts_lvar(self):
        text = "<lvar greeting>Hello world</lvar>"
        lvars, lacts = _extract_lvars_from_text(text)
        assert "greeting" in lvars

    def test_empty_on_invalid_text(self):
        lvars, lacts = _extract_lvars_from_text("no lvar here")
        assert lvars == {}
        assert lacts == {}

    def test_extracts_multiple_lvars(self):
        text = "<lvar a>alpha</lvar>\n<lvar b>beta</lvar>"
        lvars, lacts = _extract_lvars_from_text(text)
        assert "a" in lvars
        assert "b" in lvars

    def test_returns_tuple_of_dicts(self):
        lvars, lacts = _extract_lvars_from_text("")
        assert isinstance(lvars, dict)
        assert isinstance(lacts, dict)


# ---------------------------------------------------------------------------
# _get_last_assistant_text
# ---------------------------------------------------------------------------


class TestGetLastAssistantText:
    def test_returns_none_if_branch_none(self):
        session = MagicMock()
        result = _get_last_assistant_text(session, None)
        assert result is None

    def test_returns_none_if_empty_order(self):
        branch = MagicMock()
        branch.order = []
        session = MagicMock()
        result = _get_last_assistant_text(session, branch)
        assert result is None

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

        result = _get_last_assistant_text(session, branch)
        assert result == "assistant said this"

    def test_skips_non_assistant_messages(self):
        branch = MagicMock()
        branch.order = ["id1", "id2"]

        content_no_response = MagicMock(spec=[])  # no .response attribute

        content_with_response = MagicMock()
        content_with_response.response = "found it"

        msg1 = MagicMock()
        msg1.content = content_no_response

        msg2 = MagicMock()
        msg2.content = content_with_response

        session = MagicMock()
        session.messages = {"id1": msg1, "id2": msg2}

        result = _get_last_assistant_text(session, branch)
        assert result == "found it"

    def test_returns_none_if_no_response_found(self):
        branch = MagicMock()
        branch.order = ["id1"]

        content = MagicMock(spec=[])  # no .response
        msg = MagicMock()
        msg.content = content

        session = MagicMock()
        session.messages = {"id1": msg}

        result = _get_last_assistant_text(session, branch)
        assert result is None


# ---------------------------------------------------------------------------
# _actions_to_messages
# ---------------------------------------------------------------------------


class TestActionsToMessages:
    def test_action_objects_converted(self):
        action = Action(function="do_thing", arguments={"x": 1})
        messages = _actions_to_messages([action])
        assert len(messages) == 1
        assert isinstance(messages[0], Message)

    def test_dict_actions_converted(self):
        action = {"function": "do_thing", "arguments": {"x": 1}}
        messages = _actions_to_messages([action])
        assert len(messages) == 1

    def test_mixed_list(self):
        actions = [
            Action(function="a"),
            {"function": "b", "arguments": {}},
        ]
        messages = _actions_to_messages(actions)
        assert len(messages) == 2

    def test_empty_list_returns_empty(self):
        assert _actions_to_messages([]) == []

    def test_unknown_type_skipped(self):
        # Non-Action, non-dict items should be skipped
        messages = _actions_to_messages(["not_an_action"])
        assert messages == []


# ---------------------------------------------------------------------------
# _responses_to_results
# ---------------------------------------------------------------------------


class TestResponsesToResults:
    def test_empty_returns_empty(self):
        results = _responses_to_results([], [])
        assert results == []

    def test_dict_response_validated(self):
        responses = [{"function": "my_fn", "result": "ok", "error": None}]
        results = _responses_to_results(responses, [])
        assert len(results) == 1
        assert isinstance(results[0], ActionResult)

    def test_action_response_object(self):
        from lionagi.protocols.messages.action_response import (
            ActionResponseContent as ActionResponse,
        )

        resp = ActionResponse.create(request_id="id123", result="done")
        # Create a message to map the request_id
        msg = MagicMock()
        msg.id = "id123"
        content = MagicMock()
        content.function = "fn_name"
        msg.content = content

        results = _responses_to_results([resp], [msg])
        assert len(results) == 1
        assert isinstance(results[0], ActionResult)


# ---------------------------------------------------------------------------
# OperateParams construction
# ---------------------------------------------------------------------------


class TestOperateParams:
    def test_construction(self):
        from lionagi.beta.rules import Validator
        from lionagi.ln.types import Operable, Spec

        gen = GenerateParams(primary="test")
        op = Operable([Spec(str, name="answer")])
        params = OperateParams(generate_params=gen, operable=op)
        assert params.invoke_actions is False
        assert params.max_lndl_rounds == 3

    def test_invoke_actions_flag(self):
        from lionagi.ln.types import Operable, Spec

        gen = GenerateParams(primary="test")
        op = Operable([Spec(str, name="answer")])
        params = OperateParams(generate_params=gen, operable=op, invoke_actions=True)
        assert params.invoke_actions is True

    def test_action_strategy_default(self):
        from lionagi.ln.types import Operable, Spec

        gen = GenerateParams(primary="test")
        op = Operable([Spec(str, name="answer")])
        params = OperateParams(generate_params=gen, operable=op)
        assert params.action_strategy == "concurrent"
