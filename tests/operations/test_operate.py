# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/operate.py — JSON action helpers and OperateParams.

LNDL orchestration helpers live in lionagi.lndl.orchestrator and are
covered by tests/beta/lndl/test_orchestrator.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from lionagi.operations.generate import GenerateParams
from lionagi.operations.operate import (
    OperateParams,
    _to_action_messages,
    _to_action_results,
)
from lionagi.operations.specs import Action, ActionResult
from lionagi.protocols.messages import Message


class TestToActionMessages:
    def test_action_objects_converted(self):
        action = Action(function="do_thing", arguments={"x": 1})
        messages = _to_action_messages([action])
        assert len(messages) == 1
        assert isinstance(messages[0], Message)

    def test_dict_actions_converted(self):
        action = {"function": "do_thing", "arguments": {"x": 1}}
        messages = _to_action_messages([action])
        assert len(messages) == 1

    def test_mixed_list(self):
        actions = [
            Action(function="a"),
            {"function": "b", "arguments": {}},
        ]
        messages = _to_action_messages(actions)
        assert len(messages) == 2

    def test_empty_list_returns_empty(self):
        assert _to_action_messages([]) == []

    def test_unknown_type_skipped(self):
        messages = _to_action_messages(["not_an_action"])
        assert messages == []


class TestToActionResults:
    def test_empty_returns_empty(self):
        assert _to_action_results([], []) == []

    def test_dict_response_validated(self):
        responses = [{"function": "my_fn", "result": "ok", "error": None}]
        results = _to_action_results(responses, [])
        assert len(results) == 1
        assert isinstance(results[0], ActionResult)

    def test_action_response_object(self):
        from lionagi.protocols.messages.action_response import (
            ActionResponseContent as ActionResponse,
        )

        resp = ActionResponse.create(request_id="id123", result="done")
        msg = MagicMock()
        msg.id = "id123"
        content = MagicMock()
        content.function = "fn_name"
        msg.content = content

        results = _to_action_results([resp], [msg])
        assert len(results) == 1
        assert isinstance(results[0], ActionResult)


class TestOperateParams:
    def test_construction(self):
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
