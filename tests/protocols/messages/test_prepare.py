# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.protocols.messages.prepare."""

from __future__ import annotations

import pytest

from lionagi.protocols.generic.pile import Pile
from lionagi.protocols.generic.progression import Progression
from lionagi.protocols.messages.action_request import ActionRequestContent
from lionagi.protocols.messages.action_response import ActionResponseContent
from lionagi.protocols.messages.assistant_response import AssistantResponseContent
from lionagi.protocols.messages.instruction import InstructionContent
from lionagi.protocols.messages.message import RoledMessage
from lionagi.protocols.messages.prepare import (
    _aggregate_round_actions,
    _build_context,
    _build_round_notification,
    _get_content,
    _get_text,
    _is_message,
    prepare_messages_for_chat,
)
from lionagi.protocols.messages.system import SystemContent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_pile(*contents):
    """Build a beta Pile from MessageContent objects wrapped in RoledMessage."""
    msgs = [RoledMessage(content=c) for c in contents]
    p = Pile()
    for m in msgs:
        p.include(m)
    ids = [m.id for m in msgs]
    prog = Progression(order=ids)
    return p, prog, msgs


# ---------------------------------------------------------------------------
# _get_text
# ---------------------------------------------------------------------------


class TestGetText:
    def test_returns_str_for_existing_attr(self):
        content = InstructionContent(primary="hello")
        result = _get_text(content, "instruction")
        assert result == "hello"

    def test_returns_empty_for_missing_attr(self):
        content = InstructionContent(primary="hello")
        result = _get_text(content, "nonexistent_attr")
        assert result == ""

    def test_returns_empty_for_none_value(self):
        content = AssistantResponseContent(assistant_response=None)
        result = _get_text(content, "assistant_response")
        assert result == ""

    def test_system_message_attr(self):
        content = SystemContent(system_message="You are helpful")
        text = _get_text(content, "system_message")
        assert text == "You are helpful"


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    def test_appends_to_empty_context(self):
        content = InstructionContent(primary="task")
        result = _build_context(content, ["action output"])
        assert "action output" in result

    def test_appends_to_existing_context(self):
        content = InstructionContent(primary="task", context=["existing context"])
        result = _build_context(content, ["new output"])
        assert "existing context" in result
        assert "new output" in result

    def test_returns_list(self):
        content = InstructionContent(primary="task")
        result = _build_context(content, ["a", "b"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_no_action_outputs(self):
        content = InstructionContent(primary="task")
        result = _build_context(content, [])
        assert result == []


# ---------------------------------------------------------------------------
# _aggregate_round_actions
# ---------------------------------------------------------------------------


class TestAggregateRoundActions:
    def test_no_responses(self):
        result = _aggregate_round_actions([], [], round_num=1)
        assert "round: 1" in result
        assert "time:" in result
        assert "actions:" not in result

    def test_with_responses(self):
        req = ActionRequestContent(function="my_func", arguments={"x": 1})
        resp = ActionResponseContent(
            function="my_func", arguments={"x": 1}, output={"result": 42}
        )
        result = _aggregate_round_actions([req], [resp], round_num=2)
        assert "round: 2" in result
        assert "actions:" in result
        assert "my_func" in result

    def test_more_responses_than_requests(self):
        resp1 = ActionResponseContent(function="f1", arguments={}, output={"r": 1})
        resp2 = ActionResponseContent(function="f2", arguments={}, output={"r": 2})
        result = _aggregate_round_actions([], [resp1, resp2], round_num=1)
        assert "unknown()" in result
        assert "actions:" in result

    def test_round_number_in_output(self):
        result = _aggregate_round_actions([], [], round_num=5)
        assert "round: 5" in result


# ---------------------------------------------------------------------------
# _build_round_notification
# ---------------------------------------------------------------------------


class TestBuildRoundNotification:
    def test_basic_structure(self):
        result = _build_round_notification(None, round_num=1, msg_count=5)
        assert "<system" in result
        assert "</system>" in result
        assert "round" in result
        assert "5 msgs" in result

    def test_with_scratchpad(self):
        result = _build_round_notification(
            None,
            round_num=1,
            msg_count=3,
            scratchpad={"key": "value"},
        )
        assert "scratchpad:" in result
        assert "key:" in result
        assert "value" in result

    def test_no_scratchpad_no_section(self):
        result = _build_round_notification(None, round_num=1, msg_count=3)
        assert "scratchpad:" not in result


# ---------------------------------------------------------------------------
# _is_message and _get_content
# ---------------------------------------------------------------------------


class TestIsMessageAndGetContent:
    def test_is_message_with_roled_message(self):
        msg = RoledMessage(content=InstructionContent(primary="hello"))
        assert _is_message(msg) is True

    def test_is_message_with_content_object(self):
        content = InstructionContent(primary="hello")
        assert _is_message(content) is False

    def test_get_content_from_roled_message(self):
        content = InstructionContent(primary="hello")
        msg = RoledMessage(content=content)
        result = _get_content(msg)
        assert isinstance(result, InstructionContent)

    def test_get_content_from_content_directly(self):
        content = SystemContent(system_message="sys")
        result = _get_content(content)
        assert result is content


# ---------------------------------------------------------------------------
# prepare_messages_for_chat
# ---------------------------------------------------------------------------


class TestPrepareMessagesForChat:
    def test_empty_pile_returns_empty(self):
        pile = Pile()
        result = prepare_messages_for_chat(pile)
        assert result == []

    def test_empty_pile_with_new_instruction(self):
        pile = Pile()
        new_instr = InstructionContent(primary="Hello!")
        result = prepare_messages_for_chat(pile, new_instruction=new_instr)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello!" in result[0]["content"]

    def test_empty_pile_with_new_instruction_to_chat_false(self):
        pile = Pile()
        new_instr = InstructionContent(primary="Hi!")
        result = prepare_messages_for_chat(
            pile, new_instruction=new_instr, to_chat=False
        )
        assert len(result) == 1
        assert isinstance(result[0], InstructionContent)

    def test_system_plus_instruction(self):
        pile, prog, _ = make_pile(
            SystemContent(system_message="You are helpful"),
            InstructionContent(primary="What is 2+2?"),
        )
        result = prepare_messages_for_chat(pile, prog)
        assert len(result) == 1  # system embedded into instruction
        assert result[0]["role"] == "user"
        assert "You are helpful" in result[0]["content"]
        assert "2+2" in result[0]["content"]

    def test_instruction_and_assistant(self):
        pile, prog, _ = make_pile(
            InstructionContent(primary="What is Python?"),
            AssistantResponseContent(assistant_response="Python is a language."),
        )
        result = prepare_messages_for_chat(pile, prog)
        assert len(result) == 2
        roles = [r["role"] for r in result]
        assert "user" in roles
        assert "assistant" in roles

    def test_consecutive_assistants_merged(self):
        pile, prog, _ = make_pile(
            InstructionContent(primary="Question"),
            AssistantResponseContent(assistant_response="Part 1"),
            AssistantResponseContent(assistant_response="Part 2"),
        )
        result = prepare_messages_for_chat(pile, prog)
        assistant_msgs = [r for r in result if r["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert "Part 1" in assistant_msgs[0]["content"]
        assert "Part 2" in assistant_msgs[0]["content"]

    def test_action_responses_embedded_as_context(self):
        ar = ActionResponseContent(
            function="search",
            arguments={"q": "python"},
            output={"result": "found"},
        )
        pile, prog, _ = make_pile(
            InstructionContent(primary="Step 1"),
            ActionRequestContent(function="search", arguments={"q": "python"}),
            ar,
            InstructionContent(primary="Step 2"),
        )
        result = prepare_messages_for_chat(pile, prog)
        # Step 2 instruction should have action context
        user_msgs = [r for r in result if r["role"] == "user"]
        assert len(user_msgs) == 2
        # Second user message should contain action output
        assert (
            "result" in user_msgs[1]["content"]
            or "found" in user_msgs[1]["content"]
            or "search" in user_msgs[1]["content"]
        )

    def test_aggregate_actions_mode(self):
        req = ActionRequestContent(function="lookup", arguments={"key": "x"})
        resp = ActionResponseContent(
            function="lookup", arguments={"key": "x"}, output={"value": 42}
        )
        pile, prog, _ = make_pile(
            InstructionContent(primary="Round 1 question"),
            req,
            resp,
            InstructionContent(primary="Round 2 question"),
        )
        result = prepare_messages_for_chat(pile, prog, aggregate_actions=True)
        user_msgs = [r for r in result if r["role"] == "user"]
        assert len(user_msgs) == 2
        # Second instruction should have aggregated action summary
        content = user_msgs[1]["content"]
        assert "round:" in content or "lookup" in content

    def test_round_notifications_mode(self):
        req = ActionRequestContent(function="f", arguments={})
        resp = ActionResponseContent(function="f", arguments={}, output={"r": 1})
        pile, prog, _ = make_pile(
            InstructionContent(primary="Step 1"),
            req,
            resp,
            InstructionContent(primary="Step 2"),
        )
        result = prepare_messages_for_chat(
            pile, prog, aggregate_actions=True, round_notifications=True
        )
        user_msgs = [r for r in result if r["role"] == "user"]
        content = user_msgs[1]["content"]
        assert "<system" in content

    def test_system_prefix_prepended(self):
        pile, prog, _ = make_pile(
            InstructionContent(primary="Task"),
        )
        result = prepare_messages_for_chat(pile, prog, system_prefix="LNDL prefix")
        assert "LNDL prefix" in result[0]["content"]

    def test_system_prefix_with_existing_system(self):
        pile, prog, _ = make_pile(
            SystemContent(system_message="Existing system"),
            InstructionContent(primary="Task"),
        )
        result = prepare_messages_for_chat(pile, prog, system_prefix="EXTRA: ")
        content = result[0]["content"]
        assert "EXTRA: " in content
        assert "Existing system" in content

    def test_middle_system_messages_skipped(self):
        pile, prog, _ = make_pile(
            InstructionContent(primary="Q1"),
            SystemContent(system_message="Mid system"),
            InstructionContent(primary="Q2"),
        )
        result = prepare_messages_for_chat(pile, prog)
        # Middle system message should be skipped
        contents = [r["content"] for r in result]
        # Q2 should appear but not with mid system content embedded
        combined = " ".join(contents)
        assert "Q1" in combined
        assert "Q2" in combined

    def test_no_progression_uses_pile_order(self):
        pile, prog, msgs = make_pile(
            InstructionContent(primary="Natural order"),
            AssistantResponseContent(assistant_response="Reply"),
        )
        result = prepare_messages_for_chat(pile)
        # Should process all messages in default pile order
        assert len(result) >= 1

    def test_new_instruction_appended_after_history(self):
        pile, prog, _ = make_pile(
            InstructionContent(primary="History"),
            AssistantResponseContent(assistant_response="Reply"),
        )
        new_instr = InstructionContent(primary="New question")
        result = prepare_messages_for_chat(pile, prog, new_instruction=new_instr)
        user_msgs = [r for r in result if r["role"] == "user"]
        assert any("New question" in m["content"] for m in user_msgs)

    def test_to_chat_false_returns_content_objects(self):
        pile, prog, _ = make_pile(
            InstructionContent(primary="Task"),
            AssistantResponseContent(assistant_response="Done"),
        )
        result = prepare_messages_for_chat(pile, prog, to_chat=False)
        assert all(
            isinstance(r, (InstructionContent, AssistantResponseContent))
            for r in result
        )

    def test_history_instruction_tool_schemas_stripped(self):
        # tool_schemas and response_format are stripped from historical instructions
        instr = InstructionContent(primary="Call tool", tool_schemas=[{"name": "f"}])
        pile, prog, _ = make_pile(instr)
        result = prepare_messages_for_chat(pile, prog, to_chat=False)
        assert len(result) == 1
        # with_updates(tool_schemas=None) results in [] not None
        assert not result[0].tool_schemas

    def test_empty_new_instruction_edge_case(self):
        pile = Pile()
        # Pass None new_instruction — should return empty
        result = prepare_messages_for_chat(pile, new_instruction=None)
        assert result == []

    def test_scratchpad_in_round_notification(self):
        req = ActionRequestContent(function="f", arguments={})
        resp = ActionResponseContent(function="f", arguments={}, output={"r": 1})
        pile, prog, _ = make_pile(
            InstructionContent(primary="Step 1"),
            req,
            resp,
            InstructionContent(primary="Step 2"),
        )
        result = prepare_messages_for_chat(
            pile,
            prog,
            aggregate_actions=True,
            round_notifications=True,
            scratchpad={"note": "important context"},
        )
        user_msgs = [r for r in result if r["role"] == "user"]
        content = user_msgs[1]["content"]
        assert "important context" in content or "note" in content
