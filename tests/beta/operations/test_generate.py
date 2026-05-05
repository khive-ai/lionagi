# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/generate.py — GenerateParams and instruction_message composition."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from lionagi.beta.operations.generate import GenerateParams, handle_return
from lionagi.beta.operations.utils import ReturnAs
from lionagi.protocols.messages import Message
from lionagi.protocols.messages.instruction import InstructionContent as Instruction
from lionagi.protocols.messages.rendering import StructureFormat

# ---------------------------------------------------------------------------
# GenerateParams construction
# ---------------------------------------------------------------------------


class TestGenerateParamsConstruction:
    def test_defaults(self):
        p = GenerateParams()
        assert p.structure_format == "json"
        assert p.return_as == ReturnAs.CALLING

    def test_with_primary(self):
        p = GenerateParams(primary="Tell me a joke")
        assert p.primary == "Tell me a joke"

    def test_with_structure_format_lndl(self):
        p = GenerateParams(structure_format="lndl")
        assert p.structure_format == "lndl"

    def test_imodel_kwargs_default_empty(self):
        p = GenerateParams()
        assert p.imodel_kwargs == {}

    def test_with_request_model(self):
        class M(BaseModel):
            x: int

        p = GenerateParams(request_model=M)
        assert p.request_model is M


# ---------------------------------------------------------------------------
# GenerateParams.instruction_message
# ---------------------------------------------------------------------------


class TestInstructionMessage:
    def test_primary_creates_message(self):
        p = GenerateParams(primary="hello world")
        msg = p.instruction_message
        assert isinstance(msg, Message)
        assert isinstance(msg.content, Instruction)

    def test_instruction_as_message_returns_it(self):
        original_msg = Message(content=Instruction.create(primary="test"))
        p = GenerateParams(instruction=original_msg)
        result = p.instruction_message
        assert result is original_msg

    def test_instruction_as_instruction_content_wraps(self):
        instr = Instruction.create(primary="wrapped")
        p = GenerateParams(instruction=instr)
        msg = p.instruction_message
        assert isinstance(msg, Message)

    def test_no_primary_no_instruction_still_creates_message(self):
        p = GenerateParams()
        msg = p.instruction_message
        assert isinstance(msg, Message)

    def test_context_embedded_in_message(self):
        p = GenerateParams(primary="do this", context={"env": "prod"})
        msg = p.instruction_message
        assert isinstance(msg, Message)
        # Context passed through
        content = msg.content
        assert content is not None

    def test_with_tool_schemas(self):
        p = GenerateParams(primary="use tools", tool_schemas=["schema1", "schema2"])
        msg = p.instruction_message
        assert isinstance(msg, Message)

    def test_return_as_text(self):
        p = GenerateParams(return_as=ReturnAs.TEXT)
        assert p.return_as == ReturnAs.TEXT

    def test_sentinel_instruction_falls_back_to_primary(self):
        from lionagi.ln.types._sentinel import Unset, is_sentinel

        p = GenerateParams(primary="fallback")
        assert is_sentinel(p.instruction)
        msg = p.instruction_message
        assert isinstance(msg, Message)

    def test_with_updates_copies_params(self):
        p = GenerateParams(primary="original")
        p2 = p.with_updates(copy_containers="deep", primary="updated")
        assert p2.primary == "updated"
        assert p.primary == "original"

    def test_with_updates_return_as(self):
        p = GenerateParams()
        p2 = p.with_updates(copy_containers="deep", return_as=ReturnAs.MESSAGE)
        assert p2.return_as == ReturnAs.MESSAGE


# ---------------------------------------------------------------------------
# is_sentinel_field checks
# ---------------------------------------------------------------------------


class TestSentinelFields:
    def test_imodel_is_sentinel_by_default(self):
        p = GenerateParams()
        assert p.is_sentinel_field("imodel")

    def test_tool_schemas_is_sentinel_by_default(self):
        p = GenerateParams()
        assert p.is_sentinel_field("tool_schemas")

    def test_request_model_is_sentinel_by_default(self):
        p = GenerateParams()
        assert p.is_sentinel_field("request_model")

    def test_instruction_is_sentinel_by_default(self):
        p = GenerateParams()
        assert p.is_sentinel_field("instruction")

    def test_primary_not_sentinel_when_set(self):
        p = GenerateParams(primary="hello")
        assert not p.is_sentinel_field("primary")
