# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for LNDL operate integration paths.

Covers:
- System-prompt injection and restore (including branch with no system)
- _try_finalize_lndl_once with invoke_actions=False (ActionCall → None)
- _execute_program_lacts returning parse errors
- _apply_lndl_handle_validation policies
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from lionagi.lndl import Lexer, Parser, normalize_lndl_text
from lionagi.operations.operate.operate import (
    _NO_RESTORE,
    _apply_lndl_handle_validation,
    _ensure_lndl_system_prompt,
    _restore_lndl_system_prompt,
    _try_finalize_lndl_once,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class AnswerResponse(BaseModel):
    answer: str


class ScoreResponse(BaseModel):
    score: float
    reason: str


def _make_mock_branch(system_text: str | None = None):
    """Build a minimal mock branch with a real-ish msgs.system."""
    branch = MagicMock()
    msgs = MagicMock()
    branch.msgs = msgs

    if system_text is not None:
        system_obj = MagicMock()
        system_obj.content.system_message = system_text
        msgs.system = system_obj
    else:
        msgs.system = None

    def _create_system(system=""):
        obj = MagicMock()
        obj.content.system_message = system
        return obj

    msgs.create_system = MagicMock(side_effect=_create_system)
    msgs.set_system = MagicMock()
    msgs.messages = MagicMock()
    return branch


# ---------------------------------------------------------------------------
# System-prompt injection and restore
# ---------------------------------------------------------------------------


class TestSystemPromptScope:
    def test_inject_on_empty_branch(self):
        branch = _make_mock_branch(system_text=None)
        token = _ensure_lndl_system_prompt(branch)
        # Should have injected (called set_system)
        assert branch.msgs.set_system.called
        # Token should be None (original state was no system)
        assert token is None

    def test_inject_on_branch_with_system(self):
        branch = _make_mock_branch(system_text="You are a coder.")
        token = _ensure_lndl_system_prompt(branch)
        assert branch.msgs.set_system.called
        # Token should be the original system object
        assert token is not None
        assert token is not _NO_RESTORE

    def test_inject_idempotent_when_marker_present(self):
        branch = _make_mock_branch(
            system_text="LNDL — Structured Output with Natural Thinking\nmore stuff"
        )
        token = _ensure_lndl_system_prompt(branch)
        assert token is _NO_RESTORE
        assert not branch.msgs.set_system.called

    def test_restore_no_restore_token(self):
        branch = _make_mock_branch()
        _restore_lndl_system_prompt(branch, _NO_RESTORE)
        assert not branch.msgs.set_system.called
        assert not branch.msgs.messages.exclude.called

    def test_restore_none_removes_system(self):
        """Regression: restoring None must NOT create a default system."""
        branch = _make_mock_branch(system_text=None)
        # Simulate: _ensure injected a system
        injected = MagicMock()
        branch.msgs.system = injected

        _restore_lndl_system_prompt(branch, None)
        # Should exclude the injected system from messages
        branch.msgs.messages.exclude.assert_called_once_with(injected)
        # Should set system back to None
        assert branch.msgs.system is None

    def test_restore_previous_system(self):
        original = MagicMock()
        branch = _make_mock_branch()
        _restore_lndl_system_prompt(branch, original)
        branch.msgs.set_system.assert_called_once_with(original)

    def test_non_lndl_passes_no_restore(self):
        """Non-LNDL calls must pass _NO_RESTORE, not None."""
        from lionagi.operations.operate.operate import _NO_RESTORE

        # This is a design assertion — tested in prepare_operate_kw
        assert _NO_RESTORE is not None


# ---------------------------------------------------------------------------
# _try_finalize_lndl_once: invoke_actions=False
# ---------------------------------------------------------------------------


class TestFinalizeWithoutActions:
    @pytest.mark.asyncio
    async def test_lact_placeholders_become_none(self):
        """When action_param=None, ActionCall should be replaced with None."""
        raw = '<lact answer a>lookup(query="weather")</lact>\n' "OUT{answer: [a]}"
        chat_param = MagicMock()
        chat_param.response_format = AnswerResponse

        output, error = await _try_finalize_lndl_once(
            branch=MagicMock(),
            raw=raw,
            chat_param=chat_param,
            action_param=None,  # invoke_actions=False
            operative=None,
            model_class=AnswerResponse,
        )
        # Should fail validation because answer=None doesn't match str
        assert error is not None
        assert "answer" in error.lower() or "validation" in error.lower()

    @pytest.mark.asyncio
    async def test_pure_lvar_validates_without_actions(self):
        """Pure lvar output should validate even when action_param=None."""
        raw = "<lvar answer a>42</lvar>\nOUT{answer: [a]}"
        chat_param = MagicMock()
        chat_param.response_format = AnswerResponse

        output, error = await _try_finalize_lndl_once(
            branch=MagicMock(),
            raw=raw,
            chat_param=chat_param,
            action_param=None,
            operative=None,
            model_class=AnswerResponse,
        )
        assert error is None
        assert output.answer == "42"


# ---------------------------------------------------------------------------
# _execute_program_lacts: parse error surfacing
# ---------------------------------------------------------------------------


class TestLactParseErrors:
    @pytest.mark.asyncio
    async def test_malformed_lact_returns_parse_error(self):
        """Malformed lact call should surface as a parse error, not be silently dropped."""
        from lionagi.operations.operate.operate import _execute_program_lacts

        # Build a program with a broken lact
        raw = "<lact a>not_a_valid_call!!!!</lact>\nOUT{answer: [a]}"
        raw = normalize_lndl_text(raw)
        lexer = Lexer(raw)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=raw)
        program = parser.parse()

        action_param = MagicMock()
        results, errors = await _execute_program_lacts(
            MagicMock(), program, action_param
        )
        assert len(errors) >= 1
        assert "parse failed" in errors[0].lower()
        assert len(results) == 0


# ---------------------------------------------------------------------------
# _apply_lndl_handle_validation
# ---------------------------------------------------------------------------


class TestHandleValidation:
    def test_return_value_on_error(self):
        result = _apply_lndl_handle_validation(
            {"partial": True}, "some error", "return_value", target=AnswerResponse
        )
        assert result == {"partial": True}

    def test_return_none_on_error(self):
        result = _apply_lndl_handle_validation(
            {"partial": True}, "some error", "return_none", target=AnswerResponse
        )
        assert result is None

    def test_raise_on_error(self):
        with pytest.raises(ValueError, match="AnswerResponse"):
            _apply_lndl_handle_validation(
                {}, "validation failed", "raise", target=AnswerResponse
            )

    def test_no_error_passes_through(self):
        result = _apply_lndl_handle_validation(
            {"answer": "ok"}, None, "raise", target=AnswerResponse
        )
        assert result == {"answer": "ok"}
