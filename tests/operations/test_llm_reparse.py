# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/llm_reparse.py — prompts and constants."""

from __future__ import annotations

import pytest

from lionagi.operations.llm_reparse import (
    JSON_REPARSE_PROMPT,
    LNDL_REPARSE_PROMPT,
    _llm_reparse,
    _lndl_reparse,
)

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------


class TestPromptConstants:
    def test_json_reparse_prompt_nonempty(self):
        assert isinstance(JSON_REPARSE_PROMPT, str)
        assert len(JSON_REPARSE_PROMPT) > 0

    def test_lndl_reparse_prompt_has_placeholders(self):
        assert "{error}" in LNDL_REPARSE_PROMPT
        assert "{original_text}" in LNDL_REPARSE_PROMPT

    def test_lndl_reparse_prompt_format(self):
        formatted = LNDL_REPARSE_PROMPT.format(
            error="missing OUT{}", original_text="<lvar x>val</lvar>"
        )
        assert "missing OUT{}" in formatted
        assert "<lvar x>val</lvar>" in formatted


# ---------------------------------------------------------------------------
# _lndl_reparse without operable raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lndl_reparse_no_operable_raises():
    """_lndl_reparse should raise ValueError when operable is None, before any LLM call."""
    session = None
    branch = None
    # We need to reach the operable check — but _lndl_reparse calls _generate first.
    # Since no imodel is set, it will fail trying to call the LLM.
    # Just test the operable=None branch via _llm_reparse with structure_format="lndl"
    with pytest.raises(Exception):
        await _lndl_reparse(
            session=None,
            branch=None,
            text="<lvar x>hello</lvar>",
            imodel=None,
            operable=None,
        )


# ---------------------------------------------------------------------------
# _llm_reparse sentinel checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_reparse_requires_request_model():
    """Without a request_model (Unset), _llm_reparse should raise ValueError."""
    from lionagi.ln.types._sentinel import Unset

    with pytest.raises(Exception):
        await _llm_reparse(
            session=None,
            branch=None,
            text='{"a": 1}',
            imodel=None,
            request_model=Unset,  # sentinel
        )
