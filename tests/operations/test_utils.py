# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for beta/operations/utils.py — ReturnAs, handle_return."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lionagi._errors import ValidationError
from lionagi.operations.utils import ReturnAs, handle_return
from lionagi.ln.types._sentinel import Unset


def make_calling_mock(data=None, serialized=None):
    """Create a mock Calling with a Normalized response attached."""
    from lionagi.service.backend import Normalized

    normalized = Normalized(
        status="success",
        data=data or "llm_text",
        serialized=serialized or {"raw": "data"},
    )
    calling = MagicMock()
    calling.response = normalized
    calling.assert_is_normalized = MagicMock()
    return calling


# ---------------------------------------------------------------------------
# ReturnAs enum
# ---------------------------------------------------------------------------


class TestReturnAsEnum:
    def test_all_values_present(self):
        values = {m.value for m in ReturnAs}
        assert "text" in values
        assert "raw" in values
        assert "response" in values
        assert "message" in values
        assert "calling" in values
        assert "custom" in values


# ---------------------------------------------------------------------------
# handle_return
# ---------------------------------------------------------------------------


class TestHandleReturn:
    def test_calling_returns_calling_object(self):
        calling = make_calling_mock()
        result = handle_return(calling, ReturnAs.CALLING)
        assert result is calling

    def test_calling_does_not_call_assert_normalized(self):
        calling = make_calling_mock()
        handle_return(calling, ReturnAs.CALLING)
        calling.assert_is_normalized.assert_not_called()

    def test_text_returns_response_data(self):
        calling = make_calling_mock(data="the text")
        result = handle_return(calling, ReturnAs.TEXT)
        assert result == "the text"

    def test_raw_returns_serialized(self):
        calling = make_calling_mock(serialized={"raw": "dict"})
        result = handle_return(calling, ReturnAs.RAW)
        assert result == {"raw": "dict"}

    def test_response_returns_normalized(self):
        calling = make_calling_mock()
        result = handle_return(calling, ReturnAs.RESPONSE)
        assert hasattr(result, "data")

    def test_message_calls_parse_to_assistant_message(self):
        from lionagi.service.backend import Normalized

        calling = make_calling_mock()

        with pytest.MonkeyPatch().context() as m:
            mock_parse = MagicMock(return_value="assistant_msg")
            m.setattr(
                "lionagi.operations.utils.parse_to_assistant_message",
                mock_parse,
                raising=False,
            )
            # patch within the module's import
            import lionagi.operations.utils as utils_mod

            original = None
            try:
                import lionagi.protocols.messages.assistant_response as ar_mod

                original = ar_mod.parse_to_assistant_message
                ar_mod.parse_to_assistant_message = mock_parse
                result = handle_return(calling, ReturnAs.MESSAGE)
                # Result is whatever parse_to_assistant_message returned
            finally:
                if original:
                    ar_mod.parse_to_assistant_message = original

    def test_custom_with_parser(self):
        calling = make_calling_mock()
        parser = lambda c: "custom_result"
        result = handle_return(calling, ReturnAs.CUSTOM, return_parser=parser)
        assert result == "custom_result"

    def test_custom_without_parser_raises(self):
        calling = make_calling_mock()
        with pytest.raises(ValidationError):
            handle_return(calling, ReturnAs.CUSTOM)

    def test_custom_non_callable_parser_raises(self):
        calling = make_calling_mock()
        with pytest.raises(ValidationError):
            handle_return(calling, ReturnAs.CUSTOM, return_parser="not_callable")

    def test_text_calls_assert_normalized(self):
        calling = make_calling_mock()
        handle_return(calling, ReturnAs.TEXT)
        calling.assert_is_normalized.assert_called_once()

    def test_raw_calls_assert_normalized(self):
        calling = make_calling_mock()
        handle_return(calling, ReturnAs.RAW)
        calling.assert_is_normalized.assert_called_once()
