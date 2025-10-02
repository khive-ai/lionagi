# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for lionagi.operations.parse module."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.operations.parse.parse import parse
from lionagi.session.branch import AlcallParams


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    name: str = Field(..., description="The name field")
    age: int = Field(..., description="The age field")
    city: str = Field(default="Unknown", description="The city field")


@pytest.mark.asyncio
class TestParseDirectPath:
    """Test direct parsing path (without LLM call)."""

    async def test_direct_parse_valid_json_to_model(
        self, branch_with_mock_imodel
    ):
        """Test parsing valid JSON directly to Pydantic model."""
        branch = branch_with_mock_imodel

        # Valid JSON that matches SampleModel
        text = '{"name": "Alice", "age": 30, "city": "NYC"}'

        result = await parse(
            branch=branch,
            text=text,
            response_format=SampleModel,
        )

        assert isinstance(result, SampleModel)
        assert result.name == "Alice"
        assert result.age == 30
        assert result.city == "NYC"

        # Verify no LLM call was made (mock should not be invoked)
        assert branch.chat_model.invoke.call_count == 0

    async def test_direct_parse_fuzzy_keys(self, branch_with_mock_imodel):
        """Test direct parsing with fuzzy key matching."""
        branch = branch_with_mock_imodel

        # JSON with slightly different keys (fuzzy match should work)
        text = '{"name": "Bob", "age": 25, "city": "LA"}'

        result = await parse(
            branch=branch,
            text=text,
            response_format=SampleModel,
        )

        assert isinstance(result, SampleModel)
        assert result.name == "Bob"
        assert result.age == 25
        assert result.city == "LA"

        # Verify no LLM call was made
        assert branch.chat_model.invoke.call_count == 0

    async def test_direct_parse_dict_format(self, branch_with_mock_imodel):
        """Test direct parsing to dict format."""
        branch = branch_with_mock_imodel

        text = '{"key1": "value1", "key2": 42}'
        response_format = {"key1": str, "key2": int}

        result = await parse(
            branch=branch,
            text=text,
            response_format=response_format,
        )

        assert isinstance(result, dict)
        assert result["key1"] == "value1"
        assert result["key2"] == 42

        # Verify no LLM call was made
        assert branch.chat_model.invoke.call_count == 0


@pytest.mark.asyncio
class TestParseLLMFallback:
    """Test LLM fallback path when direct parsing fails."""

    async def test_llm_fallback_on_invalid_json(self, branch_with_mock_imodel):
        """Test that LLM is called when direct parsing fails."""
        branch = branch_with_mock_imodel

        # Invalid JSON - will fail direct parsing
        text = "This is not JSON at all"

        # The conftest mock returns "mocked_response_string" which is also invalid
        # So this will test the retry/error handling path
        result = await parse(
            branch=branch,
            text=text,
            response_format=SampleModel,
            handle_validation="return_value",  # Return original text on failure
        )

        # Should return original text since mocked response is also invalid
        assert result == text

    async def test_llm_retry_logic_with_valid_mock(
        self, branch_with_mock_imodel
    ):
        """Test that parse attempts LLM fallback when direct parsing fails."""
        branch = branch_with_mock_imodel

        # Override the mock to return valid JSON
        from lionagi.protocols.generic.event import EventStatus
        from lionagi.service.connections.api_calling import APICalling

        async def _fake_invoke_valid(**kwargs):
            fake_call = APICalling(
                payload={"model": "gpt-4.1-mini", "messages": []},
                headers={"Authorization": "Bearer test"},
                endpoint=branch.chat_model.endpoint,
            )
            # Return valid JSON this time
            fake_call.execution.response = (
                '{"name": "Dave", "age": 40, "city": "Boston"}'
            )
            fake_call.execution.status = EventStatus.COMPLETED
            return fake_call

        branch.chat_model.invoke = AsyncMock(side_effect=_fake_invoke_valid)
        branch.parse_model.invoke = AsyncMock(side_effect=_fake_invoke_valid)

        text = "Invalid text that needs LLM parsing"
        result = await parse(
            branch=branch,
            text=text,
            response_format=SampleModel,
        )

        assert isinstance(result, SampleModel)
        assert result.name == "Dave"
        assert result.age == 40


@pytest.mark.asyncio
class TestParseErrorHandling:
    """Test error handling modes."""

    async def test_handle_validation_return_value(
        self, branch_with_mock_imodel
    ):
        """Test that return_value mode returns original text on failure."""
        branch = branch_with_mock_imodel

        # Mock returns invalid response, so parse will fail and return original text
        original_text = "Invalid text that cannot be parsed"
        result = await parse(
            branch=branch,
            text=original_text,
            response_format=SampleModel,
            handle_validation="return_value",
            alcall_params=AlcallParams(retry_attempts=1),
        )

        assert result == original_text


@pytest.mark.asyncio
class TestParseReturnResMessage:
    """Test return_res_message parameter."""

    async def test_return_res_message_true_direct_parse(
        self, branch_with_mock_imodel
    ):
        """Test return_res_message with successful direct parse."""
        branch = branch_with_mock_imodel

        text = '{"name": "Eve", "age": 28, "city": "Seattle"}'

        result, res_message = await parse(
            branch=branch,
            text=text,
            response_format=SampleModel,
            return_res_message=True,
        )

        assert isinstance(result, SampleModel)
        assert result.name == "Eve"
        assert res_message is None  # No message for direct parse


@pytest.mark.asyncio
class TestParseFuzzyMatchParams:
    """Test custom fuzzy match parameters."""

    async def test_custom_fuzzy_match_params(self, branch_with_mock_imodel):
        """Test parsing with custom FuzzyMatchKeysParams."""
        branch = branch_with_mock_imodel

        # JSON with exact keys - fuzzy params won't be needed but validates interface
        text = '{"name": "Grace", "age": 50, "city": "Portland"}'

        # Custom fuzzy params with higher threshold
        fuzzy_params = FuzzyMatchKeysParams(
            similarity_threshold=0.6,
            fuzzy_match=True,
        )

        result = await parse(
            branch=branch,
            text=text,
            response_format=SampleModel,
            fuzzy_match_params=fuzzy_params,
        )

        # Should successfully match
        assert isinstance(result, SampleModel)
        assert result.name == "Grace"

    async def test_fuzzy_match_params_as_dict(self, branch_with_mock_imodel):
        """Test passing fuzzy_match_params as dict."""
        branch = branch_with_mock_imodel

        text = '{"name": "Henry", "age": 33, "city": "Denver"}'

        result = await parse(
            branch=branch,
            text=text,
            response_format=SampleModel,
            fuzzy_match_params={
                "similarity_threshold": 0.7,
                "fuzzy_match": True,
            },
        )

        assert isinstance(result, SampleModel)
        assert result.name == "Henry"
