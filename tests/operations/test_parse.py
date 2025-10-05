# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.operations.parse.parse import (
    ParseExecutor,
    ParseValidator,
    parse,
)
from lionagi.operations.types import ParseContext
from lionagi.protocols.messages.assistant_response import AssistantResponse
from lionagi.protocols.messages.instruction import Instruction


# Test models (not pytest test classes - don't start with "Test")
class SampleModel(BaseModel):
    """Sample model for parse tests."""

    name: str
    age: int
    email: str | None = None


class OutputModel(BaseModel):
    """Output model for LLM fallback."""

    summary: str


# ============================================================================
# P0 - Critical Coverage Tests
# ============================================================================


class TestBasicParsing:
    """P0: Core parsing functionality."""

    @pytest.mark.asyncio
    async def test_parse_with_basemodel_direct_validation(
        self, make_mocked_branch_for_parse
    ):
        """Test immediate successful validation without LLM call."""
        branch = make_mocked_branch_for_parse()

        text = '{"name": "Alice", "age": 30}'

        # Should return validated model without calling branch.chat
        result = await parse(branch, text, request_type=SampleModel)

        assert isinstance(result, SampleModel)
        assert result.name == "Alice"
        assert result.age == 30

    @pytest.mark.asyncio
    async def test_parse_with_dict_format(self, make_mocked_branch_for_parse):
        """Test dict validation with fuzzy matching."""
        branch = make_mocked_branch_for_parse()

        text = '{"user_name": "Bob", "user_age": 25}'
        response_format = {"name": str, "age": int}

        result = await parse(
            branch,
            text,
            response_format=response_format,
            fuzzy_match=True,
            similarity_threshold=0.7,
            handle_unmatched="ignore",
        )

        assert isinstance(result, dict)
        # Fuzzy matching should map user_name → name, user_age → age
        assert result.get("name") == "Bob" or result.get("user_name") == "Bob"
        assert result.get("age") == 25 or result.get("user_age") == 25

    @pytest.mark.asyncio
    async def test_parse_dict_without_fuzzy_params(
        self, make_mocked_branch_for_parse
    ):
        """Test dict validation when fuzzy_match_params is None."""
        branch = make_mocked_branch_for_parse()

        text = '{"exact_key": "value"}'
        response_format = {"exact_key": str}

        result = await parse(
            branch,
            text,
            response_format=response_format,
            fuzzy_match_params=None,
        )

        assert result["exact_key"] == "value"

    @pytest.mark.asyncio
    async def test_parse_with_llm_fallback(self, make_mocked_branch_for_parse):
        """Test parse falls back to LLM when direct validation fails."""
        branch = make_mocked_branch_for_parse()

        # Invalid JSON forces LLM call
        text = "This is not valid JSON"

        result = await parse(branch, text, request_type=OutputModel)

        # Should use mocked LLM response
        assert isinstance(result, OutputModel)
        assert result.summary == "Mocked summary"

    @pytest.mark.asyncio
    async def test_parse_error_handling_raise_mode(
        self, make_mocked_branch_for_parse
    ):
        """Test handle_validation='raise' throws ValueError on failure."""
        branch = make_mocked_branch_for_parse()

        # Mock chat to return unparseable content
        async def _fake_chat_unparseable(branch_arg, *_args, **_kwargs):
            return (
                Instruction.create(instruction="dummy", sender="user"),
                AssistantResponse.create(
                    assistant_response="unparseable content",
                    sender=branch_arg.id,
                    recipient=branch_arg.user,
                ),
            )

        with patch(
            "lionagi.operations.parse.parse.chat",
            new=AsyncMock(side_effect=_fake_chat_unparseable),
        ):
            text = "unparseable content"

            with pytest.raises(ValueError, match="Failed to parse"):
                await parse(
                    branch,
                    text,
                    request_type=SampleModel,
                    handle_validation="raise",
                    max_retries=0,  # Force immediate failure
                )

    @pytest.mark.asyncio
    async def test_parse_error_handling_return_none(
        self, make_mocked_branch_for_parse
    ):
        """Test handle_validation='return_none' returns None on failure."""
        branch = make_mocked_branch_for_parse()

        # Mock chat to return unparseable content
        async def _fake_chat_unparseable(branch_arg, *_args, **_kwargs):
            return (
                Instruction.create(instruction="dummy", sender="user"),
                AssistantResponse.create(
                    assistant_response="unparseable",
                    sender=branch_arg.id,
                    recipient=branch_arg.user,
                ),
            )

        with patch(
            "lionagi.operations.parse.parse.chat",
            new=AsyncMock(side_effect=_fake_chat_unparseable),
        ):
            text = "unparseable"

            result = await parse(
                branch,
                text,
                request_type=SampleModel,
                handle_validation="return_none",
                max_retries=0,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_parse_error_handling_return_value(
        self, make_mocked_branch_for_parse
    ):
        """Test handle_validation='return_value' returns original text."""
        branch = make_mocked_branch_for_parse()

        # Mock chat to return unparseable content
        async def _fake_chat_unparseable(branch_arg, *_args, **_kwargs):
            return (
                Instruction.create(instruction="dummy", sender="user"),
                AssistantResponse.create(
                    assistant_response="original input text",
                    sender=branch_arg.id,
                    recipient=branch_arg.user,
                ),
            )

        with patch(
            "lionagi.operations.parse.parse.chat",
            new=AsyncMock(side_effect=_fake_chat_unparseable),
        ):
            text = "original input text"

            result = await parse(
                branch,
                text,
                request_type=SampleModel,
                handle_validation="return_value",
                max_retries=0,
            )

            assert result == text


# ============================================================================
# P1 - Important Feature Coverage
# ============================================================================


class TestAdvancedFeatures:
    """P1: Advanced features and parameters."""

    @pytest.mark.asyncio
    async def test_parse_with_no_response_format_returns_text(
        self, make_mocked_branch_for_parse
    ):
        """If no target format is provided, the raw text is returned."""
        branch = make_mocked_branch_for_parse()

        text = "raw content"

        result = await parse(branch, text, response_format=None)

        assert result == text

    @pytest.mark.asyncio
    async def test_parse_return_res_message_success(
        self, make_mocked_branch_for_parse
    ):
        """Test return_res_message returns tuple on direct validation."""
        branch = make_mocked_branch_for_parse()

        text = '{"key": "value"}'

        result, res_msg = await parse(
            branch,
            text,
            response_format={"key": str},
            return_res_message=True,
        )

        assert result["key"] == "value"
        # When direct validation succeeds, res_msg should be None
        assert res_msg is None

    @pytest.mark.asyncio
    async def test_parse_return_res_message_with_llm(
        self, make_mocked_branch_for_parse
    ):
        """Test return_res_message includes AssistantResponse after LLM."""
        branch = make_mocked_branch_for_parse()

        text = "invalid json needing LLM"

        result, res_msg = await parse(
            branch,
            text,
            request_type=OutputModel,
            return_res_message=True,
        )

        assert isinstance(result, OutputModel)
        assert res_msg is not None
        assert res_msg.metadata.get("is_parsed") is True
        assert res_msg.metadata.get("original_text") == text

    @pytest.mark.asyncio
    async def test_parse_honors_custom_prompt_fields(
        self, make_mocked_branch_for_parse
    ):
        """Custom prompt values should be forwarded to branch.chat."""
        branch = make_mocked_branch_for_parse()

        captured: dict = {}

        async def _fake_chat(
            branch_arg, instruction, chat_ctx, return_ins_res_message
        ):
            assert branch_arg is branch
            captured.update(
                instruction=instruction,
                guidance=chat_ctx.guidance,
                context=chat_ctx.context,
            )
            return (
                Instruction.create(instruction="dummy", sender="user"),
                AssistantResponse.create(
                    assistant_response='{"key": "value"}',
                    sender=branch_arg.id,
                    recipient=branch_arg.user,
                ),
            )

        parse_ctx = ParseContext(
            response_format={"key": str},
            fuzzy_match_params=FuzzyMatchKeysParams(),
            handle_validation="return_value",
            alcall_params={"retry_attempts": 1},
            imodel=branch.chat_model,
            imodel_kw={},
            format_instruction="custom instruction",
            format_guidance="custom guidance",
            format_context=[{"custom": "ctx"}],
        )

        with patch(
            "lionagi.operations.parse.parse.chat",
            new=AsyncMock(side_effect=_fake_chat),
        ):
            result = await parse(
                branch,
                "not json",
                parse_ctx=parse_ctx,
                return_res_message=False,
            )

        assert result["key"] == "value"
        assert captured["instruction"] == "custom instruction"
        assert captured["guidance"] == "custom guidance"
        assert captured["context"] == [{"custom": "ctx"}]

    @pytest.mark.asyncio
    async def test_validate_dict_or_model_fuzzy_params_as_dict(self):
        """Test ParseValidator.validate_dict_or_model handles fuzzy_match_params as dict."""
        text = '{"usr_name": "Charlie"}'
        response_format = {"name": str}
        fuzzy_params_dict = {
            "similarity_threshold": 0.6,
            "fuzzy_match": True,
            "handle_unmatched": "ignore",
        }

        result = ParseValidator.validate_dict_or_model(
            text, response_format, fuzzy_params_dict
        )

        assert isinstance(result, dict)
        # Fuzzy matching should map usr_name → name
        assert (
            result.get("name") == "Charlie"
            or result.get("usr_name") == "Charlie"
        )

    @pytest.mark.asyncio
    async def test_parse_alcall_params_as_dict(
        self, make_mocked_branch_for_parse
    ):
        """Test parse converts alcall_params dict to AlcallParams."""
        branch = make_mocked_branch_for_parse()

        parse_ctx = ParseContext(
            response_format={"key": str},
            fuzzy_match_params=FuzzyMatchKeysParams(),
            handle_validation="return_value",
            alcall_params={"retry_attempts": 2, "max_concurrent": 1},
            imodel=branch.chat_model,
            imodel_kw={},
        )

        result = await parse(branch, '{"key": "val"}', parse_ctx=parse_ctx)

        assert result["key"] == "val"

    def test_default_alcall_params(self):
        """Test ParseExecutor.DEFAULT_ALCALL_PARAMS has correct defaults."""
        default_params = ParseExecutor.DEFAULT_ALCALL_PARAMS

        assert default_params is not None
        assert default_params.retry_attempts == 3
        assert default_params.retry_backoff == 1.85
        assert default_params.max_concurrent == 1
        assert default_params.retry_initial_delay == 1
        assert default_params.throttle_period == 1


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def make_mocked_branch_for_parse():
    """Factory fixture for creating branches with mocked parse responses."""

    def _make_branch():
        from unittest.mock import AsyncMock, patch

        from lionagi.protocols.generic.event import EventStatus
        from lionagi.service.connections.api_calling import APICalling
        from lionagi.service.connections.endpoint import Endpoint
        from lionagi.service.connections.providers.oai_ import _get_oai_config
        from lionagi.service.imodel import iModel
        from lionagi.service.third_party.openai_models import (
            OpenAIChatCompletionsRequest,
        )
        from lionagi.session.branch import Branch

        branch = Branch(imodel=iModel(provider="openai", model="gpt-4.1-mini"))

        # Mock imodel.invoke for when parse calls chat internally
        async def _fake_invoke(**kwargs):
            config = _get_oai_config(
                name="oai_chat",
                endpoint="chat/completions",
                request_options=OpenAIChatCompletionsRequest,
                kwargs={"model": "gpt-4.1-mini"},
            )
            endpoint = Endpoint(config=config)
            fake_call = APICalling(
                payload={"model": "gpt-4.1-mini", "messages": []},
                headers={"Authorization": "Bearer test"},
                endpoint=endpoint,
            )
            fake_call.execution.response = '{"summary": "Mocked summary"}'
            fake_call.execution.status = EventStatus.COMPLETED
            return fake_call

        branch.chat_model.invoke = AsyncMock(side_effect=_fake_invoke)

        return branch

    return _make_branch
