# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
P0 Tests for OpenAI Compatible Service Implementation.

CRITICAL: ProactiveDeadlineEnforcement test validates that services actively
enforce CallContext deadlines using fail_at() and don't wait for external timeouts.
This is a core requirement from Ocean for the v1 implementation.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import anyio
import openai
import pytest
from anyio.testing import MockClock
from openai import AsyncOpenAI

from lionagi.errors import (
    NonRetryableError,
    RateLimitError,
    RetryableError,
    ServiceError,
    TimeoutError,
)
from lionagi.services.core import CallContext
from lionagi.services.endpoint import ChatRequestModel
from lionagi.services.openai import OpenAICompatibleService


class TestOpenAIServiceCore:
    """Core functionality tests for OpenAICompatibleService."""

    @pytest.fixture
    def mock_client(self):
        """Create mock AsyncOpenAI client."""
        client = AsyncMock(spec=AsyncOpenAI)
        client.chat = AsyncMock()
        client.chat.completions = AsyncMock()
        return client

    @pytest.fixture
    def service(self, mock_client):
        """Create OpenAICompatibleService with mocked client."""
        return OpenAICompatibleService(
            client=mock_client, name="test_openai", requires={"net.out:api.openai.com"}
        )

    @pytest.fixture
    def chat_request(self):
        """Create test chat request."""
        return ChatRequestModel(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=100,
        )

    @pytest.fixture
    def call_context(self):
        """Create test call context."""
        return CallContext.new(branch_id=uuid4())

    async def test_basic_call_success(self, service, mock_client, chat_request, call_context):
        """Test successful OpenAI API call."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "test-completion",
            "choices": [{"message": {"content": "Hello back!"}}],
            "model": "gpt-3.5-turbo",
            "usage": {"total_tokens": 25},
        }
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Execute call
        result = await service.call(chat_request, ctx=call_context)

        # Verify result
        assert result["id"] == "test-completion"
        assert result["choices"][0]["message"]["content"] == "Hello back!"

        # Verify client was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "gpt-3.5-turbo"
        assert call_args[1]["messages"] == [{"role": "user", "content": "Hello"}]
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["max_tokens"] == 100
        assert call_args[1]["stream"] == False


class TestProactiveDeadlineEnforcement:
    """
    CRITICAL TEST: Validates that services actively enforce CallContext deadlines
    using fail_at() and cancel operations that exceed deadline.

    This is the most important test according to Ocean's requirements.
    """

    @pytest.fixture
    def slow_mock_client(self):
        """Create mock client that simulates slow API responses."""
        client = AsyncMock(spec=AsyncOpenAI)
        client.chat = AsyncMock()
        client.chat.completions = AsyncMock()

        async def slow_create(*args, **kwargs):
            """Simulate slow API call (500ms)."""
            await asyncio.sleep(0.5)  # 500ms delay
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {"id": "slow-response"}
            return mock_response

        client.chat.completions.create = slow_create
        return client

    @pytest.fixture
    def service_with_slow_client(self, slow_mock_client):
        """Service configured with slow mock client."""
        return OpenAICompatibleService(
            client=slow_mock_client,
            name="slow_openai",
            requires={"net.out:api.openai.com"},
        )

    @pytest.fixture
    def short_deadline_context(self):
        """CallContext with short deadline (100ms)."""
        return CallContext.with_timeout(branch_id=uuid4(), timeout_s=0.1)

    @pytest.fixture
    def basic_request(self):
        """Basic chat request for deadline tests."""
        return ChatRequestModel(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Test"}]
        )

    @pytest.mark.anyio
    async def test_proactive_deadline_enforcement_call(
        self, service_with_slow_client, basic_request, short_deadline_context
    ):
        """
        CRITICAL: Validate that call() actively enforces deadlines using fail_at().

        Mock OpenAI client takes 500ms, CallContext deadline is 100ms.
        Service MUST cancel operation and raise TimeoutError after 100ms,
        NOT wait for the full 500ms client timeout.
        """
        start_time = time.time()

        with pytest.raises(anyio.get_cancelled_exc_class()):
            # The service should use fail_at(ctx.deadline_s) which will raise
            # the cancellation exception when deadline is exceeded
            await service_with_slow_client.call(basic_request, ctx=short_deadline_context)

        elapsed_time = time.time() - start_time

        # Assert that timeout occurred close to deadline (100ms), not client timeout (500ms)
        assert elapsed_time < 0.2, f"Expected timeout around 100ms, got {elapsed_time * 1000:.1f}ms"
        assert elapsed_time > 0.08, f"Timeout too fast: {elapsed_time * 1000:.1f}ms"

    @pytest.mark.anyio
    async def test_proactive_deadline_enforcement_stream(
        self, service_with_slow_client, basic_request, short_deadline_context
    ):
        """
        CRITICAL: Validate that stream() actively enforces deadlines using fail_at().
        """

        # Configure slow streaming response
        async def slow_stream_create(*args, **kwargs):
            """Simulate slow streaming response."""
            await asyncio.sleep(0.5)  # 500ms delay before first chunk
            mock_chunk = MagicMock()
            mock_chunk.model_dump.return_value = {"delta": {"content": "chunk"}}

            async def async_generator():
                yield mock_chunk

            return async_generator()

        service_with_slow_client.client.chat.completions.create = slow_stream_create

        start_time = time.time()

        with pytest.raises(anyio.get_cancelled_exc_class()):
            # Stream should enforce deadline and cancel before slow response
            async for _ in service_with_slow_client.stream(
                basic_request, ctx=short_deadline_context
            ):
                pass

        elapsed_time = time.time() - start_time

        # Assert deadline enforcement (100ms), not waiting for slow stream (500ms)
        assert elapsed_time < 0.2, f"Expected timeout around 100ms, got {elapsed_time * 1000:.1f}ms"
        assert elapsed_time > 0.08, f"Timeout too fast: {elapsed_time * 1000:.1f}ms"

    @pytest.mark.anyio
    async def test_no_deadline_allows_completion(self, service_with_slow_client, basic_request):
        """Validate that calls without deadlines complete normally, even if slow."""
        context_no_deadline = CallContext.new(branch_id=uuid4())  # No deadline set

        # Should complete successfully despite 500ms delay
        result = await service_with_slow_client.call(basic_request, ctx=context_no_deadline)

        assert result["id"] == "slow-response"


class TestContextPropagationToSDK:
    """Test that CallContext remaining time propagates to SDK timeout parameter."""

    @pytest.fixture
    def mock_client_with_timeout_capture(self):
        """Mock client that captures timeout parameter."""
        client = AsyncMock(spec=AsyncOpenAI)
        client.chat = AsyncMock()
        client.chat.completions = AsyncMock()

        # Store captured kwargs for verification
        client._captured_kwargs = None

        async def capture_create(*args, **kwargs):
            client._captured_kwargs = kwargs
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {"id": "test"}
            return mock_response

        client.chat.completions.create = capture_create
        return client

    @pytest.fixture
    def service_with_timeout_capture(self, mock_client_with_timeout_capture):
        """Service with timeout capturing client."""
        return OpenAICompatibleService(client=mock_client_with_timeout_capture, name="timeout_test")

    async def test_context_timeout_propagation(self, service_with_timeout_capture, basic_request):
        """Test that remaining time from CallContext propagates to SDK timeout."""
        # Create context with 5 seconds remaining
        context = CallContext.with_timeout(branch_id=uuid4(), timeout_s=5.0)

        # Execute call
        await service_with_timeout_capture.call(basic_request, ctx=context)

        # Verify timeout was passed to SDK
        captured_kwargs = service_with_timeout_capture.client._captured_kwargs
        assert "timeout" in captured_kwargs

        # Should be close to 5.0 seconds (allowing for small execution delay)
        timeout_value = captured_kwargs["timeout"]
        assert 4.8 <= timeout_value <= 5.0, f"Expected timeout ~5.0, got {timeout_value}"

    async def test_minimum_timeout_enforcement(self, service_with_timeout_capture, basic_request):
        """Test that minimum timeout of 1.0s is enforced."""
        # Create context with very short remaining time (0.1s)
        context = CallContext.with_timeout(branch_id=uuid4(), timeout_s=0.1)

        # Small delay to ensure remaining time is very small
        await asyncio.sleep(0.05)

        await service_with_timeout_capture.call(basic_request, ctx=context)

        # Should enforce minimum timeout of 1.0s
        captured_kwargs = service_with_timeout_capture.client._captured_kwargs
        assert captured_kwargs["timeout"] == 1.0


class TestBuildCallKwargsValidation:
    """Test _build_call_kwargs validation - only non-default fields included."""

    @pytest.fixture
    def service(self):
        """Service for kwargs testing."""
        return OpenAICompatibleService(client=AsyncMock(spec=AsyncOpenAI), name="kwargs_test")

    def test_minimal_payload_with_defaults(self, service):
        """Test that only non-default fields are included in kwargs."""
        request = ChatRequestModel(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            # All other fields use defaults
        )
        context = CallContext.new(branch_id=uuid4())

        kwargs = service._build_call_kwargs(request, context)

        # Should only include explicitly set fields, not defaults
        expected_fields = {"model", "messages", "stream"}
        actual_fields = set(kwargs.keys())

        # Should not include default values like temperature=1.0, top_p=1.0, etc.
        assert "temperature" not in kwargs
        assert "top_p" not in kwargs
        assert "frequency_penalty" not in kwargs
        assert "presence_penalty" not in kwargs

        # Should include explicitly set fields
        assert kwargs["model"] == "gpt-4"
        assert kwargs["messages"] == [{"role": "user", "content": "test"}]
        assert kwargs["stream"] == False

    def test_non_default_values_included(self, service):
        """Test that non-default values are included in kwargs."""
        request = ChatRequestModel(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.5,  # Non-default
            max_tokens=200,  # Non-default
            stop=["END"],  # Non-default
        )
        context = CallContext.new(branch_id=uuid4())

        kwargs = service._build_call_kwargs(request, context)

        # Should include non-default values
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 200
        assert kwargs["stop"] == ["END"]

        # Should still exclude default values
        assert "top_p" not in kwargs
        assert "frequency_penalty" not in kwargs


class TestStreamYieldsImmediately:
    """Test that streaming yields immediately without buffering."""

    @pytest.fixture
    def streaming_mock_client(self):
        """Mock client that yields chunks with timing control."""
        client = AsyncMock(spec=AsyncOpenAI)
        client.chat = AsyncMock()
        client.chat.completions = AsyncMock()

        async def create_stream(*args, **kwargs):
            """Create stream that yields chunks with timing."""
            chunks = [
                {"delta": {"content": "chunk1"}},
                {"delta": {"content": "chunk2"}},
                {"delta": {"content": "chunk3"}},
            ]

            async def chunk_generator():
                for i, chunk_data in enumerate(chunks):
                    # Small delay between chunks
                    if i > 0:
                        await asyncio.sleep(0.01)

                    mock_chunk = MagicMock()
                    mock_chunk.model_dump.return_value = chunk_data
                    yield mock_chunk

            return chunk_generator()

        client.chat.completions.create = create_stream
        return client

    @pytest.fixture
    def streaming_service(self, streaming_mock_client):
        """Service configured for streaming tests."""
        return OpenAICompatibleService(client=streaming_mock_client, name="streaming_test")

    async def test_stream_yields_immediately(self, streaming_service):
        """Test that stream yields chunks immediately without buffering."""
        request = ChatRequestModel(
            model="gpt-4", messages=[{"role": "user", "content": "test"}], stream=True
        )
        context = CallContext.new(branch_id=uuid4())

        chunks = []
        chunk_times = []
        start_time = time.time()

        async for chunk in streaming_service.stream(request, ctx=context):
            chunks.append(chunk)
            chunk_times.append(time.time() - start_time)

        # Verify we got all chunks
        assert len(chunks) == 3
        assert chunks[0]["delta"]["content"] == "chunk1"
        assert chunks[1]["delta"]["content"] == "chunk2"
        assert chunks[2]["delta"]["content"] == "chunk3"

        # Verify first chunk arrived quickly (no buffering)
        assert (
            chunk_times[0] < 0.005
        ), f"First chunk took {chunk_times[0] * 1000:.1f}ms - too slow, might be buffering"

        # Verify chunks arrived with expected spacing
        if len(chunk_times) > 1:
            inter_chunk_delay = chunk_times[1] - chunk_times[0]
            assert (
                0.008 <= inter_chunk_delay <= 0.020
            ), f"Inter-chunk delay: {inter_chunk_delay * 1000:.1f}ms"


class TestOpenAISDKExceptionMapping:
    """Test mapping of OpenAI SDK exceptions to lionagi error hierarchy."""

    @pytest.fixture
    def service(self):
        """Service for exception mapping tests."""
        return OpenAICompatibleService(client=AsyncMock(spec=AsyncOpenAI), name="exception_test")

    @pytest.fixture
    def basic_request(self):
        """Basic request for exception tests."""
        return ChatRequestModel(model="gpt-4", messages=[{"role": "user", "content": "test"}])

    @pytest.fixture
    def context(self):
        """Context for exception tests."""
        return CallContext.new(branch_id=uuid4())

    async def test_rate_limit_error_mapping(self, service, basic_request, context):
        """Test RateLimitError mapping with retry_after."""
        # Mock RateLimitError
        rate_limit_error = openai.RateLimitError(
            message="Rate limit exceeded", response=MagicMock(), body={}
        )
        rate_limit_error.retry_after = 30.0

        service.client.chat.completions.create = AsyncMock(side_effect=rate_limit_error)

        with pytest.raises(RateLimitError) as exc_info:
            await service.call(basic_request, ctx=context)

        error = exc_info.value
        assert error.retry_after == 30.0
        assert "rate limited" in error.message.lower()
        assert error.context["service"] == "exception_test"
        assert error.context["call_id"] == str(context.call_id)

    async def test_connection_error_mapping(self, service, basic_request, context):
        """Test APIConnectionError maps to RetryableError."""
        connection_error = openai.APIConnectionError(
            message="Connection failed", request=MagicMock()
        )

        service.client.chat.completions.create = AsyncMock(side_effect=connection_error)

        with pytest.raises(RetryableError) as exc_info:
            await service.call(basic_request, ctx=context)

        error = exc_info.value
        assert "connection error" in error.message.lower()
        assert error.context["error_type"] == "connection"
        assert error.retryable == True

    async def test_server_error_mapping(self, service, basic_request, context):
        """Test InternalServerError maps to RetryableError."""
        server_error = openai.InternalServerError(
            message="Internal server error", response=MagicMock(), body={}
        )
        server_error.status_code = 500

        service.client.chat.completions.create = AsyncMock(side_effect=server_error)

        with pytest.raises(RetryableError) as exc_info:
            await service.call(basic_request, ctx=context)

        error = exc_info.value
        assert "server error" in error.message.lower()
        assert error.context["error_type"] == "server_error"
        assert error.context["status_code"] == 500

    async def test_bad_request_error_mapping(self, service, basic_request, context):
        """Test BadRequestError maps to NonRetryableError."""
        bad_request_error = openai.BadRequestError(
            message="Bad request", response=MagicMock(), body={}
        )
        bad_request_error.status_code = 400

        service.client.chat.completions.create = AsyncMock(side_effect=bad_request_error)

        with pytest.raises(NonRetryableError) as exc_info:
            await service.call(basic_request, ctx=context)

        error = exc_info.value
        assert "bad request" in error.message.lower()
        assert error.context["error_type"] == "bad_request"
        assert error.retryable == False

    async def test_authentication_error_mapping(self, service, basic_request, context):
        """Test AuthenticationError maps to NonRetryableError."""
        auth_error = openai.AuthenticationError(
            message="Invalid API key", response=MagicMock(), body={}
        )
        auth_error.status_code = 401

        service.client.chat.completions.create = AsyncMock(side_effect=auth_error)

        with pytest.raises(NonRetryableError) as exc_info:
            await service.call(basic_request, ctx=context)

        error = exc_info.value
        assert "authentication failed" in error.message.lower()
        assert error.context["error_type"] == "authentication"
        assert error.retryable == False

    async def test_timeout_error_mapping(self, service, basic_request, context):
        """Test asyncio.TimeoutError maps to TimeoutError."""
        service.client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError("Request timed out")
        )

        with pytest.raises(TimeoutError) as exc_info:
            await service.call(basic_request, ctx=context)

        error = exc_info.value
        assert "timed out" in error.message.lower()
        assert error.retryable == True  # TimeoutError is retryable

    async def test_generic_openai_error_mapping(self, service, basic_request, context):
        """Test generic OpenAIError maps to ServiceError."""
        generic_error = openai.OpenAIError(
            message="Unknown error",
        )
        generic_error.status_code = 418  # I'm a teapot

        service.client.chat.completions.create = AsyncMock(side_effect=generic_error)

        with pytest.raises(ServiceError) as exc_info:
            await service.call(basic_request, ctx=context)

        error = exc_info.value
        assert "openai api error" in error.message.lower()
        assert error.context["error_type"] == "openai_api"
        assert error.context["status_code"] == 418


class TestServiceCapabilityDeclaration:
    """Test service capability declaration and enforcement."""

    def test_default_capability_requirements(self):
        """Test default capability requirements."""
        service = OpenAICompatibleService(
            client=AsyncMock(spec=AsyncOpenAI), name="capability_test"
        )

        # Default should require wildcard net.out
        assert service.requires == frozenset({"net.out:*"})

    def test_specific_capability_requirements(self):
        """Test service with specific capability requirements."""
        service = OpenAICompatibleService(
            client=AsyncMock(spec=AsyncOpenAI),
            name="openai_specific",
            requires={"net.out:api.openai.com"},
        )

        assert service.requires == {"net.out:api.openai.com"}

    def test_factory_function_capabilities(self):
        """Test that factory functions set correct capabilities."""
        from lionagi.services.openai import (
            create_anthropic_service,
            create_openai_service,
        )

        # Mock the AsyncOpenAI constructor to avoid actual API calls
        with patch("lionagi.services.openai.AsyncOpenAI") as mock_openai_class:
            mock_client = AsyncMock(spec=AsyncOpenAI)
            mock_openai_class.return_value = mock_client

            openai_service = create_openai_service(api_key="test-key")
            assert openai_service.requires == {"net.out:api.openai.com"}

            anthropic_service = create_anthropic_service(api_key="test-key")
            assert anthropic_service.requires == {"net.out:api.anthropic.com"}
