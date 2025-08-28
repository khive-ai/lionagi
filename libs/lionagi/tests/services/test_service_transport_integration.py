# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
P0 Comprehensive Service-Transport Integration Tests for lionagi v1.

End-to-end behavioral validation consolidating transport and service layer testing:
- Complete error propagation pipeline (HTTP → OpenAI SDK → LionError hierarchy)
- Deadline enforcement through full request cycle
- Streaming response handling with chunk transformation
- Service capability validation and context propagation
- Request/response cycle with transport-service coordination

Consolidates test_transport_layer.py (663 lines) + test_service_integration.py (658 lines)
into comprehensive integration testing focused on behavioral validation.
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import anyio
import httpx
import msgspec
import openai
import pytest
from httpx import MockTransport, Request, Response
from openai import AsyncOpenAI

from lionagi.errors import (
    NonRetryableError,
    PolicyError,
    RateLimitError,
    RetryableError,
    ServiceError,
    TimeoutError,
    TransportError,
)
from lionagi.services.core import CallContext, Service
from lionagi.services.endpoint import ChatRequestModel, RequestModel
from lionagi.services.imodel import iModel
from lionagi.services.openai import OpenAICompatibleService, create_openai_service
from lionagi.services.provider_registry import ProviderAdapter, get_provider_registry
from lionagi.services.transport import HTTPXTransport

# ==============================================================================
# Shared Mock Utilities (Consolidates Duplicate Setup Code)
# ==============================================================================


class MockOpenAIAdapter(ProviderAdapter):
    """Mock OpenAI adapter for service transport integration testing."""

    name = "openai"
    default_base_url = "https://api.openai.com/v1"
    request_model = ChatRequestModel
    requires = {"net.out:api.openai.com"}
    ConfigModel = None

    def __init__(self, service_instance: Service = None):
        self._service_instance = service_instance

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        return (provider or "").lower() == "openai" or (model or "").lower().startswith("openai/")

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service:
        if self._service_instance:
            return self._service_instance
        # Fallback to create real service for these integration tests
        return create_openai_service(api_key=kwargs.get("api_key", "test-key"))

    def required_rights(self, *, base_url: str | None, **kwargs: Any) -> set[str]:
        return {"net.out:api.openai.com"}


def setup_mock_openai_registry(service_instance: Service = None):
    """Setup provider registry with mock OpenAI adapter."""
    registry = get_provider_registry()
    mock_adapter = MockOpenAIAdapter(service_instance)

    # Register or update mock adapter
    registry._adapters["openai"] = mock_adapter


class MockTransportBuilder:
    """Unified mock transport builder for all test scenarios."""

    @staticmethod
    def create_status_response(
        status_code: int,
        content: dict[str, Any] = None,
        extra_headers: dict[str, str] = None,
    ) -> MockTransport:
        """Create mock transport that returns specific HTTP status."""
        if content is None:
            content = {"error": f"HTTP {status_code} error"}

        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        def handler(request: Request) -> Response:
            return Response(
                status_code=status_code,
                content=json.dumps(content).encode(),
                headers=headers,
            )

        return MockTransport(handler)

    @staticmethod
    def create_timeout_transport() -> MockTransport:
        """Create transport that simulates timeout."""

        def timeout_handler(request: Request) -> Response:
            raise httpx.TimeoutException("Request timed out")

        return MockTransport(timeout_handler)

    @staticmethod
    def create_network_error_transport() -> MockTransport:
        """Create transport that simulates network error."""

        def network_error_handler(request: Request) -> Response:
            raise httpx.NetworkError("Network unreachable")

        return MockTransport(network_error_handler)

    @staticmethod
    def create_streaming_transport(
        chunks: list[dict[str, Any]], delay: float = 0.01
    ) -> MockTransport:
        """Create transport that returns streaming responses."""

        def streaming_handler(request: Request) -> Response:
            def generate_chunks():
                for i, chunk in enumerate(chunks):
                    if i > 0 and delay > 0:
                        time.sleep(delay)
                    chunk_json = json.dumps(chunk)
                    yield f"data: {chunk_json}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return Response(
                200,
                content=generate_chunks(),
                headers={"Content-Type": "text/event-stream"},
            )

        return MockTransport(streaming_handler)


class ServiceTestBuilder:
    """Unified service builder for integration testing."""

    @staticmethod
    def create_mock_openai_service(
        mock_transport: MockTransport = None, service_name: str = "test_service"
    ) -> OpenAICompatibleService:
        """Create OpenAI service with mock transport integration."""
        if mock_transport:
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client_instance = httpx.AsyncClient(transport=mock_transport)
                mock_client_class.return_value = mock_client_instance
                service = create_openai_service(api_key="test-key")
                service.name = service_name  # Override the service name after creation
                return service
        else:
            # Return service with mocked OpenAI client
            mock_client = AsyncMock(spec=AsyncOpenAI)
            return OpenAICompatibleService(client=mock_client, name=service_name)

    @staticmethod
    def create_standard_request(stream: bool = False) -> ChatRequestModel:
        """Create standard test request."""
        return ChatRequestModel(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            stream=stream,
        )

    @staticmethod
    def create_test_context(timeout_s: float = 30.0, **attrs) -> CallContext:
        """Create test context with optional timeout and attributes."""
        return CallContext.with_timeout(
            branch_id=uuid4(),
            timeout_s=timeout_s,
            capabilities={"net.out:api.openai.com"},
            attrs=attrs,
        )


# ==============================================================================
# End-to-End Error Propagation Pipeline Tests
# ==============================================================================


class TestErrorPropagationPipeline:
    """Test complete error propagation from HTTP transport through service to application."""

    @pytest.fixture
    def error_mapping_cases(self):
        """Define comprehensive error mapping test cases."""
        return [
            # (HTTP status, expected error type, retryable, extra validation)
            (429, RateLimitError, True, lambda e: e.retry_after > 0),
            (500, RetryableError, True, lambda e: "Server error: 500" in e.message),
            (502, RetryableError, True, lambda e: "Server error: 502" in e.message),
            (503, RetryableError, True, lambda e: "Server error: 503" in e.message),
            (400, NonRetryableError, False, lambda e: "Client error: 400" in e.message),
            (401, NonRetryableError, False, lambda e: "Client error: 401" in e.message),
            (403, NonRetryableError, False, lambda e: "Client error: 403" in e.message),
            (404, NonRetryableError, False, lambda e: "Client error: 404" in e.message),
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "status_code,error_type,retryable,validation",
        [
            (429, RateLimitError, True, lambda e: e.retry_after > 0),
            (500, RetryableError, True, lambda e: "Server error: 500" in e.message),
            (502, RetryableError, True, lambda e: "Server error: 502" in e.message),
            (503, RetryableError, True, lambda e: "Server error: 503" in e.message),
            (400, NonRetryableError, False, lambda e: "Client error: 400" in e.message),
            (401, NonRetryableError, False, lambda e: "Client error: 401" in e.message),
            (403, NonRetryableError, False, lambda e: "Client error: 403" in e.message),
            (404, NonRetryableError, False, lambda e: "Client error: 404" in e.message),
        ],
    )
    async def test_http_status_to_service_error_propagation(
        self, status_code, error_type, retryable, validation
    ):
        """Test HTTP status codes propagate correctly through service layer."""
        # Create transport that returns specific HTTP status
        mock_transport = MockTransportBuilder.create_status_response(
            status_code,
            {"error": f"Test error {status_code}"},
            {"Retry-After": "60"} if status_code == 429 else None,
        )

        # Create service with mock transport
        service = ServiceTestBuilder.create_mock_openai_service(mock_transport, "error_test")
        request = ServiceTestBuilder.create_standard_request()
        context = ServiceTestBuilder.create_test_context()

        # Test end-to-end error propagation
        with pytest.raises(error_type) as exc_info:
            await service.call(request, ctx=context)

        error = exc_info.value
        assert error.retryable == retryable
        assert error.context["status_code"] == status_code
        assert error.context["service"] == "error_test"
        assert error.context["call_id"] == str(context.call_id)
        assert validation(error)

    @pytest.mark.anyio
    async def test_network_error_end_to_end_propagation(self):
        """Test network errors propagate as RetryableError through complete pipeline."""
        mock_client = AsyncMock(spec=AsyncOpenAI)
        connection_error = openai.APIConnectionError(
            message="Connection failed", request=MagicMock()
        )
        mock_client.chat.completions.create = AsyncMock(side_effect=connection_error)

        service = OpenAICompatibleService(client=mock_client, name="network_test")
        request = ServiceTestBuilder.create_standard_request()
        context = ServiceTestBuilder.create_test_context()

        with pytest.raises(RetryableError) as exc_info:
            await service.call(request, ctx=context)

        error = exc_info.value
        assert error.retryable == True
        assert error.context["error_type"] == "connection"
        assert "connection error" in error.message.lower()
        assert error.context["service"] == "network_test"

    @pytest.mark.anyio
    async def test_timeout_propagation_through_pipeline(self):
        """Test timeout handling through transport-service integration."""
        mock_transport = MockTransportBuilder.create_timeout_transport()

        # Patch HTTPXTransport to use our mock
        with patch("lionagi.services.transport.HTTPXTransport") as mock_transport_class:
            mock_instance = HTTPXTransport()
            mock_instance._client = httpx.AsyncClient(transport=mock_transport)
            mock_transport_class.return_value = mock_instance

            # Test through service layer
            mock_client = AsyncMock(spec=AsyncOpenAI)
            timeout_error = openai.APITimeoutError(request=MagicMock())
            mock_client.chat.completions.create = AsyncMock(side_effect=timeout_error)

            service = OpenAICompatibleService(client=mock_client, name="timeout_test")
            request = ServiceTestBuilder.create_standard_request()
            context = ServiceTestBuilder.create_test_context(timeout_s=1.0)

            with pytest.raises(TimeoutError) as exc_info:
                await service.call(request, ctx=context)

            error = exc_info.value
            assert error.context["timeout_s"] == 1.0
            assert error.context["service"] == "timeout_test"


# ==============================================================================
# Deadline Enforcement Integration Tests
# ==============================================================================


class TestDeadlineEnforcementIntegration:
    """Test proactive deadline enforcement through complete request pipeline."""

    @pytest.mark.anyio
    async def test_service_deadline_enforcement_proactive(self):
        """Test that service layer enforces deadlines proactively before transport timeout."""

        # Create slow mock that would take 500ms
        def slow_handler(request: Request) -> Response:
            time.sleep(0.5)  # 500ms delay
            return Response(200, json={"id": "slow-response"})

        slow_transport = MockTransport(slow_handler)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = httpx.AsyncClient(transport=slow_transport)
            mock_client_class.return_value = mock_client_instance

            service = create_openai_service(api_key="test-key")

        request = ServiceTestBuilder.create_standard_request()
        context = ServiceTestBuilder.create_test_context(timeout_s=0.1)  # 100ms deadline

        start_time = time.time()

        # Should timeout at deadline (~100ms), not transport timeout (~500ms)
        with pytest.raises(anyio.get_cancelled_exc_class()):
            await service.call(request, ctx=context)

        elapsed_time = time.time() - start_time
        assert elapsed_time < 0.2, f"Expected timeout ~100ms, got {elapsed_time * 1000:.1f}ms"

    @pytest.mark.anyio
    async def test_streaming_deadline_enforcement(self):
        """Test deadline enforcement during streaming operations."""
        chunks = [
            {"delta": {"content": "Hello"}},
            {"delta": {"content": " world"}},
            {"delta": {"content": "!"}},
        ]

        mock_client = AsyncMock(spec=AsyncOpenAI)

        async def slow_streaming_create(*args, **kwargs):
            await asyncio.sleep(0.5)  # 500ms delay before first chunk
            for chunk_data in chunks:
                mock_chunk = MagicMock()
                mock_chunk.model_dump.return_value = chunk_data
                yield mock_chunk

        mock_client.chat.completions.create = slow_streaming_create

        service = OpenAICompatibleService(client=mock_client, name="stream_deadline")
        request = ServiceTestBuilder.create_standard_request(stream=True)
        context = ServiceTestBuilder.create_test_context(timeout_s=0.1)  # 100ms deadline

        start_time = time.time()

        with pytest.raises(anyio.get_cancelled_exc_class()):
            async for chunk in service.stream(request, ctx=context):
                pass

        elapsed_time = time.time() - start_time
        assert (
            elapsed_time < 0.2
        ), f"Expected stream timeout ~100ms, got {elapsed_time * 1000:.1f}ms"


# ==============================================================================
# Comprehensive Streaming Integration Tests
# ==============================================================================


class TestStreamingPipelineIntegration:
    """Test complete streaming pipeline from transport through service."""

    @pytest.fixture
    def standard_streaming_chunks(self):
        """Standard OpenAI streaming response chunks."""
        return [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [{"delta": {"content": "Hello"}, "index": 0}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [{"delta": {"content": " world"}, "index": 0}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [{"delta": {"content": "!"}, "index": 0}],
            },
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            },
        ]

    @pytest.mark.anyio
    async def test_end_to_end_streaming_pipeline(self, standard_streaming_chunks):
        """Test complete streaming from service through transport with chunk transformation."""
        mock_client = AsyncMock(spec=AsyncOpenAI)

        async def mock_streaming_create(*args, **kwargs):
            """Mock streaming that simulates OpenAI SDK behavior."""
            for chunk_data in standard_streaming_chunks:
                mock_chunk = MagicMock()
                mock_chunk.model_dump.return_value = chunk_data
                yield mock_chunk

        mock_client.chat.completions.create = mock_streaming_create

        service = OpenAICompatibleService(client=mock_client, name="streaming_test")
        request = ServiceTestBuilder.create_standard_request(stream=True)
        context = ServiceTestBuilder.create_test_context()

        received_chunks = []
        chunk_times = []
        start_time = time.time()

        async for chunk in service.stream(request, ctx=context):
            received_chunks.append(chunk)
            chunk_times.append(time.time() - start_time)

        # Verify complete streaming pipeline
        assert len(received_chunks) == len(standard_streaming_chunks)
        assert received_chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert received_chunks[1]["choices"][0]["delta"]["content"] == " world"
        assert received_chunks[2]["choices"][0]["delta"]["content"] == "!"
        assert received_chunks[3]["choices"][0]["finish_reason"] == "stop"

        # Verify streaming was immediate (no buffering)
        assert chunk_times[0] < 0.01, "First chunk should arrive immediately"

    @pytest.mark.anyio
    async def test_streaming_error_propagation(self):
        """Test error propagation during streaming operations."""
        mock_client = AsyncMock(spec=AsyncOpenAI)

        async def error_streaming_create(*args, **kwargs):
            # Yield one chunk successfully, then error
            mock_chunk = MagicMock()
            mock_chunk.model_dump.return_value = {"delta": {"content": "Hello"}}
            yield mock_chunk

            # Simulate network error during streaming
            raise openai.APIConnectionError(
                message="Connection lost during streaming", request=MagicMock()
            )

        mock_client.chat.completions.create = error_streaming_create

        service = OpenAICompatibleService(client=mock_client, name="stream_error")
        request = ServiceTestBuilder.create_standard_request(stream=True)
        context = ServiceTestBuilder.create_test_context()

        chunks_received = []

        with pytest.raises(RetryableError) as exc_info:
            async for chunk in service.stream(request, ctx=context):
                chunks_received.append(chunk)

        # Should have received first chunk before error
        assert len(chunks_received) == 1
        assert chunks_received[0]["delta"]["content"] == "Hello"

        error = exc_info.value
        assert error.retryable == True
        assert "connection" in error.message.lower()


# ==============================================================================
# Service Capability and Context Integration Tests
# ==============================================================================


class TestServiceCapabilityIntegration:
    """Test service capability validation and context propagation."""

    @pytest.mark.anyio
    async def test_service_capability_declaration_validation(self):
        """Test that services properly declare and validate capabilities."""
        with patch("lionagi.services.openai.AsyncOpenAI") as mock_openai:
            mock_client = AsyncMock(spec=AsyncOpenAI)
            mock_openai.return_value = mock_client

            # Test different service types have correct capability requirements
            openai_service = create_openai_service(api_key="test")
            assert openai_service.requires == {"net.out:api.openai.com"}

            from lionagi.services.openai import (
                create_anthropic_service,
                create_openrouter_service,
            )

            anthropic_service = create_anthropic_service(api_key="test")
            assert anthropic_service.requires == {"net.out:api.anthropic.com"}

            openrouter_service = create_openrouter_service(api_key="test")
            assert openrouter_service.requires == {"net.out:openrouter.ai"}

            # Verify capability requirements are immutable
            assert isinstance(openai_service.requires, frozenset)

    @pytest.mark.anyio
    async def test_context_propagation_through_service_pipeline(self):
        """Test CallContext data propagation through complete service pipeline."""
        mock_client = AsyncMock(spec=AsyncOpenAI)

        # Capture context data that flows through
        captured_kwargs = None

        async def capture_create(*args, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs

            mock_response = MagicMock()
            mock_response.model_dump.return_value = {
                "id": "test-response",
                "choices": [{"message": {"content": "Response"}}],
            }
            return mock_response

        mock_client.chat.completions.create = capture_create

        service = OpenAICompatibleService(client=mock_client, name="context_test")

        # Create context with specific attributes
        context = ServiceTestBuilder.create_test_context(
            timeout_s=15.0, trace_id="test-trace-123", user_id="user-456"
        )

        request = ServiceTestBuilder.create_standard_request()

        # Execute call and verify context propagation
        result = await service.call(request, ctx=context)

        # Verify timeout propagated to SDK
        assert "timeout" in captured_kwargs
        timeout_value = captured_kwargs["timeout"]
        assert 14.0 <= timeout_value <= 15.0  # Allow for small execution delay

        # Verify context attributes preserved
        assert context.attrs["trace_id"] == "test-trace-123"
        assert context.attrs["user_id"] == "user-456"
        assert context.capabilities == {"net.out:api.openai.com"}

        # Verify result structure
        assert result["id"] == "test-response"

    @pytest.mark.anyio
    async def test_call_id_tracking_through_error_pipeline(self):
        """Test call_id tracking through error propagation pipeline."""
        mock_client = AsyncMock(spec=AsyncOpenAI)

        server_error = openai.InternalServerError(
            message="Server error", response=MagicMock(status_code=500), body={}
        )

        mock_client.chat.completions.create = AsyncMock(side_effect=server_error)

        service = OpenAICompatibleService(client=mock_client, name="tracking_test")
        context = ServiceTestBuilder.create_test_context()
        request = ServiceTestBuilder.create_standard_request()

        with pytest.raises(RetryableError) as exc_info:
            await service.call(request, ctx=context)

        error = exc_info.value
        assert error.context["call_id"] == str(context.call_id)
        assert error.context["service"] == "tracking_test"
        assert context.branch_id  # Verify branch_id consistency


# ==============================================================================
# Complete Pipeline Lifecycle Tests
# ==============================================================================


class TestCompletePipelineLifecycle:
    """Test complete request/response lifecycle through service-transport integration."""

    @pytest.mark.anyio
    async def test_successful_end_to_end_request_response_cycle(self):
        """Test complete successful request cycle from service to transport to response."""
        # Standard OpenAI API response
        mock_response = {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 13, "completion_tokens": 9, "total_tokens": 22},
        }

        # Create transport that validates request and returns response
        def validating_handler(request: Request) -> Response:
            # Validate request structure
            assert request.method == "POST"
            assert "chat/completions" in str(request.url) or "api.openai.com" in str(request.url)
            assert "authorization" in {k.lower() for k in request.headers.keys()}

            return Response(
                status_code=200,
                content=json.dumps(mock_response).encode(),
                headers={"Content-Type": "application/json"},
            )

        mock_transport = MockTransport(validating_handler)

        # Create service with integrated transport and use through iModel
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_instance = httpx.AsyncClient(transport=mock_transport)
            mock_client_class.return_value = mock_client_instance

            # Use iModel for end-to-end testing instead of direct service
            async with iModel(
                provider="openai", model="gpt-3.5-turbo", api_key="test-key"
            ) as model:
                request = ServiceTestBuilder.create_standard_request()

                # Execute complete end-to-end cycle through iModel
                result = await model.invoke(
                    request,
                    capabilities={"net.out:api.openai.com"},
                    timeout_s=30.0,
                )

        # Verify complete response structure
        assert result["id"] == "chatcmpl-test123"
        assert result["model"] == "gpt-3.5-turbo-0613"
        assert result["choices"][0]["message"]["content"] == "Hello! How can I help you today?"
        assert result["usage"]["total_tokens"] == 22

    @pytest.mark.anyio
    async def test_msgspec_integration_through_pipeline(self):
        """Test msgspec JSON parsing integration through transport-service pipeline."""
        complex_response = {
            "id": "test-response-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Complex nested response with performance requirements.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 25, "completion_tokens": 12, "total_tokens": 37},
        }

        mock_transport = MockTransportBuilder.create_status_response(200, complex_response)

        # Test msgspec integration through transport layer
        transport = HTTPXTransport()
        transport._client = httpx.AsyncClient(transport=mock_transport)

        # Patch msgspec.json.decode to verify usage
        with patch("lionagi.services.transport.msgspec.json.decode") as mock_decode:
            mock_decode.return_value = complex_response

            async with transport:
                result = await transport.send_json(
                    method="POST",
                    url="https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": "Bearer test"},
                    json={"model": "gpt-4", "messages": []},
                    timeout_s=30.0,
                )

                # Verify msgspec was used for high-performance JSON parsing
                mock_decode.assert_called_once()

                # Verify result structure preserved through parsing
                assert result["id"] == "test-response-123"
                assert result["usage"]["total_tokens"] == 37

    @pytest.mark.anyio
    async def test_service_middleware_integration_pipeline(self):
        """Test middleware integration through complete service pipeline."""

        async def tracking_middleware(req: RequestModel, ctx: CallContext, next_call):
            """Middleware that adds tracking information."""
            ctx.tracking["middleware_start"] = time.time()
            result = await next_call()
            ctx.tracking["middleware_duration"] = time.time() - ctx.tracking["middleware_start"]
            result["middleware_processed"] = True
            return result

        mock_client = AsyncMock(spec=AsyncOpenAI)

        async def mock_create(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.model_dump.return_value = {
                "id": "middleware-test",
                "content": "response",
            }
            return mock_response

        mock_client.chat.completions.create = mock_create

        # Create service with middleware
        service = OpenAICompatibleService(
            client=mock_client, name="middleware_test", call_mw=(tracking_middleware,)
        )

        context = ServiceTestBuilder.create_test_context()
        request = ServiceTestBuilder.create_standard_request()

        result = await service.call(request, ctx=context)

        # Verify middleware integration
        assert "middleware_start" in context.tracking
        assert "middleware_duration" in context.tracking
        assert context.tracking["middleware_duration"] > 0
        assert result.get("middleware_processed") == True
        assert result["id"] == "middleware-test"
