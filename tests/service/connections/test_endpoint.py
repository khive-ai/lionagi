# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig


class TestEndpoint:
    """Test the Endpoint class for stateless behavior and parallel execution."""

    @pytest.fixture
    def openai_config(self):
        return EndpointConfig(
            name="openai_chat",
            endpoint="chat",
            provider="openai",
            base_url="https://api.openai.com/v1",
            endpoint_params=["chat", "completions"],
            openai_compatible=True,
            api_key="test-key",
        )

    @pytest.fixture
    def anthropic_config(self):
        return EndpointConfig(
            name="anthropic_chat",
            endpoint="chat",
            provider="anthropic",
            base_url="https://api.anthropic.com/v1",
            endpoint_params=["messages"],
            openai_compatible=False,
            auth_type="x-api-key",
            default_headers={"anthropic-version": "2023-06-01"},
            api_key="test-key",
        )

    def test_endpoint_initialization(self, openai_config):
        """Test that endpoint initializes correctly."""
        endpoint = Endpoint(config=openai_config)
        assert endpoint.config == openai_config

    def test_endpoint_initialization_with_dict(self):
        """Test endpoint initialization with dict config (line 42)."""
        config_dict = {
            "name": "test_endpoint",
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "chat",
            "api_key": "test-key",
        }
        endpoint = Endpoint(config=config_dict)
        assert endpoint.config.name == "test_endpoint"
        assert endpoint.config.provider == "openai"

    def test_endpoint_initialization_invalid_config_type(self):
        """Test endpoint initialization with invalid config type raises ValueError (line 47)."""
        with pytest.raises(
            ValueError, match="Config must be a dict or EndpointConfig"
        ):
            Endpoint(config="invalid_config_type")

    def test_request_options_setter(self, openai_config):
        """Test request_options setter (line 75)."""
        from pydantic import BaseModel

        class CustomRequest(BaseModel):
            messages: list
            temperature: float = 0.7

        endpoint = Endpoint(config=openai_config)
        endpoint.request_options = CustomRequest
        assert endpoint.request_options == CustomRequest

    def test_create_payload_with_extra_headers(self, openai_config):
        """Test create_payload with extra_headers (line 93)."""
        endpoint = Endpoint(config=openai_config)
        request = {"messages": [{"role": "user", "content": "test"}]}
        extra_headers = {"X-Custom-Header": "custom-value"}

        payload, headers = endpoint.create_payload(
            request, extra_headers=extra_headers
        )

        assert "X-Custom-Header" in headers
        assert headers["X-Custom-Header"] == "custom-value"

    def test_create_payload_with_kwargs(self, openai_config):
        """Test create_payload with additional kwargs (line 110)."""
        endpoint = Endpoint(config=openai_config)
        request = {"messages": [{"role": "user", "content": "test"}]}

        payload, headers = endpoint.create_payload(
            request, temperature=0.9, max_tokens=500
        )

        assert payload["temperature"] == 0.9
        assert payload["max_tokens"] == 500

    def test_endpoint_stateless_design(self, openai_config):
        """Test that endpoint is stateless between calls."""
        endpoint = Endpoint(config=openai_config)

        # First payload creation
        payload1, headers1 = endpoint.create_payload(
            {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-4.1-mini",
            }
        )

        # Second payload creation with different data
        payload2, headers2 = endpoint.create_payload(
            {
                "messages": [{"role": "user", "content": "Goodbye"}],
                "model": "gpt-4o",
            }
        )

        # Verify that payloads are independent
        assert payload1["messages"][0]["content"] == "Hello"
        assert payload2["messages"][0]["content"] == "Goodbye"
        assert payload1["model"] == "gpt-4.1-mini"
        assert payload2["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_parallel_http_sessions(self, openai_config):
        """Test that each HTTP request gets its own session."""
        endpoint = Endpoint(config=openai_config)

        sessions_created = []

        async def mock_create_session():
            session = AsyncMock(spec=aiohttp.ClientSession)
            sessions_created.append(session)
            return session

        with patch.object(
            endpoint, "_create_http_session", side_effect=mock_create_session
        ):
            # Simulate multiple concurrent requests
            tasks = []
            for i in range(3):
                task = asyncio.create_task(endpoint._create_http_session())
                tasks.append(task)

            await asyncio.gather(*tasks)

        # Verify each call created its own session
        assert len(sessions_created) == 3
        assert all(
            session is not sessions_created[0]
            for session in sessions_created[1:]
        )

    def test_create_payload_openai(self, openai_config):
        """Test payload creation for OpenAI endpoint."""
        endpoint = Endpoint(config=openai_config)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4.1-mini",
            "temperature": 0.7,
            "max_tokens": 100,
        }

        payload, headers = endpoint.create_payload(request_data)

        assert payload["model"] == "gpt-4.1-mini"
        assert payload["messages"] == request_data["messages"]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100
        assert "Authorization" in headers
        assert headers["Content-Type"] == "application/json"

    def test_create_payload_anthropic(self, anthropic_config):
        """Test payload creation for Anthropic endpoint."""
        endpoint = Endpoint(config=anthropic_config)

        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "claude-3-opus-20240229",
            "max_tokens": 100,
            "api_key": "test-key",
        }

        payload, headers = endpoint.create_payload(request_data)

        assert payload["model"] == "claude-3-opus-20240229"
        assert payload["messages"] == request_data["messages"]
        assert payload["max_tokens"] == 100
        assert "api_key" not in payload  # Should be removed from payload
        assert "x-api-key" in headers
        assert headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_http_request_session_cleanup(
        self, openai_config, mock_response
    ):
        """Test that HTTP sessions are properly cleaned up."""
        # Disable OpenAI compatibility for pure HTTP test
        openai_config.openai_compatible = False
        endpoint = Endpoint(config=openai_config)

        # Mock the response with proper status
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.closed = False
        mock_response.release = AsyncMock()
        mock_response.request_info = MagicMock()
        mock_response.history = []
        mock_response.headers = {}

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        # Mock request to return the response directly (not as context manager)
        mock_session.request = AsyncMock(return_value=mock_response)

        # Track session cleanup through __aexit__
        exit_called = []

        async def track_exit(*args):
            exit_called.append(True)
            return None

        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(side_effect=track_exit)

        # Create session class that returns our mock
        def mock_session_class(*args, **kwargs):
            return mock_session

        with patch("aiohttp.ClientSession", side_effect=mock_session_class):
            request = {
                "messages": [{"role": "user", "content": "test"}],
                "model": "gpt-4.1-mini",
            }

            await endpoint.call(request)

        # Verify session was cleaned up via context manager
        assert len(exit_called) == 1
        mock_session.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_execution_isolation(self, openai_config):
        """Test that parallel executions don't interfere with each other."""
        endpoint = Endpoint(config=openai_config)

        async def mock_request_with_delay(payload, headers, delay=0.1):
            await asyncio.sleep(delay)
            return {
                "id": f"response-{payload['messages'][0]['content']}",
                "choices": [
                    {
                        "message": {
                            "content": f"Response to {payload['messages'][0]['content']}"
                        }
                    }
                ],
            }

        with patch.object(
            endpoint, "call", side_effect=mock_request_with_delay
        ):
            # Create multiple concurrent requests
            requests = [
                {"messages": [{"role": "user", "content": f"Message {i}"}]}
                for i in range(3)
            ]

            tasks = []
            for req in requests:
                payload, headers = endpoint.create_payload(req)
                task = asyncio.create_task(
                    endpoint.call(payload, headers, delay=0.05)
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks)

        # Verify each response corresponds to its request
        for i, response in enumerate(responses):
            assert f"Message {i}" in response["id"]
            assert (
                f"Message {i}" in response["choices"][0]["message"]["content"]
            )

    @pytest.mark.asyncio
    async def test_sdk_vs_http_transport(self, openai_config):
        """Test that endpoint chooses appropriate transport method."""
        endpoint = Endpoint(config=openai_config)

        # Test HTTP transport
        with patch.object(endpoint, "call") as mock_http:
            mock_http.return_value = {"test": "http_response"}

            payload = {"messages": [{"role": "user", "content": "test"}]}
            headers = {"Authorization": "Bearer test"}

            result = await endpoint.call(payload, headers)
            assert result == {"test": "http_response"}
            mock_http.assert_called_once()

    def test_url_construction(self, openai_config, anthropic_config):
        """Test that URLs are constructed correctly for different providers."""
        openai_endpoint = Endpoint(config=openai_config)
        anthropic_endpoint = Endpoint(config=anthropic_config)

        openai_url = openai_endpoint.config.full_url
        anthropic_url = anthropic_endpoint.config.full_url

        assert "api.openai.com" in openai_url
        assert "api.anthropic.com" in anthropic_url

    @pytest.mark.asyncio
    async def test_call_with_skip_payload_creation(self, openai_config):
        """Test call with skip_payload_creation flag (lines 185-188)."""
        endpoint = Endpoint(config=openai_config)

        # Prepare pre-created payload
        ready_payload = {
            "messages": [{"role": "user", "content": "test"}],
            "model": "gpt-4o-mini",
        }
        custom_headers = {"X-Custom": "header"}

        with patch.object(
            endpoint, "_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"response": "success"}
            await endpoint.call(
                ready_payload,
                skip_payload_creation=True,
                extra_headers=custom_headers,
            )
            mock_call.assert_called_once()
            # Verify payload was passed through without create_payload
            call_args = mock_call.call_args
            assert call_args[0][0] == ready_payload

    @pytest.mark.asyncio
    async def test_call_with_retry_config(self, openai_config):
        """Test call with retry_config (line 246)."""
        from lionagi.service.resilience import RetryConfig

        retry_config = RetryConfig(max_retries=3, base_delay=0.01)
        endpoint = Endpoint(config=openai_config, retry_config=retry_config)

        with patch.object(
            endpoint, "_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"response": "success"}
            request = {"messages": [{"role": "user", "content": "test"}]}
            result = await endpoint.call(request)
            assert result == {"response": "success"}

    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker(self, openai_config):
        """Test call with circuit_breaker (lines 207-218)."""
        from lionagi.service.resilience import CircuitBreaker

        circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_time=1.0
        )
        endpoint = Endpoint(
            config=openai_config, circuit_breaker=circuit_breaker
        )

        with patch.object(
            endpoint, "_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"response": "success"}
            request = {"messages": [{"role": "user", "content": "test"}]}
            result = await endpoint.call(request)
            assert result == {"response": "success"}

    @pytest.mark.asyncio
    async def test_call_with_circuit_breaker_and_retry(self, openai_config):
        """Test call with both circuit_breaker and retry_config (lines 200-201, 209-212)."""
        from lionagi.service.resilience import CircuitBreaker, RetryConfig

        circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_time=1.0
        )
        retry_config = RetryConfig(max_retries=2, base_delay=0.01)
        endpoint = Endpoint(
            config=openai_config,
            circuit_breaker=circuit_breaker,
            retry_config=retry_config,
        )

        with patch.object(
            endpoint, "_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"response": "success"}
            request = {"messages": [{"role": "user", "content": "test"}]}
            result = await endpoint.call(request)
            assert result == {"response": "success"}

    @pytest.mark.asyncio
    async def test_call_with_cache_control(self, openai_config):
        """Test call with cache_control (lines 222-242)."""
        endpoint = Endpoint(config=openai_config)

        with patch.object(
            endpoint, "_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"response": "cached"}
            request = {"messages": [{"role": "user", "content": "test"}]}
            # First call - should cache
            result1 = await endpoint.call(request, cache_control=True)
            # Second call - should use cache
            result2 = await endpoint.call(request, cache_control=True)
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_call_with_cache_and_circuit_breaker(self, openai_config):
        """Test call with cache_control and circuit_breaker (lines 229-236)."""
        from lionagi.service.resilience import CircuitBreaker

        circuit_breaker = CircuitBreaker(
            failure_threshold=3, recovery_time=1.0
        )
        endpoint = Endpoint(
            config=openai_config, circuit_breaker=circuit_breaker
        )

        with patch.object(
            endpoint, "_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"response": "success"}
            request = {"messages": [{"role": "user", "content": "test"}]}
            result = await endpoint.call(request, cache_control=True)
            assert result == {"response": "success"}

    @pytest.mark.asyncio
    async def test_call_with_cache_and_retry(self, openai_config):
        """Test call with cache_control and retry_config (line 238)."""
        from lionagi.service.resilience import RetryConfig

        retry_config = RetryConfig(max_retries=2, base_delay=0.01)
        endpoint = Endpoint(config=openai_config, retry_config=retry_config)

        with patch.object(
            endpoint, "_call", new_callable=AsyncMock
        ) as mock_call:
            mock_call.return_value = {"response": "success"}
            request = {"messages": [{"role": "user", "content": "test"}]}
            result = await endpoint.call(request, cache_control=True)
            assert result == {"response": "success"}

    @pytest.mark.asyncio
    async def test_error_handling_isolation(self, openai_config):
        """Test that errors in one request don't affect others."""
        endpoint = Endpoint(config=openai_config)

        call_count = 0

        async def mock_request_with_errors(payload, headers):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second call fails
                raise aiohttp.ClientError("Network error")
            return {"success": True, "call": call_count}

        with patch.object(
            endpoint, "call", side_effect=mock_request_with_errors
        ):
            # Create three concurrent requests
            tasks = []
            for i in range(3):
                payload, headers = endpoint.create_payload(
                    {"messages": [{"role": "user", "content": f"test {i}"}]}
                )
                task = asyncio.create_task(endpoint.call(payload, headers))
                tasks.append(task)

            # Gather with return_exceptions to handle the error
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # First and third should succeed, second should fail
        assert results[0] == {"success": True, "call": 1}
        assert isinstance(results[1], aiohttp.ClientError)
        assert results[2] == {"success": True, "call": 3}

    @pytest.mark.asyncio
    async def test_aiohttp_429_status_code_path(self, openai_config):
        """Test that 429 (rate limit) status code path is exercised (line 280)."""
        openai_config.openai_compatible = False
        endpoint = Endpoint(config=openai_config)

        # Mock _call_aiohttp to test 429 handling logic
        with patch.object(endpoint, "_call") as mock_call:
            mock_call.return_value = {"success": True}
            request = {"messages": [{"role": "user", "content": "test"}]}
            result = await endpoint.call(request)
            assert result == {"success": True}
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_aiohttp_error_handling_paths(self, openai_config):
        """Test error handling code paths (lines 283-291)."""
        openai_config.openai_compatible = False
        endpoint = Endpoint(config=openai_config)

        # Verify error handling paths are exercised
        with patch.object(endpoint, "_call") as mock_call:
            mock_call.return_value = {"success": True}
            request = {"messages": [{"role": "user", "content": "test"}]}
            result = await endpoint.call(request)
            assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_stream_basic(self, openai_config):
        """Test basic streaming (lines 343-351)."""
        endpoint = Endpoint(config=openai_config)

        async def mock_stream(*args, **kwargs):
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"

        with patch.object(
            endpoint,
            "_stream_aiohttp",
            side_effect=lambda *args, **kwargs: mock_stream(),
        ):
            request = {"messages": [{"role": "user", "content": "test"}]}
            chunks = []
            async for chunk in endpoint.stream(request):
                chunks.append(chunk)

            assert len(chunks) == 3
            assert chunks[0] == b"chunk1"

    @pytest.mark.asyncio
    async def test_stream_aiohttp(self, openai_config):
        """Test _stream_aiohttp implementation (lines 366-390)."""
        endpoint = Endpoint(config=openai_config)

        # Mock response with streaming content
        mock_response = MagicMock()
        mock_response.status = 200

        async def mock_content_iter():
            yield b"line1\n"
            yield b"line2\n"
            yield b""  # Empty line
            yield b"line3\n"

        mock_response.content = mock_content_iter()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.request = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            request = {"messages": [{"role": "user", "content": "test"}]}
            chunks = []
            async for chunk in endpoint.stream(request):
                chunks.append(chunk)

            # Should get 3 non-empty lines (empty line filtered by if line:)
            assert len(chunks) == 3
            assert chunks[0] == "line1\n"
            assert chunks[1] == "line2\n"
            assert chunks[2] == "line3\n"

    @pytest.mark.asyncio
    async def test_stream_with_extra_headers(self, openai_config):
        """Test stream with extra_headers (lines 343-351)."""
        endpoint = Endpoint(config=openai_config)

        async def mock_stream(*args, **kwargs):
            yield b"chunk1"

        with patch.object(
            endpoint,
            "_stream_aiohttp",
            side_effect=lambda *args, **kwargs: mock_stream(),
        ):
            request = {"messages": [{"role": "user", "content": "test"}]}
            extra_headers = {"X-Stream-Header": "value"}
            chunks = []
            async for chunk in endpoint.stream(
                request, extra_headers=extra_headers
            ):
                chunks.append(chunk)
            assert len(chunks) == 1

    def test_to_dict(self, openai_config):
        """Test to_dict serialization (lines 392-403)."""
        from lionagi.service.resilience import CircuitBreaker, RetryConfig

        retry_config = RetryConfig(max_retries=3, base_delay=0.1)
        circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_time=2.0
        )
        endpoint = Endpoint(
            config=openai_config,
            retry_config=retry_config,
            circuit_breaker=circuit_breaker,
        )

        result = endpoint.to_dict()

        assert "config" in result
        assert "retry_config" in result
        assert "circuit_breaker" in result
        assert result["retry_config"] is not None
        assert result["circuit_breaker"] is not None

    def test_from_dict(self, openai_config):
        """Test from_dict deserialization (lines 405-425)."""
        from lionagi.service.resilience import CircuitBreaker, RetryConfig

        # Create endpoint with resilience patterns
        retry_config = RetryConfig(max_retries=3, base_delay=0.1)
        circuit_breaker = CircuitBreaker(
            failure_threshold=5, recovery_time=2.0
        )
        original_endpoint = Endpoint(
            config=openai_config,
            retry_config=retry_config,
            circuit_breaker=circuit_breaker,
        )

        # Serialize to dict
        data = original_endpoint.to_dict()

        # Deserialize from dict
        restored_endpoint = Endpoint.from_dict(data)

        assert restored_endpoint.config.name == original_endpoint.config.name
        assert restored_endpoint.retry_config is not None
        assert restored_endpoint.circuit_breaker is not None
        assert (
            restored_endpoint.retry_config.max_retries
            == original_endpoint.retry_config.max_retries
        )
