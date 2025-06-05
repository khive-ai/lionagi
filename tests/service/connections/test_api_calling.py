# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from lionagi.service.connections.api_calling import APICalling
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.protocols.generic.event import EventStatus


class TestAPICalling:
    """Test the APICalling class for error handling and response property."""

    @pytest.fixture
    def sample_payload(self):
        return {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7
        }

    @pytest.fixture
    def sample_headers(self):
        return {
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json"
        }

    @pytest.fixture
    def mock_endpoint(self):
        config = EndpointConfig(
            name="openai_chat",
            endpoint="chat",
            provider="openai",
            base_url="https://api.openai.com/v1",
            endpoint_params=["chat", "completions"],
            openai_compatible=True,
        )
        return Endpoint(config=config)

    def test_api_calling_initialization(self, sample_payload, sample_headers, mock_endpoint):
        """Test APICalling initialization."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        assert api_call.payload == sample_payload
        assert api_call.headers == sample_headers
        assert api_call.endpoint == mock_endpoint
        assert api_call.status == EventStatus.PENDING
        assert api_call.execution is not None
        assert api_call.execution.status == EventStatus.PENDING
        assert api_call.response is None

    def test_response_property_before_execution(self, sample_payload, sample_headers, mock_endpoint):
        """Test response property returns None before execution."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        assert api_call.response is None

    def test_response_property_after_execution(self, sample_payload, sample_headers, mock_endpoint):
        """Test response property returns execution response."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        # Mock execution with response
        mock_execution = MagicMock()
        mock_execution.response = {"test": "response"}
        api_call.execution = mock_execution
        
        assert api_call.response == {"test": "response"}

    @pytest.mark.asyncio
    async def test_successful_execution(self, sample_payload, sample_headers, mock_endpoint, mock_response):
        """Test successful API call execution."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        with patch.object(mock_endpoint, 'call', return_value=mock_response.json.return_value):
            await api_call.invoke()
        
        assert api_call.status == EventStatus.COMPLETED
        assert api_call.execution is not None
        assert api_call.response is not None

    @pytest.mark.asyncio
    async def test_execution_error_handling(self, sample_payload, sample_headers, mock_endpoint):
        """Test error handling during execution."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        # Mock endpoint to raise an exception
        with patch.object(mock_endpoint, 'call', side_effect=Exception("API Error")):
            await api_call.invoke()
        
        assert api_call.status == EventStatus.FAILED
        assert api_call.execution is not None
        assert "API Error" in str(api_call.execution.error)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, sample_payload, sample_headers, mock_endpoint):
        """Test timeout handling during execution."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        with patch.object(mock_endpoint, 'call', side_effect=asyncio.TimeoutError("Request timed out")):
            await api_call.invoke()
        
        assert api_call.status == EventStatus.FAILED

    @pytest.mark.asyncio
    async def test_streaming_execution(self, sample_payload, sample_headers, mock_endpoint):
        """Test streaming API call execution."""
        sample_payload["stream"] = True
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        async def mock_stream():
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}
            yield {"choices": [{"delta": {}}]}  # End of stream
        
        with patch.object(mock_endpoint, 'stream', return_value=mock_stream()):
            chunks = []
            async for chunk in api_call.stream():
                chunks.append(chunk)
        
        assert len(chunks) >= 2
        assert api_call.status == EventStatus.COMPLETED

    def test_cache_control_handling(self, sample_payload, sample_headers, mock_endpoint):
        """Test cache control parameter handling."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint,
            cache_control=True
        )
        
        assert api_call.cache_control is True

    @pytest.mark.asyncio
    async def test_concurrent_execution_isolation(self, sample_headers, mock_endpoint):
        """Test that concurrent API calls don't interfere with each other."""
        payloads = [
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": f"Message {i}"}]}
            for i in range(3)
        ]
        
        api_calls = [
            APICalling(payload=payload, headers=sample_headers, endpoint=mock_endpoint)
            for payload in payloads
        ]
        
        responses = [{"response": f"Response {i}"} for i in range(3)]
        
        async def mock_request(request, cache_control=False, **kwargs):
            # Simulate different response times
            i = int(request["messages"][0]["content"][-1])
            await asyncio.sleep(0.1 * (i + 1))
            return responses[i]
        
        with patch.object(mock_endpoint, 'call', side_effect=mock_request):
            tasks = [api_call.invoke() for api_call in api_calls]
            await asyncio.gather(*tasks)
        
        # Verify each call got the correct response
        for i, api_call in enumerate(api_calls):
            assert api_call.status == EventStatus.COMPLETED
            assert api_call.response == responses[i]

    @pytest.mark.asyncio
    async def test_error_propagation(self, sample_payload, sample_headers, mock_endpoint):
        """Test that errors are properly propagated and not swallowed."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        original_error = ValueError("Custom API error")
        
        with patch.object(mock_endpoint, 'call', side_effect=original_error):
            await api_call.invoke()
        
        assert api_call.status == EventStatus.FAILED
        assert api_call.execution.error is not None
        assert "Custom API error" in str(api_call.execution.error)

    def test_include_token_usage_to_model(self, sample_payload, sample_headers, mock_endpoint):
        """Test include_token_usage_to_model parameter."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint,
            include_token_usage_to_model=True
        )
        
        assert api_call.include_token_usage_to_model is True

    @pytest.mark.asyncio
    async def test_retry_logic(self, sample_payload, sample_headers, mock_endpoint):
        """Test that retry calls can be made with fresh API calling objects."""
        responses = []
        
        # First call fails
        api_call1 = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        with patch.object(mock_endpoint, 'call', side_effect=ConnectionError("Transient error")):
            await api_call1.invoke()
        
        assert api_call1.status == EventStatus.FAILED
        
        # Second call succeeds
        api_call2 = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        with patch.object(mock_endpoint, 'call', return_value={"success": True}):
            await api_call2.invoke()
        
        assert api_call2.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_payload_immutability(self, sample_payload, sample_headers, mock_endpoint):
        """Test that payload is not mutated during execution."""
        original_payload = sample_payload.copy()
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        with patch.object(mock_endpoint, 'call', return_value={"test": "response"}):
            await api_call.invoke()
        
        # Verify payload wasn't mutated
        assert api_call.payload == original_payload

    def test_str_representation(self, sample_payload, sample_headers, mock_endpoint):
        """Test string representation of APICalling."""
        api_call = APICalling(
            payload=sample_payload,
            headers=sample_headers,
            endpoint=mock_endpoint
        )
        
        str_repr = str(api_call)
        # APICalling inherits from Event, check that basic representation works
        assert "id=" in str_repr
        assert "payload=" in str_repr
        assert api_call.status == EventStatus.PENDING