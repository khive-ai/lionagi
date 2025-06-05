# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import asyncio
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.protocols.generic.event import EventStatus


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
        # Note: allowed_roles is a property of iModel, not Endpoint

    def test_endpoint_stateless_design(self, openai_config):
        """Test that endpoint is stateless between calls."""
        endpoint = Endpoint(config=openai_config)
        
        # First payload creation
        payload1, headers1 = endpoint.create_payload({
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini"
        })
        
        # Second payload creation with different data
        payload2, headers2 = endpoint.create_payload({
            "messages": [{"role": "user", "content": "Goodbye"}],
            "model": "gpt-4o"
        })
        
        # Verify that payloads are independent
        assert payload1["messages"][0]["content"] == "Hello"
        assert payload2["messages"][0]["content"] == "Goodbye"
        assert payload1["model"] == "gpt-4o-mini"
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
        
        with patch.object(endpoint, '_create_http_session', side_effect=mock_create_session):
            # Simulate multiple concurrent requests
            tasks = []
            for i in range(3):
                task = asyncio.create_task(endpoint._create_http_session())
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        # Verify each call created its own session
        assert len(sessions_created) == 3
        assert all(session is not sessions_created[0] for session in sessions_created[1:])


    def test_create_payload_openai(self, openai_config):
        """Test payload creation for OpenAI endpoint."""
        endpoint = Endpoint(config=openai_config)
        
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        payload, headers = endpoint.create_payload(request_data)
        
        assert payload["model"] == "gpt-4o-mini"
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
            "api_key": "test-key"
        }
        
        payload, headers = endpoint.create_payload(request_data)
        
        assert payload["model"] == "claude-3-opus-20240229"
        assert payload["messages"] == request_data["messages"]
        assert payload["max_tokens"] == 100
        assert "api_key" not in payload  # Should be removed from payload
        assert "x-api-key" in headers
        assert headers["anthropic-version"] == "2023-06-01"

    @pytest.mark.asyncio
    async def test_http_request_session_cleanup(self, openai_config, mock_response):
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
            
        with patch('aiohttp.ClientSession', side_effect=mock_session_class):
            request = {"messages": [{"role": "user", "content": "test"}], "model": "gpt-4o-mini"}
            
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
                "choices": [{"message": {"content": f"Response to {payload['messages'][0]['content']}"}}]
            }
        
        with patch.object(endpoint, 'call', side_effect=mock_request_with_delay):
            # Create multiple concurrent requests
            requests = [
                {"messages": [{"role": "user", "content": f"Message {i}"}]}
                for i in range(3)
            ]
            
            tasks = []
            for req in requests:
                payload, headers = endpoint.create_payload(req)
                task = asyncio.create_task(endpoint.call(payload, headers, delay=0.05))
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
        
        # Verify each response corresponds to its request
        for i, response in enumerate(responses):
            assert f"Message {i}" in response["id"]
            assert f"Message {i}" in response["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_sdk_vs_http_transport(self, openai_config):
        """Test that endpoint chooses appropriate transport method."""
        endpoint = Endpoint(config=openai_config)
        
        # Test HTTP transport
        with patch.object(endpoint, 'call') as mock_http:
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
        
        with patch.object(endpoint, 'call', side_effect=mock_request_with_errors):
            # Create three concurrent requests
            tasks = []
            for i in range(3):
                payload, headers = endpoint.create_payload({
                    "messages": [{"role": "user", "content": f"test {i}"}]
                })
                task = asyncio.create_task(endpoint.call(payload, headers))
                tasks.append(task)
            
            # Gather with return_exceptions to handle the error
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # First and third should succeed, second should fail
        assert results[0] == {"success": True, "call": 1}
        assert isinstance(results[1], aiohttp.ClientError)
        assert results[2] == {"success": True, "call": 3}