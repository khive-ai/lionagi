# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import AsyncMock

import aiohttp
import pytest

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.imodel import iModel


@pytest.fixture
def mock_endpoint_config():
    """Create a basic endpoint configuration for testing."""
    return EndpointConfig(
        name="openai_chat",
        endpoint="chat",
        provider="openai",
        base_url="https://api.openai.com/v1",
        endpoint_params=["chat", "completions"],
        openai_compatible=True,
    )


@pytest.fixture
def mock_endpoint(mock_endpoint_config):
    """Create a mock endpoint with the configuration."""
    endpoint = Endpoint(config=mock_endpoint_config)
    endpoint._sdk_client = None
    return endpoint


@pytest.fixture
def mock_response():
    """Create a mock response for API calls."""
    response = AsyncMock(spec=aiohttp.ClientResponse)
    response.status = 200
    response.headers = {"content-type": "application/json"}
    response.json = AsyncMock(
        return_value={
            "id": "test-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4.1-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
    )
    response.text = AsyncMock(
        return_value=json.dumps(response.json.return_value)
    )
    return response


@pytest.fixture
def mock_anthropic_response():
    """Create a mock response for Anthropic API calls."""
    response = AsyncMock(spec=aiohttp.ClientResponse)
    response.status = 200
    response.headers = {"content-type": "application/json"}
    response.json = AsyncMock(
        return_value={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Test Anthropic response"}],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    )
    response.text = AsyncMock(
        return_value=json.dumps(response.json.return_value)
    )
    return response


@pytest.fixture
def mock_streaming_response():
    """Create a mock streaming response."""

    async def mock_iter_chunks():
        chunks = [
            b'data: {"id":"test","choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"id":"test","choices":[{"delta":{"content":" world"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        for chunk in chunks:
            yield chunk

    response = AsyncMock(spec=aiohttp.ClientResponse)
    response.status = 200
    response.headers = {"content-type": "text/event-stream"}
    response.content.iter_chunks = mock_iter_chunks
    return response


@pytest.fixture
def mock_imodel(mock_endpoint):
    """Create a mock iModel instance."""
    imodel = iModel(
        provider="openai",
        endpoint=mock_endpoint,
        model="gpt-4.1-mini",
        api_key="test-key",
    )
    return imodel


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]


@pytest.fixture
def sample_payload():
    """Sample payload for API requests."""
    return {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.7,
        "max_tokens": 100,
    }
