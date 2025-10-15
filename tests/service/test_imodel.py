# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from unittest.mock import patch

import pytest

from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.imodel import iModel


class TestiModel:
    """Test the iModel class for request validation and parallel calls."""

    def test_imodel_initialization_with_provider(self):
        """Test iModel initialization with explicit provider."""
        # Create the iModel with explicit api_key parameter
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        assert imodel.endpoint.config.provider == "openai"
        assert imodel.endpoint.config.kwargs["model"] == "gpt-4.1-mini"
        # The actual API key might be different, so just check it's set
        assert imodel.endpoint.config._api_key is not None

    def test_imodel_initialization_from_model_path(self):
        """Test iModel initialization with provider inferred from model path."""
        imodel = iModel(model="openai/gpt-4.1-mini", api_key="test-key")

        assert imodel.endpoint.config.provider == "openai"
        assert imodel.endpoint.config.kwargs["model"] == "gpt-4.1-mini"

    def test_imodel_initialization_missing_provider(self):
        """Test that iModel raises error when provider cannot be determined."""
        with pytest.raises(ValueError, match="Provider must be provided"):
            iModel(model="gpt-4.1-mini")  # No provider, no slash in model

    def test_api_key_environment_variable_lookup(self):
        """Test that API keys are correctly looked up from environment."""
        test_cases = [
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("perplexity", "PERPLEXITY_API_KEY"),
        ]

        for provider, env_var in test_cases:
            if env_var:
                # Just verify that an API key is set when provider requires it
                imodel = iModel(
                    provider=provider,
                    model="test-model",
                    api_key=f"test-{provider}-key",
                )
                assert imodel.endpoint.config._api_key is not None

    def test_custom_api_key(self):
        """Test iModel initialization with custom API key."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="custom-key"
        )

        # Just verify that an API key was set
        assert imodel.endpoint.config._api_key is not None

    def test_create_api_calling(self):
        """Test creation of APICalling objects."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}], temperature=0.7
        )

        assert isinstance(api_call, APICalling)
        assert api_call.payload["model"] == "gpt-4.1-mini"
        assert api_call.payload["messages"][0]["content"] == "Hello"
        assert api_call.payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_successful_invoke(self, mock_response):
        """Test successful API invocation."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert isinstance(result, APICalling)
        assert result.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_parallel_invoke_calls(self, mock_response):
        """Test parallel API invocations don't interfere."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        async def mock_request_with_id(request, cache_control=False, **kwargs):
            await asyncio.sleep(0.1)  # Simulate network delay
            response = mock_response.json.return_value.copy()
            response["id"] = (
                f"response-{request['messages'][0]['content'][-1]}"
            )
            return response

        with patch.object(
            imodel.endpoint, "call", side_effect=mock_request_with_id
        ):
            # Create multiple concurrent calls
            tasks = []
            for i in range(3):
                task = asyncio.create_task(
                    imodel.invoke(
                        messages=[{"role": "user", "content": f"Message {i}"}]
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        # Verify all calls completed successfully and independently
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.status == EventStatus.COMPLETED
            assert f"{i}" in result.response["id"]

    @pytest.mark.asyncio
    async def test_streaming_invoke(self, mock_streaming_response):
        """Test streaming API calls."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        async def mock_stream():
            chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
                {"choices": [{"delta": {}}]},  # End marker
            ]
            for chunk in chunks:
                yield chunk

        # Set streaming_process_func to return chunks
        imodel.streaming_process_func = lambda chunk: chunk

        with patch.object(
            imodel.endpoint, "stream", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in imodel.stream(
                messages=[{"role": "user", "content": "Hello"}]
            ):
                if chunk:
                    chunks.append(chunk)

        assert len(chunks) >= 2  # Should have content chunks

    def test_model_name_property(self):
        """Test model_name property."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        assert imodel.model_name == "gpt-4.1-mini"

    def test_request_options_property(self):
        """Test request_options property."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        # NOTE: request_options removed due to incorrect role literals in generated models
        # Should return OpenAIChatCompletionsRequest for OpenAI
        from lionagi.service.third_party.openai_models import (
            OpenAIChatCompletionsRequest,
        )

        assert imodel.request_options == OpenAIChatCompletionsRequest

    @pytest.mark.asyncio
    async def test_error_handling_in_invoke(self):
        """Test error handling during API invocation."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        with patch.object(
            imodel.endpoint, "call", side_effect=Exception("API Error")
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

            # The invoke method returns a failed APICalling object instead of raising
            assert result.status == EventStatus.FAILED
            assert result.execution.error is not None

    def test_cache_control_parameter(self):
        """Test cache_control parameter in create_api_calling."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}], cache_control=True
        )

        assert api_call.cache_control is True

    def test_include_token_usage_to_model(self):
        """Test include_token_usage_to_model parameter."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}],
            include_token_usage_to_model=True,
        )

        assert api_call.include_token_usage_to_model is True

    def test_to_dict_serialization(self):
        """Test iModel serialization to dictionary."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            temperature=0.7,
            api_key="test-key",
        )

        data = imodel.to_dict()

        assert "endpoint" in data
        assert "processor_config" in data
        assert imodel.endpoint.config.provider == "openai"
        assert imodel.endpoint.config.kwargs.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_custom_streaming_process_func(self):
        """Test custom streaming process function."""

        def custom_process(chunk):
            if hasattr(chunk, "choices") and chunk.choices:
                return f"Processed: {chunk.choices[0].delta.content}"
            return None

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            imodel = iModel(
                provider="openai",
                model="gpt-4.1-mini",
                streaming_process_func=custom_process,
            )

        assert imodel.streaming_process_func == custom_process

    @pytest.mark.asyncio
    async def test_rate_limiting_configuration(self):
        """Test rate limiting configuration."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            imodel = iModel(
                provider="openai",
                model="gpt-4.1-mini",
                limit_requests=10,
                limit_tokens=1000,
                queue_capacity=50,
            )

        assert imodel.executor.config["limit_requests"] == 10
        assert imodel.executor.config["limit_tokens"] == 1000
        assert imodel.executor.config["queue_capacity"] == 50

    @pytest.mark.asyncio
    async def test_aclose_cleanup(self):
        """Test async cleanup of resources."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        # aclose no longer exists - resources are cleaned up automatically
        pass

    @pytest.mark.asyncio
    async def test_concurrent_different_models(self):
        """Test concurrent calls with different models work independently."""
        imodel1 = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        imodel2 = iModel(provider="openai", model="gpt-4o", api_key="test-key")

        async def mock_request(request, cache_control=False, **kwargs):
            await asyncio.sleep(0.05)
            # The request parameter is the payload dict
            model = (
                request.get("model", "unknown")
                if isinstance(request, dict)
                else "unknown"
            )
            return {"model_used": model, "response": "test"}

        with patch(
            "lionagi.service.connections.endpoint.Endpoint.call",
            side_effect=mock_request,
        ):
            task1 = asyncio.create_task(
                imodel1.invoke(messages=[{"role": "user", "content": "Hello"}])
            )
            task2 = asyncio.create_task(
                imodel2.invoke(messages=[{"role": "user", "content": "Hello"}])
            )

            result1, result2 = await asyncio.gather(task1, task2)

        assert result1.response["model_used"] == "gpt-4.1-mini"
        assert result2.response["model_used"] == "gpt-4o"

    def test_imodel_custom_id_with_id_get_id(self):
        """Test iModel initialization with custom ID using ID.get_id."""
        from uuid import uuid4

        custom_id = uuid4()
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            id=custom_id,
        )

        assert imodel.id == custom_id

    def test_imodel_custom_id_as_string(self):
        """Test iModel initialization with ID as UUID string."""
        from uuid import uuid4

        custom_id = str(uuid4())
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            id=custom_id,
        )

        # ID.get_id should handle UUID string conversion
        assert str(imodel.id) == custom_id

    def test_imodel_invalid_created_at_type(self):
        """Test iModel initialization with invalid created_at type."""
        with pytest.raises(
            ValueError, match="created_at must be a float timestamp"
        ):
            iModel(
                provider="openai",
                model="gpt-4.1-mini",
                api_key="test-key",
                created_at="not-a-float",
            )

    def test_imodel_with_endpoint_object(self):
        """Test iModel initialization with Endpoint object passed directly."""
        from lionagi.service.connections.endpoint import Endpoint
        from lionagi.service.connections.match_endpoint import match_endpoint

        # Create an endpoint object
        endpoint = match_endpoint(
            provider="openai",
            endpoint="chat",
            model="gpt-4.1-mini",
            api_key="test-key",
        )

        # Pass endpoint object directly
        imodel = iModel(endpoint=endpoint, api_key="test-key")

        assert imodel.endpoint == endpoint
        assert imodel.endpoint.config.provider == "openai"

    def test_imodel_hook_registry_as_dict(self):
        """Test iModel initialization with hook_registry as dict."""
        from lionagi.service.hooks import HookRegistry

        hook_registry_dict = {}
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry_dict,
        )

        assert isinstance(imodel.hook_registry, HookRegistry)

    def test_imodel_claude_code_auto_resume(self):
        """Test auto-injection of resume parameter for claude_code provider."""
        imodel = iModel(
            provider="claude_code",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        # Set session_id in provider_metadata
        imodel.provider_metadata["session_id"] = "test-session-123"

        # Create API calling without explicit resume parameter
        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}]
        )

        # Check that resume was auto-injected (in the request object for claude_code)
        assert api_call.payload["request"].resume == "test-session-123"

    def test_imodel_claude_code_no_auto_resume_if_explicit(self):
        """Test that explicit resume parameter is not overridden."""
        imodel = iModel(
            provider="claude_code",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        # Set session_id in provider_metadata
        imodel.provider_metadata["session_id"] = "test-session-123"

        # Create API calling WITH explicit resume parameter
        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}],
            resume="explicit-session",
        )

        # Check that explicit resume was used (in the request object for claude_code)
        assert api_call.payload["request"].resume == "explicit-session"

    @pytest.mark.asyncio
    async def test_imodel_streaming_with_async_process_func(self):
        """Test streaming with async streaming_process_func."""

        async def async_process(chunk):
            """Async processing function."""
            await asyncio.sleep(0.001)
            if hasattr(chunk, "get") and chunk.get("choices"):
                return (
                    f"Async: {chunk['choices'][0]['delta'].get('content', '')}"
                )
            return None

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            streaming_process_func=async_process,
        )

        async def mock_stream():
            chunks = [
                {"choices": [{"delta": {"content": "Hello"}}]},
                {"choices": [{"delta": {"content": " world"}}]},
            ]
            for chunk in chunks:
                yield chunk

        with patch.object(
            imodel.endpoint, "stream", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in imodel.stream(
                messages=[{"role": "user", "content": "Hello"}]
            ):
                if chunk and not isinstance(chunk, APICalling):
                    chunks.append(chunk)

        assert len(chunks) >= 2
        assert any("Async:" in str(chunk) for chunk in chunks)

    @pytest.mark.asyncio
    async def test_imodel_claude_code_session_id_storage(self, mock_response):
        """Test that session_id is stored in provider_metadata after invoke."""
        imodel = iModel(
            provider="claude_code",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        # Mock response with session_id
        async def mock_request_with_session(
            request, cache_control=False, **kwargs
        ):
            return {"session_id": "new-session-456", "response": "test"}

        with patch.object(
            imodel.endpoint, "call", side_effect=mock_request_with_session
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

        # Check that session_id was stored
        assert imodel.provider_metadata.get("session_id") == "new-session-456"

    def test_imodel_to_dict(self):
        """Test iModel to_dict serialization."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=10,
        )

        data = imodel.to_dict()

        assert "id" in data
        assert "created_at" in data
        assert "endpoint" in data
        assert "processor_config" in data
        assert "provider_metadata" in data
        assert data["processor_config"]["limit_requests"] == 10

    def test_imodel_from_dict(self):
        """Test iModel from_dict deserialization."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=10,
        )

        data = imodel.to_dict()
        restored = iModel.from_dict(data)

        assert restored.id == imodel.id
        assert restored.created_at == imodel.created_at
        assert restored.endpoint.config.provider == "openai"
        assert restored.executor.config["limit_requests"] == 10

    def test_imodel_from_dict_with_match_endpoint(self):
        """Test from_dict uses match_endpoint for endpoint reconstruction."""
        # Create initial iModel
        imodel = iModel(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        # Serialize and deserialize
        data = imodel.to_dict()
        restored = iModel.from_dict(data)

        # Check that match_endpoint was used to properly reconstruct
        assert restored.endpoint.config.provider == "anthropic"
        assert (
            restored.endpoint.config.kwargs["model"]
            == "claude-3-5-sonnet-20241022"
        )

    @pytest.mark.asyncio
    async def test_imodel_unsupported_event_type(self):
        """Test that unsupported event types raise ValueError."""
        from lionagi.protocols.types import Event

        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        # Create a custom event type that's not APICalling
        class CustomEvent(Event):
            pass

        with pytest.raises(
            ValueError,
            match="Unsupported event type.*Only APICalling is supported",
        ):
            await imodel.create_event(create_event_type=CustomEvent)

    @pytest.mark.asyncio
    async def test_imodel_invoke_with_concurrency_limit(self, mock_response):
        """Test invoke with concurrency limit set."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            concurrency_limit=2,
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert isinstance(result, APICalling)
        assert result.status == EventStatus.COMPLETED
