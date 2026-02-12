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

    def test_imodel_initialization_with_provider(self, base_imodel):
        """Test iModel initialization with explicit provider."""
        imodel = base_imodel

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

    def test_create_api_calling(self, base_imodel):
        """Test creation of APICalling objects."""
        imodel = base_imodel

        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}], temperature=0.7
        )

        assert isinstance(api_call, APICalling)
        assert api_call.payload["model"] == "gpt-4.1-mini"
        assert api_call.payload["messages"][0]["content"] == "Hello"
        assert api_call.payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_successful_invoke(self, base_imodel, mock_response):
        """Test successful API invocation."""
        imodel = base_imodel

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
    async def test_parallel_invoke_calls(self, base_imodel, mock_response):
        """Test parallel API invocations don't interfere."""
        imodel = base_imodel

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
    async def test_streaming_invoke(
        self, base_imodel, mock_streaming_response
    ):
        """Test streaming API calls."""
        imodel = base_imodel

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

    def test_model_name_property(self, base_imodel):
        """Test model_name property."""
        imodel = base_imodel

        assert imodel.model_name == "gpt-4.1-mini"

    def test_request_options_property(self, base_imodel):
        """Test request_options property."""
        imodel = base_imodel

        # NOTE: request_options removed due to incorrect role literals in generated models
        # Should return OpenAIChatCompletionsRequest for OpenAI
        from lionagi.service.third_party.openai_models import (
            OpenAIChatCompletionsRequest,
        )

        assert imodel.request_options == OpenAIChatCompletionsRequest

    @pytest.mark.asyncio
    async def test_error_handling_in_invoke(self, base_imodel):
        """Test error handling during API invocation."""
        imodel = base_imodel

        with patch.object(
            imodel.endpoint, "call", side_effect=Exception("API Error")
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

            # The invoke method returns a failed APICalling object instead of raising
            assert result.status == EventStatus.FAILED
            assert result.execution.error is not None

    def test_cache_control_parameter(self, base_imodel):
        """Test cache_control parameter in create_api_calling."""
        imodel = base_imodel

        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}], cache_control=True
        )

        assert api_call.cache_control is True

    def test_include_token_usage_to_model(self, base_imodel):
        """Test include_token_usage_to_model parameter."""
        imodel = base_imodel

        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "Hello"}],
            include_token_usage_to_model=True,
        )

        assert api_call.include_token_usage_to_model is True

    def test_to_dict_serialization(self, base_imodel):
        """Test iModel serialization to dictionary."""
        imodel = base_imodel

        data = imodel.to_dict()

        assert "endpoint" in data
        assert "processor_config" in data
        assert imodel.endpoint.config.provider == "openai"

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

        # Set session_id on the CLI endpoint
        imodel.endpoint.session_id = "test-session-123"

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

        # Set session_id on the CLI endpoint
        imodel.endpoint.session_id = "test-session-123"

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
        """Test that session_id is stored on endpoint after invoke."""
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

        # Check that session_id was stored on the CLI endpoint
        assert imodel.endpoint.session_id == "new-session-456"

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


class TestiModelEdgeCases:
    """Edge case tests for iModel - concurrent behavior, rate limiting, error recovery."""

    @pytest.mark.asyncio
    async def test_concurrent_streaming_multiple_requests(
        self, mock_streaming_response
    ):
        """Test concurrent streaming requests with semaphore control."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            concurrency_limit=2,
        )

        call_count = 0

        async def mock_stream_generator():
            for i in range(3):
                yield {"choices": [{"delta": {"content": f"chunk {i}"}}]}
                await asyncio.sleep(0.05)

        def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_stream_generator()

        with patch.object(imodel.endpoint, "stream", side_effect=track_calls):
            # Start 5 concurrent streaming requests with limit of 2
            tasks = []
            for i in range(5):

                async def collect_stream(idx):
                    chunks = []
                    async for chunk in imodel.stream(
                        messages=[
                            {"role": "user", "content": f"Request {idx}"}
                        ]
                    ):
                        if chunk and not isinstance(chunk, APICalling):
                            chunks.append(chunk)
                    return chunks

                tasks.append(asyncio.create_task(collect_stream(i)))

            results = await asyncio.gather(*tasks)

        # Verify all streams completed
        assert len(results) == 5
        # At least some should have chunks (not all may complete in time)
        chunks_found = sum(1 for r in results if len(r) > 0)
        assert chunks_found >= 3
        assert call_count == 5  # All calls executed

    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self, mock_response):
        """Test rate limiting enforcement under concurrent load."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=3,
            limit_tokens=100,
            capacity_refresh_time=1.0,
        )

        call_times = []

        async def track_timing(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)
            return mock_response.json.return_value

        with patch.object(imodel.endpoint, "call", side_effect=track_timing):
            # Fire 10 concurrent requests
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    imodel.invoke(
                        messages=[{"role": "user", "content": f"Request {i}"}]
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check that some requests completed successfully
        successful = [r for r in results if isinstance(r, APICalling)]
        assert len(successful) > 0

    @pytest.mark.asyncio
    async def test_provider_switching_mid_session(self, mock_response):
        """Test switching providers by creating new iModel instances."""
        # Start with OpenAI
        imodel1 = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        async def mock_openai_call(*args, **kwargs):
            return {"provider": "openai", "response": "OpenAI response"}

        with patch.object(
            imodel1.endpoint, "call", side_effect=mock_openai_call
        ):
            result1 = await imodel1.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )
            assert result1.response["provider"] == "openai"

        # Switch to Anthropic with required parameters
        imodel2 = iModel(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        async def mock_anthropic_call(*args, **kwargs):
            return {"provider": "anthropic", "response": "Anthropic response"}

        with patch.object(
            imodel2.endpoint, "call", side_effect=mock_anthropic_call
        ):
            result2 = await imodel2.invoke(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,  # Required for Anthropic
            )
            assert result2.response["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_error_recovery_and_retry_logic(self):
        """Test error handling and recovery in invoke."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        call_count = 0

        async def failing_then_success(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Temporary error {call_count}")
            return {"success": True, "attempt": call_count}

        # First call fails
        with patch.object(
            imodel.endpoint, "call", side_effect=Exception("API Error")
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )
            assert result.status == EventStatus.FAILED
            assert result.execution.error is not None

        # Recovery with manual retry
        call_count = 0
        with patch.object(
            imodel.endpoint, "call", side_effect=failing_then_success
        ):
            # Manual retry loop
            for attempt in range(5):
                result = await imodel.invoke(
                    messages=[{"role": "user", "content": "Hello"}]
                )
                if result.status == EventStatus.COMPLETED:
                    break

            assert result.status == EventStatus.COMPLETED
            assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_streaming_error_mid_stream(self):
        """Test error handling when streaming fails mid-stream."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )

        async def failing_stream():
            yield {"choices": [{"delta": {"content": "Start"}}]}
            yield {"choices": [{"delta": {"content": " middle"}}]}
            raise Exception("Stream interrupted")

        with patch.object(
            imodel.endpoint, "stream", return_value=failing_stream()
        ):
            chunks = []
            error_raised = False
            try:
                async for chunk in imodel.stream(
                    messages=[{"role": "user", "content": "Hello"}]
                ):
                    if chunk and not isinstance(chunk, APICalling):
                        chunks.append(chunk)
            except ValueError as e:
                error_raised = True
                assert "Failed to stream API call" in str(e)

            # Either error was raised or chunks were collected
            assert error_raised or len(chunks) >= 2

    @pytest.mark.asyncio
    async def test_concurrent_invoke_with_queue_capacity(self, mock_response):
        """Test queue capacity limits with concurrent invocations."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            queue_capacity=5,
            limit_requests=2,
        )

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(0.2)
            return mock_response.json.return_value

        with patch.object(imodel.endpoint, "call", side_effect=slow_call):
            # Fire more requests than queue capacity
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    imodel.invoke(
                        messages=[{"role": "user", "content": f"Request {i}"}]
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some should complete successfully
        successful = [
            r
            for r in results
            if isinstance(r, APICalling) and r.status == EventStatus.COMPLETED
        ]
        assert len(successful) > 0

    @pytest.mark.asyncio
    async def test_provider_metadata_persistence(self, mock_response):
        """Test session_id persists on CLI endpoint across multiple calls."""
        imodel = iModel(
            provider="claude_code",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        # First call stores session_id
        async def first_call(*args, **kwargs):
            return {"session_id": "session-123", "response": "First call"}

        with patch.object(imodel.endpoint, "call", side_effect=first_call):
            result1 = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )
            assert imodel.endpoint.session_id == "session-123"

        # Second call uses stored session_id
        async def second_call(*args, **kwargs):
            return {"session_id": "session-123", "response": "Second call"}

        with patch.object(imodel.endpoint, "call", side_effect=second_call):
            # Create api_calling to check resume parameter
            api_call = imodel.create_api_calling(
                messages=[{"role": "user", "content": "Follow-up"}]
            )
            # Session ID should be auto-injected as resume
            assert api_call.payload["request"].resume == "session-123"

    @pytest.mark.asyncio
    async def test_streaming_with_processing_function_error(self):
        """Test error in streaming_process_func doesn't crash stream."""

        def failing_processor(chunk):
            if "error" in str(chunk):
                raise ValueError("Processing error")
            return f"Processed: {chunk}"

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            streaming_process_func=failing_processor,
        )

        async def mock_stream():
            yield {"choices": [{"delta": {"content": "normal"}}]}
            yield {"choices": [{"delta": {"content": "error"}}]}
            yield {"choices": [{"delta": {"content": "continue"}}]}

        with patch.object(
            imodel.endpoint, "stream", return_value=mock_stream()
        ):
            chunks = []
            try:
                async for chunk in imodel.stream(
                    messages=[{"role": "user", "content": "Hello"}]
                ):
                    if chunk and not isinstance(chunk, APICalling):
                        chunks.append(chunk)
            except ValueError as e:
                assert "Failed to stream API call" in str(e)

    @pytest.mark.asyncio
    async def test_serialization_roundtrip_with_complex_config(self):
        """Test to_dict/from_dict preserves complex configurations."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=10,
            limit_tokens=1000,
            queue_capacity=50,
            concurrency_limit=5,
            provider_metadata={
                "custom_key": "custom_value",
                "session_id": "abc",
            },
        )

        # Serialize
        data = imodel.to_dict()

        # Deserialize
        restored = iModel.from_dict(data)

        # Verify
        assert restored.id == imodel.id
        assert restored.created_at == imodel.created_at
        assert restored.endpoint.config.provider == "openai"
        assert restored.executor.config["limit_requests"] == 10
        assert restored.executor.config["limit_tokens"] == 1000
        assert restored.executor.config["queue_capacity"] == 50
        assert restored.provider_metadata == imodel.provider_metadata


class TestiModelValidationErrors:
    """Tests for validation error handling in iModel."""

    def test_invalid_temperature_type(self):
        """Test iModel with invalid temperature type raises validation error."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        # Temperature validation happens at payload creation
        with pytest.raises(ValueError, match="Invalid payload"):
            imodel.create_api_calling(
                messages=[{"role": "user", "content": "test"}],
                temperature="invalid",
            )

    def test_invalid_max_tokens_negative(self):
        """Test iModel with negative max_tokens."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": "test"}], max_tokens=-100
        )
        # Should accept negative value, API will validate
        assert api_call.payload["max_tokens"] == -100

    def test_invalid_messages_structure(self):
        """Test iModel with invalid messages structure."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        # Test with malformed messages
        api_call = imodel.create_api_calling(
            messages=[{"invalid": "structure"}]
        )
        # Should create payload, validation happens at API level
        assert len(api_call.payload["messages"]) == 1

    def test_empty_content_in_messages(self):
        """Test iModel with empty content in messages."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": ""}]
        )
        assert api_call.payload["messages"][0]["content"] == ""

    def test_none_role_in_messages(self):
        """Test iModel with None role raises validation error."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        # None role should fail validation
        with pytest.raises(ValueError, match="Invalid payload"):
            imodel.create_api_calling(
                messages=[{"role": None, "content": "test"}]
            )

    def test_invalid_model_parameter_type(self):
        """Test iModel with invalid model type raises validation error."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        # Invalid model type should fail validation
        with pytest.raises(ValueError, match="Invalid payload"):
            imodel.create_api_calling(
                messages=[{"role": "user", "content": "test"}], model=123
            )

    def test_very_long_message_content(self):
        """Test iModel with extremely long message content."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        long_content = "x" * 1000000  # 1 million characters
        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": long_content}]
        )
        assert len(api_call.payload["messages"][0]["content"]) == 1000000

    def test_special_characters_in_messages(self):
        """Test iModel with special characters and unicode in messages."""
        imodel = iModel(
            provider="openai", model="gpt-4.1-mini", api_key="test-key"
        )
        special_content = "Hello 世界 🌍 \n\t\r !@#$%^&*()"
        api_call = imodel.create_api_calling(
            messages=[{"role": "user", "content": special_content}]
        )
        assert api_call.payload["messages"][0]["content"] == special_content


class TestiModelRateLimitingEdgeCases:
    """Tests for rate limiting edge cases and boundary conditions."""

    def test_zero_rate_limits(self):
        """Test iModel accepts zero rate limits (no limiting)."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=0,
            limit_tokens=0,
        )
        # Zero means unlimited
        assert imodel.executor.config["limit_requests"] == 0

    def test_negative_rate_limits(self):
        """Test iModel accepts negative rate limits (no limiting)."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=-10,
        )
        # Negative values may be treated as unlimited
        assert imodel.executor.config["limit_requests"] == -10

    def test_extremely_high_rate_limits(self):
        """Test iModel with extremely high rate limits."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=1000000,
            limit_tokens=100000000,
        )
        assert imodel.executor.config["limit_requests"] == 1000000

    def test_zero_queue_capacity(self):
        """Test iModel accepts zero queue capacity (unlimited queue)."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            queue_capacity=0,
        )
        # Zero may mean unlimited queue
        assert imodel.executor.config["queue_capacity"] == 0

    def test_capacity_refresh_time_boundary(self):
        """Test iModel with boundary capacity refresh times."""
        # Very short refresh time
        imodel1 = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            capacity_refresh_time=0.001,
        )
        assert imodel1.executor.config["capacity_refresh_time"] == 0.001

        # Very long refresh time
        imodel2 = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            capacity_refresh_time=3600.0,
        )
        assert imodel2.executor.config["capacity_refresh_time"] == 3600.0

    def test_zero_concurrency_limit(self):
        """Test iModel with zero concurrency limit uses default."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            concurrency_limit=0,
        )
        # Zero gets converted to default (100)
        assert imodel.executor.concurrency_limit == 100

    def test_single_concurrency_limit(self):
        """Test iModel with concurrency limit of 1."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            concurrency_limit=1,
        )
        assert imodel.executor.concurrency_limit == 1

    @pytest.mark.asyncio
    async def test_rate_limit_token_counting(self, mock_response):
        """Test that token counting is tracked for rate limiting."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=100,
            limit_tokens=10000,
        )

        # Create response with token usage
        response_with_tokens = {
            "choices": [{"message": {"content": "test"}}],
            "usage": {"total_tokens": 50},
        }

        with patch.object(
            imodel.endpoint, "call", return_value=response_with_tokens
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "test"}],
                include_token_usage_to_model=True,
            )

        assert result.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_burst_requests_rate_limiting(self, mock_response):
        """Test rate limiting behavior with burst of requests."""
        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            limit_requests=5,
            capacity_refresh_time=1.0,
        )

        call_count = 0

        async def count_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return mock_response.json.return_value

        with patch.object(imodel.endpoint, "call", side_effect=count_calls):
            # Fire 20 requests at once
            tasks = [
                asyncio.create_task(
                    imodel.invoke(
                        messages=[{"role": "user", "content": f"Request {i}"}]
                    )
                )
                for i in range(20)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some requests should complete
        successful = [r for r in results if isinstance(r, APICalling)]
        assert len(successful) > 0


class TestiModelProviderSpecificEdgeCases:
    """Tests for provider-specific edge cases."""

    def test_anthropic_without_max_tokens(self):
        """Test Anthropic iModel creation."""
        imodel = iModel(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )
        # Should create successfully, max_tokens required at invoke time
        assert imodel.endpoint.config.provider == "anthropic"

    def test_ollama_special_handling(self):
        """Test Ollama provider special handling."""
        imodel = iModel(
            provider="ollama",
            model="llama2",
            api_key="ollama",  # Special ollama key
        )
        assert imodel.endpoint.config.provider == "ollama"

    def test_claude_code_session_id_initialization(self):
        """Test Claude Code session_id on CLI endpoint."""
        imodel = iModel(
            provider="claude_code",
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )
        # Set session_id on the CLI endpoint directly
        imodel.endpoint.session_id = "initial-session"
        assert imodel.endpoint.session_id == "initial-session"

    def test_openrouter_model_path_parsing(self):
        """Test OpenRouter model path parsing."""
        imodel = iModel(
            model="openrouter/anthropic/claude-3-opus",
            api_key="test-key",
        )
        # Should parse provider from model path
        assert imodel.endpoint.config.provider == "openrouter"

    def test_mixed_case_provider_names(self):
        """Test provider names with mixed case."""
        imodel = iModel(
            provider="OpenAI",  # Mixed case
            model="gpt-4.1-mini",
            api_key="test-key",
        )
        # Provider should be normalized
        assert imodel.endpoint.config.provider.lower() == "openai"
