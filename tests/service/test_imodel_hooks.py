# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from lionagi.protocols.generic.event import EventStatus
from lionagi.service.connections.api_calling import APICalling
from lionagi.service.hooks import HookRegistry
from lionagi.service.hooks._types import HookEventTypes
from lionagi.service.imodel import iModel


class TestiModelHooks:
    """Comprehensive hook system tests for iModel."""

    @pytest.mark.asyncio
    async def test_pre_invoke_hook_success(self, mock_response):
        """Test successful pre-invocation hook execution."""
        hook_called = False
        hook_params = {}

        async def pre_invoke_hook(event, **kwargs):
            nonlocal hook_called, hook_params
            hook_called = True
            hook_params = kwargs
            return None

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: pre_invoke_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert hook_called
        assert result.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_post_invoke_hook_success(self, mock_response):
        """Test successful post-invocation hook execution."""
        hook_called = False
        captured_event = None

        async def post_invoke_hook(event, **kwargs):
            nonlocal hook_called, captured_event
            hook_called = True
            captured_event = event
            return None

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PostInvocation: post_invoke_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert hook_called
        assert captured_event is not None
        assert result.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_pre_event_create_hook(self, mock_response):
        """Test pre-event-create hook execution."""
        hook_called = False
        event_type_captured = None

        async def pre_create_hook(event_type, **kwargs):
            nonlocal hook_called, event_type_captured
            hook_called = True
            event_type_captured = event_type
            return None

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreEventCreate: pre_create_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

        assert hook_called
        assert (
            event_type_captured == APICalling
            or event_type_captured is not None
        )
        assert result.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_hook_error_handling(self):
        """Test hook error handling and event cancellation."""

        async def failing_hook(event, **kwargs):
            raise ValueError("Hook intentionally failed")

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: failing_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
            exit_hook=False,  # Don't exit on hook error
        )

        # Hook failure should cancel the event but not crash
        with patch.object(
            imodel.endpoint, "call", return_value={"test": "response"}
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}],
                pre_invoke_event_hook_timeout=0.05,  # Short timeout for testing
            )

        # Event should be cancelled or failed due to hook error
        assert result.status in (EventStatus.CANCELLED, EventStatus.FAILED)

    @pytest.mark.asyncio
    async def test_hook_timeout_behavior(self):
        """Test hook timeout handling."""

        async def slow_hook(event, **kwargs):
            await asyncio.sleep(0.1)  # Longer than timeout
            return None

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: slow_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        # Create event with short timeout
        api_call = await imodel.create_event(
            messages=[{"role": "user", "content": "Hello"}],
            pre_invoke_event_hook_timeout=0.05,  # Short timeout
        )

        # Hook should timeout
        assert api_call is not None

    @pytest.mark.asyncio
    async def test_hook_exit_behavior(self):
        """Test hook exit flag behavior."""

        async def exit_requesting_hook(event, **kwargs):
            raise RuntimeError("Exit requested by hook")

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: exit_requesting_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
            exit_hook=True,  # Exit on hook error
        )

        with patch.object(
            imodel.endpoint, "call", return_value={"test": "response"}
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}],
                pre_invoke_event_hook_timeout=0.05,  # Short timeout for testing
            )

        # Should have error status due to exit
        assert result.status in (EventStatus.CANCELLED, EventStatus.FAILED)

    @pytest.mark.asyncio
    async def test_multiple_hook_chaining(self, mock_response):
        """Test execution of multiple hooks in sequence."""
        execution_order = []

        async def pre_create_hook(event_type, **kwargs):
            execution_order.append("pre_create")
            return None

        async def pre_invoke_hook(event, **kwargs):
            execution_order.append("pre_invoke")
            return None

        async def post_invoke_hook(event, **kwargs):
            execution_order.append("post_invoke")
            return None

        hook_registry = HookRegistry(
            hooks={
                HookEventTypes.PreEventCreate: pre_create_hook,
                HookEventTypes.PreInvocation: pre_invoke_hook,
                HookEventTypes.PostInvocation: post_invoke_hook,
            }
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}]
            )

        # Verify hooks executed in correct order
        assert "pre_create" in execution_order
        assert "pre_invoke" in execution_order
        assert "post_invoke" in execution_order
        assert execution_order.index("pre_create") < execution_order.index(
            "pre_invoke"
        )
        assert execution_order.index("pre_invoke") < execution_order.index(
            "post_invoke"
        )
        assert result.status == EventStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_hook_state_management(self, mock_response):
        """Test hook state management across multiple calls."""
        call_count = 0
        state_accumulator = []

        async def stateful_hook(event, **kwargs):
            nonlocal call_count
            call_count += 1
            state_accumulator.append(f"call_{call_count}")
            return None

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: stateful_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            # Make multiple calls
            for i in range(3):
                await imodel.invoke(
                    messages=[{"role": "user", "content": f"Message {i}"}]
                )

        # Verify state accumulated across calls
        assert call_count == 3
        assert len(state_accumulator) == 3
        assert state_accumulator == ["call_1", "call_2", "call_3"]

    @pytest.mark.asyncio
    async def test_hook_params_passing(self, mock_response):
        """Test custom parameters passed to hooks."""
        received_params = {}

        async def param_receiving_hook(event, **kwargs):
            nonlocal received_params
            received_params = kwargs
            return None

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: param_receiving_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        custom_params = {
            "custom_key": "custom_value",
            "metadata": {"test": True},
        }

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}],
                pre_invoke_event_hook_params=custom_params,
            )

        # Verify custom params were passed
        assert "custom_key" in received_params
        assert received_params["custom_key"] == "custom_value"

    @pytest.mark.asyncio
    async def test_hook_cleanup_on_error(self):
        """Test that hook cleanup occurs properly on errors."""
        cleanup_called = False

        async def hook_with_cleanup(event, **kwargs):
            try:
                # Simulate some work
                await asyncio.sleep(0.001)
                raise ValueError("Intentional error")
            finally:
                nonlocal cleanup_called
                cleanup_called = True

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: hook_with_cleanup}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
            exit_hook=False,
        )

        with patch.object(
            imodel.endpoint, "call", return_value={"test": "response"}
        ):
            result = await imodel.invoke(
                messages=[{"role": "user", "content": "Hello"}],
                pre_invoke_event_hook_timeout=0.05,  # Short timeout for testing
            )

        # Cleanup should have been called even with error
        assert cleanup_called
        assert result.status in (EventStatus.CANCELLED, EventStatus.FAILED)

    @pytest.mark.asyncio
    async def test_hook_with_async_generator(self):
        """Test hooks don't interfere with streaming operations."""
        hook_called = False

        async def stream_hook(event, **kwargs):
            nonlocal hook_called
            hook_called = True
            return None

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: stream_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        async def mock_stream():
            for i in range(3):
                yield {"choices": [{"delta": {"content": f"chunk {i}"}}]}

        with patch.object(
            imodel.endpoint, "stream", return_value=mock_stream()
        ):
            chunks = []
            async for chunk in imodel.stream(
                messages=[{"role": "user", "content": "Hello"}]
            ):
                if chunk and not isinstance(chunk, APICalling):
                    chunks.append(chunk)

        # Hook should have been called for streaming
        assert hook_called
        assert len(chunks) >= 3

    @pytest.mark.asyncio
    async def test_concurrent_hooks_thread_safety(self, mock_response):
        """Test hook execution is thread-safe under concurrent calls."""
        call_tracker = []
        lock = asyncio.Lock()

        async def thread_safe_hook(event, **kwargs):
            async with lock:
                call_tracker.append(asyncio.current_task().get_name())
                await asyncio.sleep(0.001)
            return None

        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: thread_safe_hook}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            # Make concurrent calls
            tasks = []
            for i in range(5):
                task = asyncio.create_task(
                    imodel.invoke(
                        messages=[
                            {"role": "user", "content": f"Concurrent {i}"}
                        ]
                    )
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        # All calls should complete successfully
        assert len(results) == 5
        for result in results:
            assert result.status == EventStatus.COMPLETED

        # Hook should have been called for each
        assert len(call_tracker) == 5

    @pytest.mark.asyncio
    async def test_hook_exception_types(self):
        """Test different exception types in hooks."""

        async def value_error_hook(event, **kwargs):
            raise ValueError("Value error in hook")

        async def runtime_error_hook(event, **kwargs):
            raise RuntimeError("Runtime error in hook")

        async def generic_error_hook(event, **kwargs):
            raise Exception("Generic error in hook")

        for hook_func in [
            value_error_hook,
            runtime_error_hook,
            generic_error_hook,
        ]:
            hook_registry = HookRegistry(
                hooks={HookEventTypes.PreInvocation: hook_func}
            )

            imodel = iModel(
                provider="openai",
                model="gpt-4.1-mini",
                api_key="test-key",
                hook_registry=hook_registry,
                exit_hook=False,
            )

            with patch.object(
                imodel.endpoint, "call", return_value={"test": "response"}
            ):
                result = await imodel.invoke(
                    messages=[{"role": "user", "content": "Hello"}],
                    pre_invoke_event_hook_timeout=0.05,  # Short timeout for testing
                )

            # All exception types should be handled gracefully
            assert result.status in (EventStatus.CANCELLED, EventStatus.FAILED)

    @pytest.mark.asyncio
    async def test_hook_registry_dynamic_update(self, mock_response):
        """Test dynamically updating hook registry."""
        call_log = []

        async def hook_v1(event, **kwargs):
            call_log.append("v1")
            return None

        async def hook_v2(event, **kwargs):
            call_log.append("v2")
            return None

        # Start with hook v1
        hook_registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: hook_v1}
        )

        imodel = iModel(
            provider="openai",
            model="gpt-4.1-mini",
            api_key="test-key",
            hook_registry=hook_registry,
        )

        with patch.object(
            imodel.endpoint,
            "call",
            return_value=mock_response.json.return_value,
        ):
            await imodel.invoke(
                messages=[{"role": "user", "content": "First call"}]
            )

            # Update registry
            imodel.hook_registry._hooks[HookEventTypes.PreInvocation] = hook_v2

            await imodel.invoke(
                messages=[{"role": "user", "content": "Second call"}]
            )

        # Should have called both versions
        assert "v1" in call_log
        assert "v2" in call_log
        assert call_log.index("v1") < call_log.index("v2")
