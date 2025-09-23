# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Test HookedEvent integration with pre/post hooks."""

import pytest

from lionagi.protocols.types import EventStatus
from lionagi.service.hooks._types import HookEventTypes
from lionagi.service.hooks.hook_registry import HookRegistry
from lionagi.service.hooks.hooked_event import HookedEvent
from tests.service.hooks.conftest import MyCancelled


class MockHookedEvent(HookedEvent):
    """Test implementation of HookedEvent for testing."""

    def __init__(self, invoke_result="test_invoke_result", invoke_error=None):
        super().__init__()
        self.invoke_result = invoke_result
        self.invoke_error = invoke_error
        self.invoke_called = False

    async def _invoke(self):
        """Test implementation that returns configured result or raises error."""
        self.invoke_called = True
        if self.invoke_error:
            raise self.invoke_error
        return self.invoke_result

    async def _stream(self):
        """Test implementation for streaming (not used in these tests)."""
        yield "test_chunk"


class MockHookedEventPreHookIntegration:
    """Test pre-invocation hook integration."""

    @pytest.mark.anyio
    async def test_pre_hook_normal_allows_invoke(
        self, patch_cancellation, patch_logger
    ):
        """Test that normal pre-hook execution allows _invoke() to proceed."""

        async def pre_hook(ev, **kw):
            return "pre_ok"

        registry = HookRegistry(hooks={HookEventTypes.PreInvocation: pre_hook})
        event = MockHookedEvent(invoke_result="main_result")
        event.create_pre_invoke_hook(hook_registry=registry, exit_hook=False)

        await event.invoke()

        # Pre-hook should have run and allowed main invoke
        assert event.invoke_called is True
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.response == "main_result"
        assert event.execution.error is None

        # Logger should have been called once for the pre-hook
        assert len(patch_logger) == 1

    @pytest.mark.anyio
    async def test_pre_hook_exit_aborts_invoke_and_logs_once(
        self, patch_cancellation, patch_logger
    ):
        """Test that pre-hook exit aborts _invoke() and logs once."""

        async def pre_hook(ev, **kw):
            raise MyCancelled("pre-hook denied")

        registry = HookRegistry(hooks={HookEventTypes.PreInvocation: pre_hook})
        event = MockHookedEvent(invoke_result="SHOULD_NOT_HAPPEN")
        event.create_pre_invoke_hook(hook_registry=registry, exit_hook=True)

        await event.invoke()

        # Main _invoke() should NOT have been called
        assert event.invoke_called is False
        assert event.execution.status == EventStatus.FAILED
        assert "Pre-invocation hook requested exit" in event.execution.error

        # Pre-hook should have been logged once
        assert len(patch_logger) == 1

    @pytest.mark.anyio
    async def test_pre_hook_error_with_exit_false_continues(
        self, patch_cancellation, patch_logger
    ):
        """Test that pre-hook error with exit=False still allows continuation."""

        async def pre_hook(ev, **kw):
            raise RuntimeError("pre-hook error")

        registry = HookRegistry(hooks={HookEventTypes.PreInvocation: pre_hook})
        event = MockHookedEvent(invoke_result="main_result")
        event.create_pre_invoke_hook(hook_registry=registry, exit_hook=False)

        await event.invoke()

        # Main _invoke() should still be called because exit_hook=False
        assert event.invoke_called is True
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.response == "main_result"

        # Pre-hook should have been logged
        assert len(patch_logger) == 1

    @pytest.mark.anyio
    async def test_pre_hook_error_with_exit_true_aborts(
        self, patch_cancellation, patch_logger
    ):
        """Test that pre-hook error with exit=True aborts execution."""

        async def pre_hook(ev, **kw):
            raise RuntimeError("pre-hook critical error")

        registry = HookRegistry(hooks={HookEventTypes.PreInvocation: pre_hook})
        event = MockHookedEvent(invoke_result="SHOULD_NOT_HAPPEN")
        event.create_pre_invoke_hook(hook_registry=registry, exit_hook=True)

        await event.invoke()

        # Main _invoke() should NOT have been called
        assert event.invoke_called is False
        assert event.execution.status == EventStatus.FAILED
        assert "pre-hook critical error" in event.execution.error

        # Pre-hook should have been logged once
        assert len(patch_logger) == 1


class MockHookedEventPostHookIntegration:
    """Test post-invocation hook integration."""

    @pytest.mark.anyio
    async def test_post_hook_normal_completion(
        self, patch_cancellation, patch_logger
    ):
        """Test that normal post-hook execution completes successfully."""

        async def post_hook(ev, **kw):
            return "post_logged"

        registry = HookRegistry(
            hooks={HookEventTypes.PostInvocation: post_hook}
        )
        event = MockHookedEvent(invoke_result="main_result")
        event.create_post_invoke_hook(hook_registry=registry, exit_hook=False)

        await event.invoke()

        # Both main invoke and post-hook should have run
        assert event.invoke_called is True
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.response == "main_result"

        # Post-hook should have been logged once
        assert len(patch_logger) == 1

    @pytest.mark.anyio
    async def test_post_hook_exit_discards_main_result(
        self, patch_cancellation, patch_logger
    ):
        """Test that post-hook exit discards main result and fails."""

        async def post_hook(ev, **kw):
            raise MyCancelled("post-hook failed")

        registry = HookRegistry(
            hooks={HookEventTypes.PostInvocation: post_hook}
        )
        event = MockHookedEvent(invoke_result="main_result")
        event.create_post_invoke_hook(hook_registry=registry, exit_hook=True)

        await event.invoke()

        # Main invoke should have run, but result discarded due to post-hook exit
        assert event.invoke_called is True
        assert event.execution.status == EventStatus.FAILED
        assert "Post-invocation hook requested exit" in event.execution.error
        # Response should be None because hook exit discarded it
        assert event.execution.response is None

        # Post-hook should have been logged once
        assert len(patch_logger) == 1

    @pytest.mark.anyio
    async def test_post_hook_error_with_exit_false_keeps_result(
        self, patch_cancellation, patch_logger
    ):
        """Test that post-hook error with exit=False keeps main result."""

        async def post_hook(ev, **kw):
            raise RuntimeError("post-hook error")

        registry = HookRegistry(
            hooks={HookEventTypes.PostInvocation: post_hook}
        )
        event = MockHookedEvent(invoke_result="main_result")
        event.create_post_invoke_hook(hook_registry=registry, exit_hook=False)

        await event.invoke()

        # Main result should be preserved because exit_hook=False
        assert event.invoke_called is True
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.response == "main_result"

        # Post-hook should have been logged
        assert len(patch_logger) == 1


class MockHookedEventBothHooks:
    """Test HookedEvent with both pre and post hooks."""

    @pytest.mark.anyio
    async def test_both_hooks_normal_execution_order(
        self, patch_cancellation, patch_logger
    ):
        """Test that both hooks run in correct order: pre -> _invoke -> post."""
        execution_order = []

        async def pre_hook(ev, **kw):
            execution_order.append("pre")
            return "pre_ok"

        async def post_hook(ev, **kw):
            execution_order.append("post")
            return "post_ok"

        class OrderTestEvent(MockHookedEvent):
            async def _invoke(self):
                execution_order.append("main")
                return await super()._invoke()

        registry = HookRegistry(
            hooks={
                HookEventTypes.PreInvocation: pre_hook,
                HookEventTypes.PostInvocation: post_hook,
            }
        )
        event = OrderTestEvent(invoke_result="main_result")
        event.create_pre_invoke_hook(hook_registry=registry, exit_hook=False)
        event.create_post_invoke_hook(hook_registry=registry, exit_hook=False)

        await event.invoke()

        # Check execution order
        assert execution_order == ["pre", "main", "post"]
        assert event.execution.status == EventStatus.COMPLETED
        assert event.execution.response == "main_result"

        # Both hooks should have been logged
        assert len(patch_logger) == 2

    @pytest.mark.anyio
    async def test_pre_hook_exit_prevents_post_hook(
        self, patch_cancellation, patch_logger
    ):
        """Test that pre-hook exit prevents both _invoke and post-hook."""
        hooks_called = []

        async def pre_hook(ev, **kw):
            hooks_called.append("pre")
            raise MyCancelled("pre exit")

        async def post_hook(ev, **kw):
            hooks_called.append("post")  # Should never be called
            return "post_ok"

        registry = HookRegistry(
            hooks={
                HookEventTypes.PreInvocation: pre_hook,
                HookEventTypes.PostInvocation: post_hook,
            }
        )
        event = MockHookedEvent(invoke_result="SHOULD_NOT_HAPPEN")
        event.create_pre_invoke_hook(hook_registry=registry, exit_hook=True)
        event.create_post_invoke_hook(hook_registry=registry, exit_hook=False)

        await event.invoke()

        # Only pre-hook should have been called
        assert hooks_called == ["pre"]
        assert event.invoke_called is False
        assert event.execution.status == EventStatus.FAILED

        # Only pre-hook should have been logged
        assert len(patch_logger) == 1

    @pytest.mark.anyio
    async def test_main_invoke_error_still_runs_post_hook(
        self, patch_cancellation, patch_logger
    ):
        """Test that _invoke errors still allow post-hook to run."""
        hooks_called = []

        async def pre_hook(ev, **kw):
            hooks_called.append("pre")
            return "pre_ok"

        async def post_hook(ev, **kw):
            hooks_called.append("post")
            return "post_ok"

        registry = HookRegistry(
            hooks={
                HookEventTypes.PreInvocation: pre_hook,
                HookEventTypes.PostInvocation: post_hook,
            }
        )
        event = MockHookedEvent(
            invoke_error=RuntimeError("main invoke failed")
        )
        event.create_pre_invoke_hook(hook_registry=registry, exit_hook=False)
        event.create_post_invoke_hook(hook_registry=registry, exit_hook=False)

        await event.invoke()

        # Both hooks should have been called despite main invoke error
        assert hooks_called == ["pre", "post"]
        assert event.invoke_called is True
        assert event.execution.status == EventStatus.FAILED
        assert "main invoke failed" in event.execution.error

        # Both hooks should have been logged
        assert len(patch_logger) == 2


class MockHookedEventParameterForwarding:
    """Test parameter forwarding in HookedEvent hook creation."""

    @pytest.mark.anyio
    async def test_hook_params_forwarded_to_hook(
        self, patch_cancellation, patch_logger
    ):
        """Test that hook_params are forwarded to the hook function."""
        captured_params = {}

        async def param_hook(ev, **kw):
            captured_params.update(kw)
            return "ok"

        registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: param_hook}
        )
        event = MockHookedEvent()
        event.create_pre_invoke_hook(
            hook_registry=registry,
            exit_hook=False,
            hook_timeout=60.0,
            hook_params={"custom": "value", "number": 42},
        )

        await event.invoke()

        # Check that custom params were forwarded
        assert captured_params["custom"] == "value"
        assert captured_params["number"] == 42
        assert captured_params["exit"] is False

    @pytest.mark.anyio
    async def test_hook_timeout_configuration(self, patch_cancellation):
        """Test that hook timeout is properly configured."""
        registry = HookRegistry(
            hooks={HookEventTypes.PreInvocation: lambda ev, **kw: "ok"}
        )
        event = MockHookedEvent()
        event.create_pre_invoke_hook(
            hook_registry=registry, exit_hook=True, hook_timeout=120.0
        )

        # Check that the hook event was configured with correct timeout
        assert event._pre_invoke_hook_event.timeout == 120.0
        assert event._pre_invoke_hook_event.exit is True

    @pytest.mark.anyio
    async def test_hook_creation_defaults(self, patch_cancellation):
        """Test default values for hook creation parameters."""
        registry = HookRegistry(
            hooks={HookEventTypes.PostInvocation: lambda ev, **kw: "ok"}
        )
        event = MockHookedEvent()
        event.create_post_invoke_hook(hook_registry=registry)

        # Check defaults
        hook_event = event._post_invoke_hook_event
        assert hook_event.exit is False  # Default exit_hook=None -> False
        assert hook_event.timeout == 30.0  # Default timeout
        assert hook_event.params == {}  # Default empty params


class MockHookedEventCancellationPropagation:
    """Test cancellation propagation in HookedEvent."""

    @pytest.mark.anyio
    async def test_main_invoke_cancellation_propagates(
        self, patch_cancellation, patch_logger
    ):
        """Test that cancellation in _invoke() propagates correctly."""
        registry = HookRegistry(
            hooks={HookEventTypes.PostInvocation: lambda ev, **kw: "post"}
        )
        event = MockHookedEvent(invoke_error=MyCancelled("main cancelled"))
        event.create_post_invoke_hook(hook_registry=registry, exit_hook=False)

        # Cancellation should propagate out
        with pytest.raises(MyCancelled, match="main cancelled"):
            await event.invoke()

        # Status should be CANCELLED
        assert event.execution.status == EventStatus.CANCELLED
        assert event.execution.error == "Invocation cancelled"

        # Post hook should not have run due to cancellation
        assert len(patch_logger) == 0
