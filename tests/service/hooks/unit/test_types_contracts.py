# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Test contracts for hook types and enums."""

from lionagi.service.hooks._types import (
    ALLOWED_HOOKS_TYPES,
    AssosiatedEventInfo,
    HookDict,
    HookEventTypes,
)


class TestHookEventTypes:
    def test_hook_event_types_enum_values(self):
        """Test that HookEventTypes contains the expected values."""
        assert HookEventTypes.PreEventCreate == "pre_event_create"
        assert HookEventTypes.PreInvocation == "pre_invocation"
        assert HookEventTypes.PostInvocation == "post_invocation"

    def test_allowed_hooks_types_contains_all(self):
        """Test that ALLOWED_HOOKS_TYPES contains all HookEventTypes."""
        expected = {
            HookEventTypes.PreEventCreate,
            HookEventTypes.PreInvocation,
            HookEventTypes.PostInvocation,
        }
        assert set(ALLOWED_HOOKS_TYPES) == expected


class TestAssosiatedEventInfo:
    def test_assosiated_event_info_structure(self):
        """Test AssosiatedEventInfo TypedDict structure."""
        # Test that we can create instances with expected keys
        info = AssosiatedEventInfo(
            lion_class="test.Module.Event",
            event_id="E123",
            event_created_at=42.0,
        )
        assert info["lion_class"] == "test.Module.Event"
        assert info["event_id"] == "E123"
        assert info["event_created_at"] == 42.0

    def test_assosiated_event_info_partial(self):
        """Test AssosiatedEventInfo works with partial data (total=False)."""
        info = AssosiatedEventInfo(lion_class="test.Event")
        assert info["lion_class"] == "test.Event"
        # Should not have other keys
        assert "event_id" not in info
        assert "event_created_at" not in info


class TestHookDict:
    def test_hook_dict_structure(self):
        """Test HookDict TypedDict structure."""
        hook_dict = HookDict(
            pre_event_create=lambda: None,
            pre_invocation=lambda: None,
            post_invocation=lambda: None,
        )
        assert callable(hook_dict["pre_event_create"])
        assert callable(hook_dict["pre_invocation"])
        assert callable(hook_dict["post_invocation"])

    def test_hook_dict_with_none_values(self):
        """Test HookDict allows None values."""
        hook_dict = HookDict(
            pre_event_create=None,
            pre_invocation=lambda: None,
            post_invocation=None,
        )
        assert hook_dict["pre_event_create"] is None
        assert callable(hook_dict["pre_invocation"])
        assert hook_dict["post_invocation"] is None
