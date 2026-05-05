"""Surgical gap-fill tests for SystemContent missing branches.

Targets ~8 missing statements in lionagi/protocols/messages/system.py:
Lines: 34, 43, 47, 57-61
"""

import pytest

from lionagi.protocols.messages.message import MessageRole
from lionagi.protocols.messages.system import System, SystemContent

# ---------------------------------------------------------------------------
# SystemContent.rendered — datetime_factory path (line 34)
# ---------------------------------------------------------------------------


def test_rendered_with_datetime_factory():
    """Line 34: datetime_factory is called when system_datetime is None."""
    c = SystemContent(
        system_message="Hello",
        system_datetime=None,
        datetime_factory=lambda: "2026-01-01T00:00:00",
    )
    result = c.rendered
    assert "System Time: 2026-01-01T00:00:00" in result
    assert "Hello" in result


def test_rendered_with_no_datetime():
    """Lines 31-38: rendered without datetime omits System Time line."""
    c = SystemContent(system_message="Just a message")
    result = c.rendered
    assert "System Time" not in result
    assert "Just a message" in result


# ---------------------------------------------------------------------------
# SystemContent.role (line 43)
# ---------------------------------------------------------------------------


def test_role_property():
    """Line 43: role returns MessageRole.SYSTEM."""
    c = SystemContent()
    assert c.role == MessageRole.SYSTEM


# ---------------------------------------------------------------------------
# SystemContent.render method (line 47)
# ---------------------------------------------------------------------------


def test_render_method_delegates_to_rendered():
    """Line 47: render() delegates to rendered."""
    c = SystemContent(system_message="test msg")
    assert c.render() == c.rendered
    assert c.render(extra="arg") == c.rendered


# ---------------------------------------------------------------------------
# SystemContent.create — system_datetime branches (lines 57-61)
# ---------------------------------------------------------------------------


def test_create_with_system_datetime_true():
    """Line 57-58: system_datetime=True generates current datetime string."""
    c = SystemContent.create(system_message="Hello", system_datetime=True)
    assert c.system_datetime is not None
    # Should be a valid ISO format string
    assert "T" in c.system_datetime or "-" in c.system_datetime


def test_create_with_system_datetime_false():
    """Line 59-60: system_datetime=False → None."""
    c = SystemContent.create(system_message="Hello", system_datetime=False)
    assert c.system_datetime is None


def test_create_with_none_message_uses_default():
    """Line 62-65: system_message=None uses default message."""
    c = SystemContent.create(system_message=None)
    assert "helpful AI assistant" in c.system_message


def test_create_with_callable_datetime_factory():
    """Line 65: callable datetime_factory is stored."""
    factory = lambda: "fixed-time"
    c = SystemContent.create(system_message="msg", datetime_factory=factory)
    assert callable(c.datetime_factory)
    assert c.datetime_factory() == "fixed-time"


def test_create_with_non_callable_datetime_factory():
    """Line 65: non-callable datetime_factory is discarded."""
    c = SystemContent.create(system_message="msg", datetime_factory="not-callable")
    assert c.datetime_factory is None


# ---------------------------------------------------------------------------
# System validator — bad type
# ---------------------------------------------------------------------------


def test_system_validator_raises_on_bad_type():
    """TypeError raised for unsupported content type."""
    with pytest.raises(TypeError, match="content must be dict or SystemContent"):
        System(content=42)
