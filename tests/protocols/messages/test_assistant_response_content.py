"""Surgical gap-fill tests for AssistantResponseContent missing branches.

Targets ~15 missing statements in lionagi/protocols/messages/assistant_response.py:
Lines: 74-75, 105, 110, 119, 133-136, 151-156, 185
"""

import pytest
from pydantic import BaseModel

from lionagi.protocols.messages.assistant_response import (
    AssistantResponse,
    AssistantResponseContent,
    parse_assistant_response,
    parse_to_assistant_message,
)
from lionagi.protocols.messages.message import MessageRole

# ---------------------------------------------------------------------------
# parse_assistant_response — missing branches
# ---------------------------------------------------------------------------


def test_parse_output_message_type_with_string_content():
    """Lines 74-75: output item content items that are str."""
    response = {
        "output": [
            {
                "type": "message",
                "content": ["plain string content"],
            }
        ]
    }
    text, _ = parse_assistant_response(response)
    assert text == "plain string content"


def test_parse_output_message_type_with_output_text():
    """Line 72-73 path (output_text type)."""
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ]
    }
    text, _ = parse_assistant_response(response)
    assert text == "hello"


# ---------------------------------------------------------------------------
# AssistantResponseContent properties
# ---------------------------------------------------------------------------


def test_role_property():
    """Line 105: role returns MessageRole.ASSISTANT."""
    c = AssistantResponseContent(assistant_response="hi")
    assert c.role == MessageRole.ASSISTANT


def test_response_property_alias():
    """Line 110: response is alias for assistant_response."""
    c = AssistantResponseContent(assistant_response="the answer")
    assert c.response == "the answer"


def test_render_method_delegates():
    """Line 119: render() delegates to rendered."""
    c = AssistantResponseContent(assistant_response="result")
    assert c.render() == "result"
    assert c.render(some_arg=1) == "result"


# ---------------------------------------------------------------------------
# AssistantResponseContent.create classmethod
# ---------------------------------------------------------------------------


def test_create_with_response_object_has_data():
    """Lines 133-135: response_object with .data attribute."""

    class FakeResponse:
        data = "extracted text"

    c = AssistantResponseContent.create(response_object=FakeResponse())
    assert c.assistant_response == "extracted text"


def test_create_with_response_object_data_none():
    """Line 135: response_object.data is None → empty string."""

    class FakeResponse:
        data = None

    c = AssistantResponseContent.create(response_object=FakeResponse())
    assert c.assistant_response == ""


def test_create_with_response_object_no_data_attr():
    """Lines 133-136: response_object with no .data attr → empty string."""
    # object() has no .data, getattr returns None → text = "" → assistant_response = ""
    c = AssistantResponseContent.create(response_object=object())
    assert c.assistant_response == ""


def test_create_with_text_only():
    """Line 136: response_object is None, text is used directly."""
    c = AssistantResponseContent.create(text="just text")
    assert c.assistant_response == "just text"


# ---------------------------------------------------------------------------
# parse_to_assistant_message
# ---------------------------------------------------------------------------


def test_parse_to_assistant_message():
    """Lines 151-156: parse_to_assistant_message creates Message."""

    class FakeResponse:
        serialized = {"raw": "data"}
        metadata = {"tokens": 42}

    class FakeData:
        data = "response text"

    class FakeFullResponse:
        serialized = {"raw": "data"}
        metadata = {"model": "gpt-4"}
        data = "response text"

    # The function expects response with .serialized and .metadata
    # and passes it to AssistantResponseContent.create(response_object=response)
    class MinimalResponse:
        serialized = {"raw": "value"}
        metadata = {"extra": "info"}
        data = "the answer"

    msg = parse_to_assistant_message(MinimalResponse())
    # Result is a Message with AssistantResponseContent
    assert hasattr(msg, "content")
    assert hasattr(msg, "metadata")
    assert "raw_response" in msg.metadata


# ---------------------------------------------------------------------------
# AssistantResponse.from_response
# ---------------------------------------------------------------------------


def test_from_response_with_string():
    """Line 185 path via from_response."""
    ar = AssistantResponse.from_response("plain text response")
    assert ar.response == "plain text response"
    assert ar.recipient == MessageRole.USER


def test_from_response_with_model_response():
    """Lines 209-216: from_response with various formats."""
    response = {"choices": [{"message": {"content": "openai style"}}]}
    ar = AssistantResponse.from_response(response, sender="assistant")
    assert ar.response == "openai style"
    assert ar.model_response == response


def test_assistant_response_content_validate_raises_on_bad_type():
    """Line 185: _validate_content raises TypeError for bad content type."""
    with pytest.raises(
        TypeError, match="content must be dict or AssistantResponseContent"
    ):
        AssistantResponse(content=12345)
