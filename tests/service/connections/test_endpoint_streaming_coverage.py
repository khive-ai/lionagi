# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Coverage-targeted tests for Endpoint streaming pure-logic branches."""

from __future__ import annotations

import json

import pytest

from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.connections.endpoint_config import EndpointConfig
from lionagi.service.types.stream_chunk import StreamChunk


def make_endpoint() -> Endpoint:
    cfg = EndpointConfig(
        name="test_ep",
        endpoint="chat",
        provider="test",
        base_url="https://example.com",
        endpoint_params=["v1"],
    )
    return Endpoint(config=cfg)


# ---------------------------------------------------------------------------
# _looks_like_complete_stream_data
# ---------------------------------------------------------------------------


class TestLooksLikeCompleteStreamData:
    def test_done_marker_returns_true(self):
        assert Endpoint._looks_like_complete_stream_data(["[DONE]"]) is True

    def test_valid_json_returns_true(self):
        assert Endpoint._looks_like_complete_stream_data(['{"type":"text"}']) is True

    def test_invalid_json_returns_false(self):
        assert Endpoint._looks_like_complete_stream_data(["{bad json"]) is False

    def test_multiline_valid_json(self):
        assert Endpoint._looks_like_complete_stream_data(['{"a":1,', '"b":2}']) is True

    def test_empty_string_invalid_json(self):
        assert Endpoint._looks_like_complete_stream_data([""]) is False


# ---------------------------------------------------------------------------
# _line_to_stream_chunk
# ---------------------------------------------------------------------------


class TestLineToStreamChunk:
    def setup_method(self):
        self.ep = make_endpoint()

    def test_done_returns_result_chunk(self):
        chunk = self.ep._line_to_stream_chunk("[DONE]")
        assert chunk.type == "result"
        assert chunk.metadata.get("done") is True

    def test_data_done_prefix_returns_result_chunk(self):
        chunk = self.ep._line_to_stream_chunk("data: [DONE]")
        assert chunk.type == "result"

    def test_invalid_json_returns_text_chunk(self):
        chunk = self.ep._line_to_stream_chunk("not valid json")
        assert chunk.type == "text"
        assert chunk.content == "not valid json"
        assert chunk.is_delta is True

    def test_non_dict_json_returns_text_chunk(self):
        # list value → not a dict → line 470
        chunk = self.ep._line_to_stream_chunk("[1, 2, 3]")
        assert chunk.type == "text"
        assert chunk.content == "[1, 2, 3]"

    def test_json_integer_returns_text_chunk(self):
        chunk = self.ep._line_to_stream_chunk("42")
        assert chunk.type == "text"

    def test_event_without_known_type_returns_system_chunk(self):
        # unknown type → _event_to_stream_chunk returns None → line 480
        data = json.dumps({"type": "some_unknown_event", "data": "x"})
        chunk = self.ep._line_to_stream_chunk(data)
        assert chunk.type == "system"
        assert "raw" in chunk.metadata


# ---------------------------------------------------------------------------
# _event_to_stream_chunk
# ---------------------------------------------------------------------------


class TestEventToStreamChunk:
    def setup_method(self):
        self.ep = make_endpoint()

    def test_error_type_returns_error_chunk(self):
        event = {"type": "error", "error": {"message": "bad request"}}
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk.type == "error"
        assert chunk.is_error is True
        assert "bad request" in chunk.content

    def test_response_error_type_returns_error_chunk(self):
        event = {"type": "response.error", "error": "oops"}
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk.type == "error"

    def test_response_completed_returns_result_chunk(self):
        event = {"type": "response.completed", "response": {}}
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk.type == "result"

    def test_content_block_delta_text(self):
        event = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "hello"},
        }
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk.type == "text"
        assert chunk.content == "hello"
        assert chunk.is_delta is True

    def test_content_block_delta_thinking(self):
        event = {
            "type": "content_block_delta",
            "delta": {"type": "thinking_delta", "thinking": "pondering"},
        }
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk.type == "thinking"
        assert chunk.content == "pondering"

    def test_choices_with_tool_calls_returns_tool_use_chunk(self):
        event = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "test"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk.type == "tool_use"
        assert chunk.tool_name == "search"
        assert chunk.tool_id == "call-1"
        assert chunk.is_delta is True

    def test_choices_with_no_content_no_tools_returns_system_chunk(self):
        event = {"choices": [{"delta": {}}]}
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk.type == "system"

    def test_choices_empty_list_returns_none(self):
        event = {"choices": []}
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk is None

    def test_unknown_type_returns_none(self):
        event = {"type": "totally_unknown"}
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk is None

    def test_response_output_text_delta(self):
        event = {"type": "response.output_text.delta", "delta": "world"}
        chunk = self.ep._event_to_stream_chunk(event)
        assert chunk.type == "text"
        assert chunk.content == "world"


# ---------------------------------------------------------------------------
# _stream_line_to_chunk — SSE framing
# ---------------------------------------------------------------------------


class TestStreamLineToChunk:
    def setup_method(self):
        self.ep = make_endpoint()

    def test_empty_line_with_no_event_data_returns_none(self):
        event_data = []
        result = self.ep._stream_line_to_chunk("", event_data)
        assert result is None

    def test_empty_line_with_accumulated_event_data_flushes(self):
        event_data = ['{"type": "text", "content": "hi"}']
        result = self.ep._stream_line_to_chunk("", event_data)
        assert result is not None
        assert event_data == []

    def test_comment_line_returns_none(self):
        event_data = []
        result = self.ep._stream_line_to_chunk(": ping", event_data)
        assert result is None

    def test_event_meta_line_returns_none(self):
        for prefix in ["event: text", "id: 123", "retry: 3000"]:
            event_data = []
            result = self.ep._stream_line_to_chunk(prefix, event_data)
            assert result is None

    def test_data_line_accumulates_in_event_data(self):
        event_data = []
        result = self.ep._stream_line_to_chunk('data: {"type":"text"}', event_data)
        assert result is None
        assert len(event_data) == 1

    def test_second_complete_data_line_flushes_previous(self):
        # First data line - complete JSON
        event_data = ['{"type":"text","content":"first"}']
        # Second data line should flush first and start fresh
        result = self.ep._stream_line_to_chunk(
            'data: {"type":"text","content":"second"}', event_data
        )
        assert result is not None  # flushed previous
        assert len(event_data) == 1  # second line now in buffer
        assert "second" in event_data[0]

    def test_non_data_non_comment_line_flushes_event_data(self):
        # Accumulated event_data gets flushed when a non-data line appears
        event_data = ['{"type":"text","content":"buffered"}']
        result = self.ep._stream_line_to_chunk("some plain line", event_data)
        assert event_data == []  # flushed

    def test_non_data_no_event_data_passes_through(self):
        event_data = []
        result = self.ep._stream_line_to_chunk("[DONE]", event_data)
        assert result is not None
        assert result.type == "result"
