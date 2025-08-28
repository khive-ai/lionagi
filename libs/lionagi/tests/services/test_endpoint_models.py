# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Tests for msgspec-based endpoint request models.

Tests msgspec serialization/deserialization, field validation, and model behavior
for RequestModel, ChatRequestModel, CompletionRequestModel, and EmbeddingRequestModel.
"""

import json

import msgspec
import msgspec.json

from lionagi.services.endpoint import (
    ChatRequestModel,
    CompletionRequestModel,
    EmbeddingRequestModel,
    RequestModel,
)


class TestRequestModelBase:
    """Tests for base RequestModel msgspec compliance and behavior."""

    def test_msgspec_compliance_and_serialization(self):
        """RequestModel msgspec compliance with complete roundtrip serialization."""
        # Verify RequestModel is msgspec.Struct for v1 performance
        assert issubclass(
            RequestModel, msgspec.Struct
        ), "RequestModel must inherit from msgspec.Struct"

        # Create RequestModel with all fields populated
        model = RequestModel(
            model="gpt-4",
            stream=True,
        )

        # Test msgspec serialization/deserialization roundtrip
        encoded = msgspec.json.encode(model)
        decoded = msgspec.json.decode(encoded, type=RequestModel)

        # Validate complete data preservation
        assert decoded.model == "gpt-4"
        assert decoded.stream is True

    def test_default_values(self):
        """RequestModel default values are set correctly."""
        model = RequestModel()

        assert model.model is None
        assert model.stream is False

    def test_field_assignment(self):
        """RequestModel fields can be assigned values correctly."""
        model = RequestModel(model="test-model", stream=True)

        assert model.model == "test-model"
        assert model.stream is True

    def test_allows_extra_fields(self):
        """RequestModel allows extra fields via msgspec default behavior."""
        # Create model with extra fields
        data = {
            "model": "gpt-4",
            "stream": False,
            "extra_field": "extra_value",
            "custom_param": 42,
        }

        # This should work since msgspec allows extra fields by default
        encoded = msgspec.json.encode(data)
        decoded = msgspec.json.decode(encoded, type=RequestModel)

        assert decoded.model == "gpt-4"
        assert decoded.stream is False
        # Note: extra fields are not preserved in the decoded object,
        # but msgspec doesn't raise an error during decoding


class TestChatRequestModel:
    """Tests for ChatRequestModel msgspec compliance and chat-specific fields."""

    def test_msgspec_compliance_and_serialization(self):
        """ChatRequestModel msgspec compliance with complete roundtrip."""
        assert issubclass(
            ChatRequestModel, RequestModel
        ), "ChatRequestModel must inherit from RequestModel"

        # Create complex ChatRequestModel with all fields populated
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        model = ChatRequestModel(
            model="gpt-4",
            stream=True,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["\\n\\n", "END"],
        )

        # Test msgspec serialization/deserialization roundtrip
        encoded = msgspec.json.encode(model)
        decoded = msgspec.json.decode(encoded, type=ChatRequestModel)

        # Validate complete data preservation
        assert decoded.model == "gpt-4"
        assert decoded.stream is True
        assert decoded.messages == messages
        assert decoded.temperature == 0.7
        assert decoded.max_tokens == 150
        assert decoded.top_p == 0.9
        assert decoded.frequency_penalty == 0.1
        assert decoded.presence_penalty == 0.2
        assert decoded.stop == ["\\n\\n", "END"]

    def test_default_values(self):
        """ChatRequestModel default values are set correctly."""
        messages = [{"role": "user", "content": "Test message"}]
        model = ChatRequestModel(messages=messages)

        # Base RequestModel defaults
        assert model.model is None
        assert model.stream is False

        # ChatRequestModel specific defaults
        assert model.messages == messages
        assert model.temperature == 1.0
        assert model.max_tokens is None
        assert model.top_p == 1.0
        assert model.frequency_penalty == 0.0
        assert model.presence_penalty == 0.0
        assert model.stop is None

    def test_stop_parameter_variants(self):
        """ChatRequestModel stop parameter accepts string, list, or None."""
        messages = [{"role": "user", "content": "Test"}]

        # Test with string stop
        model_str = ChatRequestModel(messages=messages, stop="END")
        encoded = msgspec.json.encode(model_str)
        decoded = msgspec.json.decode(encoded, type=ChatRequestModel)
        assert decoded.stop == "END"

        # Test with list stop
        model_list = ChatRequestModel(messages=messages, stop=["END", "STOP"])
        encoded = msgspec.json.encode(model_list)
        decoded = msgspec.json.decode(encoded, type=ChatRequestModel)
        assert decoded.stop == ["END", "STOP"]

        # Test with None stop
        model_none = ChatRequestModel(messages=messages, stop=None)
        encoded = msgspec.json.encode(model_none)
        decoded = msgspec.json.decode(encoded, type=ChatRequestModel)
        assert decoded.stop is None

    def test_complex_messages_structure(self):
        """ChatRequestModel handles complex message structures correctly."""
        complex_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with advanced capabilities.",
            },
            {
                "role": "user",
                "content": "Analyze this data and provide insights.",
                "metadata": {"priority": "high", "timestamp": 1234567890},
            },
            {
                "role": "assistant",
                "content": "I'll analyze the data for you.",
                "function_call": {
                    "name": "analyze_data",
                    "arguments": '{"data": "sample"}',
                },
            },
        ]

        model = ChatRequestModel(messages=complex_messages, model="gpt-4")

        # Test serialization preserves complex nested structures
        encoded = msgspec.json.encode(model)
        decoded = msgspec.json.decode(encoded, type=ChatRequestModel)

        assert decoded.messages == complex_messages
        assert decoded.messages[1]["metadata"]["priority"] == "high"
        assert decoded.messages[2]["function_call"]["name"] == "analyze_data"


class TestCompletionRequestModel:
    """Tests for CompletionRequestModel msgspec compliance and completion-specific fields."""

    def test_msgspec_compliance_and_serialization(self):
        """CompletionRequestModel msgspec compliance with complete roundtrip."""
        assert issubclass(
            CompletionRequestModel, RequestModel
        ), "CompletionRequestModel must inherit from RequestModel"

        # Create complex CompletionRequestModel
        model = CompletionRequestModel(
            model="text-davinci-003",
            stream=False,
            prompt="Write a story about",
            temperature=0.8,
            max_tokens=200,
            top_p=0.95,
            frequency_penalty=0.3,
            presence_penalty=0.4,
            stop=["The End", "\\n\\n\\n"],
        )

        # Test msgspec serialization/deserialization roundtrip
        encoded = msgspec.json.encode(model)
        decoded = msgspec.json.decode(encoded, type=CompletionRequestModel)

        # Validate complete data preservation
        assert decoded.model == "text-davinci-003"
        assert decoded.stream is False
        assert decoded.prompt == "Write a story about"
        assert decoded.temperature == 0.8
        assert decoded.max_tokens == 200
        assert decoded.top_p == 0.95
        assert decoded.frequency_penalty == 0.3
        assert decoded.presence_penalty == 0.4
        assert decoded.stop == ["The End", "\\n\\n\\n"]

    def test_default_values(self):
        """CompletionRequestModel default values are set correctly."""
        model = CompletionRequestModel(prompt="Test prompt")

        # Base RequestModel defaults
        assert model.model is None
        assert model.stream is False

        # CompletionRequestModel specific defaults
        assert model.prompt == "Test prompt"
        assert model.temperature == 1.0
        assert model.max_tokens is None
        assert model.top_p == 1.0
        assert model.frequency_penalty == 0.0
        assert model.presence_penalty == 0.0
        assert model.stop is None


class TestEmbeddingRequestModel:
    """Tests for EmbeddingRequestModel msgspec compliance and embedding-specific fields."""

    def test_msgspec_compliance_and_serialization(self):
        """EmbeddingRequestModel msgspec compliance with complete roundtrip."""
        assert issubclass(
            EmbeddingRequestModel, RequestModel
        ), "EmbeddingRequestModel must inherit from RequestModel"

        # Create complex EmbeddingRequestModel with list input
        model = EmbeddingRequestModel(
            model="text-embedding-ada-002",
            input=["Text to embed", "Another text to embed", "Third text"],
            encoding_format="base64",
        )

        # Test msgspec serialization/deserialization roundtrip
        encoded = msgspec.json.encode(model)
        decoded = msgspec.json.decode(encoded, type=EmbeddingRequestModel)

        # Validate complete data preservation
        assert decoded.model == "text-embedding-ada-002"
        assert decoded.input == ["Text to embed", "Another text to embed", "Third text"]
        assert decoded.encoding_format == "base64"

    def test_input_variants(self):
        """EmbeddingRequestModel input accepts string or list of strings."""
        # Test with string input
        model_str = EmbeddingRequestModel(input="Single text to embed")
        encoded = msgspec.json.encode(model_str)
        decoded = msgspec.json.decode(encoded, type=EmbeddingRequestModel)
        assert decoded.input == "Single text to embed"

        # Test with list input
        model_list = EmbeddingRequestModel(input=["Text 1", "Text 2", "Text 3"])
        encoded = msgspec.json.encode(model_list)
        decoded = msgspec.json.decode(encoded, type=EmbeddingRequestModel)
        assert decoded.input == ["Text 1", "Text 2", "Text 3"]

    def test_default_values(self):
        """EmbeddingRequestModel default values are set correctly."""
        model = EmbeddingRequestModel(input="Test input")

        # Base RequestModel defaults
        assert model.model is None
        assert model.stream is False

        # EmbeddingRequestModel specific defaults
        assert model.input == "Test input"
        assert model.encoding_format == "float"


class TestModelInteroperability:
    """Tests for model interoperability and edge cases."""

    def test_model_inheritance_chain(self):
        """All models properly inherit from RequestModel and msgspec.Struct."""
        models = [
            RequestModel,
            ChatRequestModel,
            CompletionRequestModel,
            EmbeddingRequestModel,
        ]

        for model_class in models:
            assert issubclass(model_class, msgspec.Struct)
            assert issubclass(model_class, RequestModel)

    def test_serialization_with_none_values(self):
        """Models handle None values correctly in serialization."""
        # Test ChatRequestModel with None values
        model = ChatRequestModel(
            messages=[{"role": "user", "content": "test"}],
            model=None,
            max_tokens=None,
            stop=None,
        )

        encoded = msgspec.json.encode(model)
        decoded = msgspec.json.decode(encoded, type=ChatRequestModel)

        assert decoded.model is None
        assert decoded.max_tokens is None
        assert decoded.stop is None

    def test_large_data_serialization(self):
        """Models handle large data structures correctly."""
        # Create large messages list
        large_messages = [
            {"role": "user", "content": f"Message number {i} with content"} for i in range(100)
        ]

        model = ChatRequestModel(messages=large_messages, model="gpt-4")

        # Should handle large data without issues
        encoded = msgspec.json.encode(model)
        decoded = msgspec.json.decode(encoded, type=ChatRequestModel)

        assert len(decoded.messages) == 100
        assert decoded.messages[99]["content"] == "Message number 99 with content"

    def test_json_compatibility(self):
        """Models are compatible with standard JSON serialization."""
        model = ChatRequestModel(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            temperature=0.7,
        )

        # Convert to dict and use standard JSON
        model_dict = msgspec.to_builtins(model)
        json_str = json.dumps(model_dict)
        parsed = json.loads(json_str)

        assert parsed["model"] == "gpt-4"
        assert parsed["temperature"] == 0.7
        assert parsed["messages"][0]["content"] == "Test"
