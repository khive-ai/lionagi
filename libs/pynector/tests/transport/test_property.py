"""
Property-based tests for the Transport Abstraction Layer.

This module contains property-based tests for the Transport Abstraction Layer
using the hypothesis framework.
"""

from hypothesis import given
from hypothesis import strategies as st

from pynector.transport.message.binary import BinaryMessage
from pynector.transport.message.json import JsonMessage


@given(
    headers=st.dictionaries(
        keys=st.text(), values=st.one_of(st.text(), st.integers(), st.booleans())
    ),
    payload=st.one_of(
        st.dictionaries(
            keys=st.text(), values=st.one_of(st.text(), st.integers(), st.booleans())
        ),
        st.lists(st.one_of(st.text(), st.integers(), st.booleans())),
        st.text(),
        st.integers(),
        st.booleans(),
    ),
)
def test_json_message_roundtrip(headers, payload):
    """Test that JsonMessage serialization/deserialization roundtrip works."""
    message = JsonMessage(headers, payload)
    serialized = message.serialize()
    deserialized = JsonMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload


@given(
    headers=st.dictionaries(
        keys=st.text(), values=st.one_of(st.text(), st.integers(), st.booleans())
    ),
    payload=st.binary(min_size=0, max_size=1000),
)
def test_binary_message_roundtrip(headers, payload):
    """Test that BinaryMessage serialization/deserialization roundtrip works."""
    message = BinaryMessage(headers, payload)
    serialized = message.serialize()
    deserialized = BinaryMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload


@given(
    headers=st.dictionaries(
        keys=st.text(min_size=0, max_size=100),
        values=st.one_of(
            st.text(min_size=0, max_size=100),
            st.integers(),
            st.booleans(),
            st.lists(st.integers(), max_size=10),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.one_of(st.text(), st.integers(), st.booleans()),
                max_size=5,
            ),
        ),
        min_size=0,
        max_size=10,
    ),
    payload=st.binary(min_size=0, max_size=1000),
)
def test_binary_message_complex_headers(headers, payload):
    """Test BinaryMessage with complex nested headers."""
    message = BinaryMessage(headers, payload)
    serialized = message.serialize()
    deserialized = BinaryMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload


@given(
    headers=st.dictionaries(
        keys=st.text(min_size=0, max_size=100),
        values=st.one_of(
            st.text(min_size=0, max_size=100),
            st.integers(),
            st.booleans(),
            st.lists(st.integers(), max_size=10),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=10),
                values=st.one_of(st.text(), st.integers(), st.booleans()),
                max_size=5,
            ),
        ),
        min_size=0,
        max_size=10,
    ),
    payload=st.one_of(
        st.dictionaries(
            keys=st.text(min_size=0, max_size=100),
            values=st.one_of(
                st.text(min_size=0, max_size=100),
                st.integers(),
                st.booleans(),
                st.lists(st.integers(), max_size=10),
                st.none(),
            ),
            min_size=0,
            max_size=10,
        ),
        st.lists(
            st.one_of(
                st.text(min_size=0, max_size=100),
                st.integers(),
                st.booleans(),
                st.none(),
            ),
            max_size=10,
        ),
        st.text(min_size=0, max_size=100),
        st.integers(),
        st.booleans(),
        st.none(),
    ),
)
def test_json_message_complex_structures(headers, payload):
    """Test JsonMessage with complex nested structures."""
    message = JsonMessage(headers, payload)
    serialized = message.serialize()
    deserialized = JsonMessage.deserialize(serialized)

    assert deserialized.get_headers() == headers
    assert deserialized.get_payload() == payload


@given(data=st.binary(min_size=0, max_size=100))
def test_binary_message_invalid_data_handling(data):
    """Test BinaryMessage handles invalid data appropriately."""
    # This test verifies that either:
    # 1. The data is successfully deserialized, or
    # 2. An appropriate exception is raised
    try:
        message = BinaryMessage.deserialize(data)
        # If deserialization succeeded, verify we can serialize it back
        serialized = message.serialize()
        deserialized = BinaryMessage.deserialize(serialized)
        assert deserialized.get_headers() == message.get_headers()
        assert deserialized.get_payload() == message.get_payload()
    except Exception as e:
        # If an exception was raised, it should be a DeserializationError
        from pynector.transport.errors import DeserializationError

        assert isinstance(e, DeserializationError)
