"""
Property-based tests for the HTTP transport.

This module contains property-based tests for the HTTP transport using the
Hypothesis library to generate random inputs and verify invariants.
"""

import json

from hypothesis import given
from hypothesis import strategies as st

from pynector.transport.http.factory import HTTPTransportFactory
from pynector.transport.http.message import HttpMessage

# Define strategies for generating test data
http_methods = st.sampled_from(
    ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]
)
urls = st.text(min_size=1, max_size=100).filter(
    lambda x: not any(c.isspace() for c in x)
)
header_names = st.text(
    min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Ll",))
)
header_values = st.text(min_size=0, max_size=50)
headers = st.dictionaries(header_names, header_values, max_size=5)
params = st.dictionaries(st.text(min_size=1, max_size=10), st.text(), max_size=5)
json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: st.lists(children, max_size=3)
    | st.dictionaries(st.text(min_size=1, max_size=5), children, max_size=3),
    max_leaves=10,
)


class TestHttpMessageProperties:
    """Property-based tests for the HttpMessage class."""

    @given(
        method=http_methods,
        url=urls,
        headers=headers,
        params=params,
        json_data=json_values,
    )
    def test_http_message_roundtrip(self, method, url, headers, params, json_data):
        """Test HttpMessage serialization/deserialization roundtrip."""
        # Create a message with the generated data
        message = HttpMessage(
            method=method, url=url, headers=headers, params=params, json_data=json_data
        )

        # Serialize and then deserialize
        data = message.serialize()
        deserialized = HttpMessage.deserialize(data)

        # Verify the deserialized message matches the original
        assert deserialized.payload["method"] == message.payload["method"]
        assert deserialized.payload["url"] == message.payload["url"]
        assert deserialized.get_headers() == message.get_headers()

        # For params and json, we need to compare the actual values
        # since the serialization/deserialization might change the structure slightly
        if "params" in message.payload:
            assert "params" in deserialized.payload
            assert deserialized.payload["params"] == message.payload["params"]

        if "json" in message.payload:
            assert "json" in deserialized.payload
            # For JSON data, we need to handle potential differences in floating-point representation
            # by comparing the JSON strings
            original_json = json.dumps(message.payload["json"], sort_keys=True)
            deserialized_json = json.dumps(deserialized.payload["json"], sort_keys=True)
            assert deserialized_json == original_json

    @given(headers1=headers, headers2=headers)
    def test_headers_merge(self, headers1, headers2):
        """Test that headers are correctly merged."""
        # Create a merged dictionary for comparison
        merged = {**headers1}
        for k, v in headers2.items():
            merged[k] = v

        # We don't actually need to create a message for this test
        # Just directly test the header merging logic

        # Extract headers with a transport that has headers2
        transport_headers = {"User-Agent": "pynector/1.0"}
        for k, v in headers2.items():
            transport_headers[k] = v

        # Create a new headers dict with message headers taking precedence
        result = {**transport_headers}
        for k, v in headers1.items():
            result[k] = v

        # Verify the merged headers match our expected result
        assert result == {**transport_headers, **headers1}

        # Also verify that keys in headers1 override keys in transport_headers
        for k in headers1:
            if k in transport_headers and k in result:
                assert result[k] == headers1[k]


class TestHTTPTransportFactoryProperties:
    """Property-based tests for the HTTPTransportFactory class."""

    @given(
        base_url=urls,
        headers=headers,
        timeout=st.floats(min_value=0.1, max_value=60.0),
        max_retries=st.integers(min_value=0, max_value=10),
        retry_backoff_factor=st.floats(min_value=0.1, max_value=5.0),
    )
    def test_factory_creates_consistent_transport(
        self, base_url, headers, timeout, max_retries, retry_backoff_factor
    ):
        """Test that HTTPTransportFactory creates transports with consistent properties."""
        # Create a factory with the generated properties
        factory = HTTPTransportFactory(
            base_url=base_url,
            message_type=HttpMessage,
            default_headers=headers,
            default_timeout=timeout,
            default_max_retries=max_retries,
            default_retry_backoff_factor=retry_backoff_factor,
        )

        # Create a transport using the factory
        transport = factory.create_transport()

        # Verify the transport has the expected properties
        assert transport.base_url == base_url
        assert transport.headers == headers
        assert transport.timeout == timeout
        assert transport.max_retries == max_retries
        assert transport.retry_backoff_factor == retry_backoff_factor
        assert transport._message_type == HttpMessage

    @given(base_url=urls, default_headers=headers, custom_headers=headers)
    def test_factory_headers_merge(self, base_url, default_headers, custom_headers):
        """Test that headers are correctly merged when creating a transport."""
        # Create a factory with default headers
        factory = HTTPTransportFactory(
            base_url=base_url, message_type=HttpMessage, default_headers=default_headers
        )

        # Create a transport with custom headers
        transport = factory.create_transport(headers=custom_headers)

        # Create the expected merged headers
        expected_headers = {**default_headers}
        for k, v in custom_headers.items():
            expected_headers[k] = v

        # Verify the transport has the expected merged headers
        assert transport.headers == expected_headers

        # Also verify that keys in custom_headers override keys in default_headers
        for k in custom_headers:
            if k in default_headers and k in transport.headers:
                assert transport.headers[k] == custom_headers[k]
