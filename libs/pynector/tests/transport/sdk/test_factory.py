"""
Tests for the SdkTransportFactory class.
"""

from pynector.transport.sdk.factory import SdkTransportFactory


def test_sdk_transport_factory_init():
    """Test SdkTransportFactory initialization."""
    # Default initialization
    factory = SdkTransportFactory()
    assert factory.sdk_type == "openai"
    assert factory.api_key is None
    assert factory.base_url is None
    assert factory.timeout == 60.0
    assert factory.default_config == {}

    # Custom initialization
    factory = SdkTransportFactory(
        sdk_type="anthropic",
        api_key="test-key",
        base_url="https://example.com",
        timeout=30.0,
        model="claude-3-opus-20240229",
    )
    assert factory.sdk_type == "anthropic"
    assert factory.api_key == "test-key"
    assert factory.base_url == "https://example.com"
    assert factory.timeout == 30.0
    assert factory.default_config == {"model": "claude-3-opus-20240229"}


def test_sdk_transport_factory_create_transport():
    """Test SdkTransportFactory create_transport method."""
    # Create factory
    factory = SdkTransportFactory(
        sdk_type="openai", api_key="default-key", timeout=60.0, model="gpt-3.5-turbo"
    )

    # Create transport with default settings
    transport = factory.create_transport()

    # Verify transport was created with correct settings
    assert transport.sdk_type == "openai"
    assert transport.api_key == "default-key"
    assert transport.base_url is None
    assert transport.timeout == 60.0
    assert transport.config == {"model": "gpt-3.5-turbo"}

    # Create transport with custom settings
    transport = factory.create_transport(
        sdk_type="anthropic",
        api_key="custom-key",
        base_url="https://example.com",
        timeout=30.0,
        model="claude-3-opus-20240229",
    )

    # Verify transport was created with correct settings
    assert transport.sdk_type == "anthropic"
    assert transport.api_key == "custom-key"
    assert transport.base_url == "https://example.com"
    assert transport.timeout == 30.0
    assert transport.config == {"model": "claude-3-opus-20240229"}


def test_sdk_transport_factory_create_transport_merge_config():
    """Test SdkTransportFactory create_transport method with merged config."""
    # Create factory with default config
    factory = SdkTransportFactory(
        sdk_type="openai", model="gpt-3.5-turbo", temperature=0.7, max_tokens=100
    )

    # Create transport with partial override
    transport = factory.create_transport(model="gpt-4o", max_tokens=200)

    # Verify transport was created with merged config
    assert transport.sdk_type == "openai"
    assert transport.config["model"] == "gpt-4o"  # Overridden
    assert transport.config["temperature"] == 0.7  # From default
    assert transport.config["max_tokens"] == 200  # Overridden
