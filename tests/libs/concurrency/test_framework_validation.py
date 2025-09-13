"""Simple test to validate our testing framework setup."""

import pytest


def test_basic_framework():
    """Basic test to validate pytest setup."""
    assert True


@pytest.mark.anyio
async def test_anyio_basic():
    """Test basic anyio functionality."""
    import anyio

    await anyio.sleep(0.001)
    assert True
