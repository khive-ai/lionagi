"""Tests for policy using live capability rights view.

These tests verify that policy_check() honors runtime capability updates
made through the Branch.capabilities API, ensuring security decisions
use the current state rather than stale capability tuples.
"""

import pytest

from lionagi.base.policy import policy_check
from lionagi.base.types import create_branch


class DummyMorphism:
    """Test morphism that requires net.out capability."""

    name = "test_morphism"
    requires = {"net.out"}


@pytest.mark.anyio
async def test_policy_reads_live_capabilities(anyio_backend):
    """Test that policy_check uses live capability view from runtime updates."""
    # Create branch with initial capabilities
    br = create_branch(capabilities={"fs.read:/data/*"})
    m = DummyMorphism()

    # Initially, no net.out capability should deny access
    assert policy_check(br, m) is False

    # Grant capability at runtime via the supported API
    # Note: Keep a strong reference to prevent WeakValueDictionary cleanup
    capabilities = br.capabilities
    capabilities.add("net.out")

    # Policy should now see the runtime-added capability
    assert policy_check(br, m) is True


@pytest.mark.anyio
async def test_policy_respects_capability_removal(anyio_backend):
    """Test that policy_check respects capability removal at runtime."""
    # Create branch with initial capabilities including net.out
    br = create_branch(capabilities={"fs.read:/data/*", "net.out"})
    m = DummyMorphism()

    # Initially should have access
    assert policy_check(br, m) is True

    # Remove capability at runtime
    capabilities = br.capabilities
    capabilities.discard("net.out")

    # Policy should now deny access
    assert policy_check(br, m) is False


@pytest.mark.anyio
async def test_policy_with_multiple_capability_updates(anyio_backend):
    """Test policy with multiple runtime capability changes."""
    br = create_branch(capabilities=set())

    class MultiRequireMorphism:
        name = "multi_require"
        requires = {"fs.read:/tmp/*", "net.out", "db.write"}

    m = MultiRequireMorphism()

    # Initially no capabilities
    assert policy_check(br, m) is False

    # Add capabilities one by one
    capabilities = br.capabilities
    capabilities.add("fs.read:/tmp/*")
    assert policy_check(br, m) is False  # Still missing some

    capabilities.add("net.out")
    assert policy_check(br, m) is False  # Still missing db.write

    capabilities.add("db.write")
    assert policy_check(br, m) is True  # Now has all required

    # Remove one capability
    capabilities.remove("net.out")
    assert policy_check(br, m) is False  # Missing net.out again


@pytest.mark.anyio
async def test_policy_with_wildcard_capabilities(anyio_backend):
    """Test policy with wildcard pattern matching on live capabilities."""
    br = create_branch(capabilities=set())

    class WildcardMorphism:
        name = "wildcard_test"
        requires = {"fs.read:/data/users/123"}

    m = WildcardMorphism()

    # Add wildcard capability at runtime
    capabilities = br.capabilities
    capabilities.add("fs.read:/data/users/*")

    # Should match the specific required path
    assert policy_check(br, m) is True

    # Remove and add more general wildcard
    capabilities.clear()
    capabilities.add("fs.read:/data/*")

    # Should still match
    assert policy_check(br, m) is True
