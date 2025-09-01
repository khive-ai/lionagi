"""Tests for Branch.fork() inheriting live capability rights.

These tests verify that when a branch is forked, the child inherits
the current live capability state (including runtime updates) rather
than just the original static capability tuples.
"""

import pytest

from lionagi.base.policy import policy_check
from lionagi.base.types import create_branch


class NeedsX:
    """Test morphism that requires 'x' capability."""

    name = "needs_x"
    requires = {"x"}


class NeedsMultiple:
    """Test morphism that requires multiple capabilities."""

    name = "needs_multiple"
    requires = {"a", "b", "c"}


@pytest.mark.anyio
async def test_fork_inherits_live_rights(anyio_backend):
    """Test that fork inherits current live rights, not original caps."""
    # Create parent with no initial capabilities
    parent = create_branch(capabilities=set())

    # Add capability at runtime
    parent_capabilities = parent.capabilities
    parent_capabilities.add("x")

    # Fork should inherit the runtime-added capability
    child = parent.fork()

    # Both parent and child should satisfy the requirement
    assert policy_check(parent, NeedsX())
    assert policy_check(child, NeedsX())


@pytest.mark.anyio
async def test_fork_inheritance_isolation(anyio_backend):
    """Test that fork creates isolated capability inheritance."""
    parent = create_branch(capabilities={"initial"})

    # Add runtime capability to parent
    parent_capabilities = parent.capabilities
    parent_capabilities.add("runtime_added")

    # Fork inherits current state
    child = parent.fork()

    # Verify inheritance
    assert "initial" in child.capabilities  # Note: this accesses child's live view
    assert "runtime_added" in child.capabilities

    # Modify child capabilities - should not affect parent
    child_capabilities = child.capabilities
    child_capabilities.add("child_only")

    assert "child_only" in child_capabilities
    assert "child_only" not in parent_capabilities

    # Modify parent capabilities - should not affect child
    parent_capabilities.add("parent_only")

    assert "parent_only" in parent_capabilities
    assert "parent_only" not in child_capabilities


@pytest.mark.anyio
async def test_fork_with_multiple_runtime_changes(anyio_backend):
    """Test fork inheritance with multiple runtime capability changes."""
    parent = create_branch(capabilities={"base"})

    # Make several runtime changes
    parent_capabilities = parent.capabilities
    parent_capabilities.add("a")
    parent_capabilities.add("b")
    parent_capabilities.add("c")
    parent_capabilities.discard("base")  # Remove original

    # Fork should inherit current live state
    child = parent.fork()

    # Verify child got the current state, not original
    child_capabilities = child.capabilities
    assert "base" not in child_capabilities  # Was removed
    assert "a" in child_capabilities
    assert "b" in child_capabilities
    assert "c" in child_capabilities

    # Child should satisfy multi-requirement morphism
    assert policy_check(child, NeedsMultiple())


@pytest.mark.anyio
async def test_deep_fork_chain_inheritance(anyio_backend):
    """Test inheritance through multiple levels of forking."""
    # Create grandparent
    grandparent = create_branch(capabilities={"level0"})

    # Add capability and fork to create parent
    grandparent_capabilities = grandparent.capabilities
    grandparent_capabilities.add("level1")
    parent = grandparent.fork()

    # Add capability to parent and fork to create child
    parent_capabilities = parent.capabilities
    parent_capabilities.add("level2")
    child = parent.fork()

    # Child should have capabilities from all levels
    # Note: fork() creates new caps with combined rights, so we check policy
    class NeedsAllLevels:
        name = "needs_all"
        requires = {"level0", "level1", "level2"}

    # Child should have inherited all capabilities through the chain
    # This tests that fork() properly captures the live rights view
    expected_rights = {"level0", "level1", "level2"}
    child_rights = set()
    for cap in child.caps:
        child_rights.update(cap.rights)

    assert child_rights == expected_rights


@pytest.mark.anyio
async def test_fork_empty_to_populated(anyio_backend):
    """Test forking after going from empty to populated capabilities."""
    parent = create_branch(capabilities=set())  # Start empty

    # Verify initially empty
    parent_capabilities = parent.capabilities
    assert len(parent_capabilities) == 0

    # Add capabilities at runtime
    parent_capabilities.add("first")
    parent_capabilities.add("second")

    # Fork should inherit the populated state
    child = parent.fork()

    # Verify child has the added capabilities
    child_rights = set()
    for cap in child.caps:
        child_rights.update(cap.rights)

    assert "first" in child_rights
    assert "second" in child_rights
