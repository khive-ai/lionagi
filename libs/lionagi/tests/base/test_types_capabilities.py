"""Test suite for base/types.py - CapabilitySet mutation and policy integration.

Focus: CapabilitySet._update_branch() must update br.caps so policy_check
observes capability changes.
"""

from uuid import uuid4

import pytest

from lionagi.base.policy import policy_check
from lionagi.base.types import Branch, Capability, CapabilitySet, create_branch


class MockMorphism:
    """Mock morphism for policy checking tests."""

    def __init__(self, requires: set[str]):
        self.requires = requires


class TestCapabilitySetMutation:
    """Test CapabilitySet updates are visible to policy checks."""

    def test_capability_add_updates_branch_caps(self):
        """Test that adding capabilities via CapabilitySet updates branch.caps."""
        # Create branch without the required capability
        branch = create_branch(id=uuid4(), capabilities=set())
        morphism = MockMorphism(requires={"net.out:api.com"})

        # Initially should fail policy check
        assert policy_check(branch, morphism) is False, "Should fail without required capability"

        # Add capability via CapabilitySet
        branch.capabilities.add("net.out:api.com")

        # Now should pass policy check (tests _update_branch() effectiveness)
        assert (
            policy_check(branch, morphism) is True
        ), "Should pass after adding required capability"

        # Verify capability is in the live view
        assert "net.out:api.com" in branch.capabilities, "Capability should be in live set"

    def test_capability_remove_updates_branch_caps(self):
        """Test that removing capabilities via CapabilitySet updates branch.caps."""
        # Create branch with capability
        initial_caps = {"fs.read:/data/*", "net.out:api.com"}
        branch = create_branch(id=uuid4(), capabilities=initial_caps)

        morphism_fs = MockMorphism(requires={"fs.read:/data/*"})
        morphism_net = MockMorphism(requires={"net.out:api.com"})

        # Initially both should pass
        assert policy_check(branch, morphism_fs) is True
        assert policy_check(branch, morphism_net) is True

        # Remove one capability
        branch.capabilities.remove("fs.read:/data/*")

        # Verify policy checks reflect the change
        assert (
            policy_check(branch, morphism_fs) is False
        ), "Should fail after removing fs capability"
        assert (
            policy_check(branch, morphism_net) is True
        ), "Should still pass for remaining capability"

        # Verify live capability set was updated
        assert (
            "fs.read:/data/*" not in branch.capabilities
        ), "Removed capability should not be in live set"
        assert (
            "net.out:api.com" in branch.capabilities
        ), "Remaining capability should be in live set"

    def test_capability_discard_updates_branch_caps(self):
        """Test that discarding capabilities (no error if missing) updates branch.caps."""
        # Create branch with one capability
        branch = create_branch(id=uuid4(), capabilities={"existing.perm"})
        morphism = MockMorphism(requires={"existing.perm"})

        # Should initially pass
        assert policy_check(branch, morphism) is True

        # Discard existing capability
        branch.capabilities.discard("existing.perm")

        # Should now fail
        assert policy_check(branch, morphism) is False, "Should fail after discarding capability"

        # Discard non-existent capability (should not error)
        branch.capabilities.discard("non.existent")

        # Should still fail (no change)
        assert policy_check(branch, morphism) is False

    def test_capability_update_operations_update_branch_caps(self):
        """Test set update operations (union, intersection, etc.) update branch.caps."""
        # Create branch with initial capabilities
        initial_caps = {"read.data", "write.logs"}
        branch = create_branch(id=uuid4(), capabilities=initial_caps)

        # Test union update
        new_caps = {"net.access", "db.query"}
        branch.capabilities.update(new_caps)

        # Verify all capabilities are accessible via policy check
        for cap in initial_caps | new_caps:
            morphism = MockMorphism(requires={cap})
            assert policy_check(branch, morphism) is True, f"Should have capability {cap}"

        # Verify live capability set contains all rights
        assert branch.capabilities == initial_caps | new_caps

    def test_capability_clear_updates_branch_caps(self):
        """Test that clearing all capabilities updates branch.caps."""
        # Create branch with multiple capabilities
        initial_caps = {"perm1", "perm2", "perm3"}
        branch = create_branch(id=uuid4(), capabilities=initial_caps)

        # Verify initially have capabilities
        for cap in initial_caps:
            morphism = MockMorphism(requires={cap})
            assert policy_check(branch, morphism) is True

        # Clear all capabilities
        branch.capabilities.clear()

        # Verify all policy checks now fail
        for cap in initial_caps:
            morphism = MockMorphism(requires={cap})
            assert (
                policy_check(branch, morphism) is False
            ), f"Should not have capability {cap} after clear"

        # Verify live capability set reflects empty state
        assert len(branch.capabilities) == 0

    def test_multiple_capability_mutations_maintain_consistency(self):
        """Test multiple capability mutations maintain consistent branch.caps state."""
        branch = create_branch(id=uuid4(), capabilities=set())

        # Add multiple capabilities in sequence
        capabilities_to_add = [
            "fs.read:/path1",
            "fs.write:/path2",
            "net.out:host1.com",
            "db.query:table1",
            "cache.set:region1",
        ]

        for cap in capabilities_to_add:
            branch.capabilities.add(cap)

            # After each addition, verify policy check works
            morphism = MockMorphism(requires={cap})
            assert policy_check(branch, morphism) is True, f"Should have {cap} after adding"

            # Verify live capability set is consistent
            assert cap in branch.capabilities

        # Remove capabilities in different order
        removal_order = ["db.query:table1", "fs.read:/path1", "cache.set:region1"]

        for cap in removal_order:
            branch.capabilities.remove(cap)

            # After each removal, verify policy check fails for removed cap
            morphism = MockMorphism(requires={cap})
            assert policy_check(branch, morphism) is False, f"Should not have {cap} after removing"

            # Verify remaining capabilities still work
            remaining = set(capabilities_to_add) - set(
                removal_order[: removal_order.index(cap) + 1]
            )
            for remaining_cap in remaining:
                morphism = MockMorphism(requires={remaining_cap})
                assert policy_check(branch, morphism) is True, f"Should still have {remaining_cap}"

    def test_capability_set_isolation_between_branches(self):
        """Test that capability mutations don't affect other branches."""
        # Create two branches with same initial capabilities
        initial_caps = {"shared.perm1", "shared.perm2"}
        branch1 = create_branch(id=uuid4(), capabilities=initial_caps.copy())
        branch2 = create_branch(id=uuid4(), capabilities=initial_caps.copy())

        # Modify branch1 capabilities
        branch1.capabilities.add("branch1.exclusive")
        branch1.capabilities.remove("shared.perm1")

        # Verify branch1 changes
        morphism_exclusive = MockMorphism(requires={"branch1.exclusive"})
        morphism_shared1 = MockMorphism(requires={"shared.perm1"})
        morphism_shared2 = MockMorphism(requires={"shared.perm2"})

        assert policy_check(branch1, morphism_exclusive) is True
        assert policy_check(branch1, morphism_shared1) is False
        assert policy_check(branch1, morphism_shared2) is True

        # Verify branch2 is unchanged
        assert policy_check(branch2, morphism_exclusive) is False
        assert policy_check(branch2, morphism_shared1) is True
        assert policy_check(branch2, morphism_shared2) is True

    def test_forked_branch_capability_mutations_isolated(self):
        """Test that forked branches have isolated capability mutations."""
        # Create parent branch
        parent = create_branch(id=uuid4(), capabilities={"parent.perm"})

        # Fork child
        child = parent.fork()

        # Verify child inherits parent capabilities
        morphism = MockMorphism(requires={"parent.perm"})
        assert policy_check(child, morphism) is True

        # Modify child capabilities
        child.capabilities.add("child.exclusive")
        child.capabilities.remove("parent.perm")

        # Verify parent unchanged
        morphism_parent = MockMorphism(requires={"parent.perm"})
        morphism_child = MockMorphism(requires={"child.exclusive"})

        assert policy_check(parent, morphism_parent) is True, "Parent should retain its capability"
        assert (
            policy_check(parent, morphism_child) is False
        ), "Parent should not have child's capability"

        assert (
            policy_check(child, morphism_parent) is False
        ), "Child should not have removed capability"
        assert policy_check(child, morphism_child) is True, "Child should have its new capability"
