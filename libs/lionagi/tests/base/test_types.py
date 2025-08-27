"""Test suite for types.py (Branch) - TDD Specification Implementation.

Focus: State isolation, capability management, and lineage tracking.
"""

from uuid import uuid4

from lionagi.base.types import Branch, create_branch

# Mock capabilities for testing
MOCK_CAPABILITIES = {
    "fs.read:/data/input.txt",
    "fs.write:/data/output.txt",
    "net.out:api.service.com",
}


class TestBranchContextAndIsolation:
    """TestSuite: BranchContextAndIsolation - State isolation, capability management, and lineage tracking."""

    def test_forking_state_isolation(self):
        """Test: ForkingStateIsolation (CRITICAL)

        GIVEN a Branch 'Parent' with ctx={"a": 1}
        WHEN 'Child' is forked from 'Parent'
        AND 'Child.ctx' is modified (e.g., Child.ctx["a"] = 2, Child.ctx["b"] = 10)
        THEN 'Parent.ctx["a"]' must still be 1
        AND 'b' must not exist in Parent.ctx.
        """
        # Create parent branch with initial context
        parent = Branch(id=uuid4(), ctx={"a": 1})

        # Fork child from parent
        child = parent.fork()

        # Modify child context
        child.ctx["a"] = 2
        child.ctx["b"] = 10

        # Verify parent context is unchanged
        assert (
            parent.ctx["a"] == 1
        ), "Parent context 'a' must remain unchanged after child modification"
        assert "b" not in parent.ctx, "Parent context must not contain keys added to child"

        # Verify child has expected modifications
        assert child.ctx["a"] == 2, "Child context 'a' should be modified"
        assert child.ctx["b"] == 10, "Child context should contain new key 'b'"

        # Test deep modification isolation
        parent.ctx["nested"] = {"x": 1, "y": [1, 2]}
        child2 = parent.fork()

        # Modify nested structures in child
        child2.ctx["nested"]["x"] = 999
        child2.ctx["nested"]["y"].append(3)

        # Verify parent nested structures are unchanged
        assert parent.ctx["nested"]["x"] == 1, "Parent nested dict must remain unchanged"
        assert parent.ctx["nested"]["y"] == [1, 2], "Parent nested list must remain unchanged"

    def test_forking_capability_inheritance_and_isolation(self):
        """Test: ForkingCapabilityInheritanceAndIsolation

        GIVEN a Branch 'Parent' with capabilities={"fs.read"}
        WHEN 'Child' is forked
        THEN Child inherits "fs.read".
        WHEN Child capabilities are modified (e.g., reduced or augmented)
        THEN Parent capabilities must remain unchanged.
        """
        # Create parent with capabilities
        parent_caps = {"fs.read:/data/input.txt", "net.out:api.service.com"}
        parent = create_branch(id=uuid4(), capabilities=parent_caps.copy())

        # Fork child from parent
        child = parent.fork()

        # Verify child inherits all parent capabilities
        assert child.capabilities == parent_caps, "Child must inherit all parent capabilities"

        # Modify child capabilities - add new capability
        child.capabilities.add("fs.write:/data/output.txt")

        # Verify parent capabilities unchanged
        assert (
            parent.capabilities == parent_caps
        ), "Parent capabilities must remain unchanged after child augmentation"
        assert (
            "fs.write:/data/output.txt" not in parent.capabilities
        ), "Parent must not gain child's new capabilities"

        # Modify child capabilities - remove capability
        child.capabilities.remove("fs.read:/data/input.txt")

        # Verify parent still has the removed capability
        assert (
            "fs.read:/data/input.txt" in parent.capabilities
        ), "Parent must retain capability removed from child"
        assert (
            "fs.read:/data/input.txt" not in child.capabilities
        ), "Child should not have removed capability"

    def test_lineage_tracking(self):
        """Test: LineageTracking

        GIVEN Parent -> ChildA -> Grandchild
        WHEN inspecting Grandchild.parent and lineage
        THEN it should reflect the correct ancestral path (Parent, ChildA).
        """
        # Create lineage chain: Parent -> ChildA -> Grandchild
        parent = Branch(id=uuid4(), ctx={"generation": "parent"})
        child_a = parent.fork()
        child_a.ctx["generation"] = "child_a"
        grandchild = child_a.fork()
        grandchild.ctx["generation"] = "grandchild"

        # Verify direct parent relationship
        assert grandchild.parent is child_a, "Grandchild's parent must be ChildA"
        assert child_a.parent is parent, "ChildA's parent must be Parent"
        assert parent.parent is None, "Parent should have no parent"

        # Test lineage traversal
        lineage = []
        current = grandchild
        while current is not None:
            lineage.append(current.ctx.get("generation", "unknown"))
            current = current.parent

        expected_lineage = ["grandchild", "child_a", "parent"]
        assert lineage == expected_lineage, f"Lineage should be {expected_lineage}, got {lineage}"

        # Verify each branch maintains its own identity
        assert grandchild.id != child_a.id != parent.id, "Each branch must have unique identity"

    def test_forked_context_deep_copy_behavior(self):
        """Test that forked context changes don't affect parent through deep references."""
        # Create parent with complex nested structures
        parent = Branch(
            id=uuid4(),
            ctx={
                "config": {
                    "settings": {"timeout": 30, "retries": 3},
                    "features": ["auth", "cache"],
                },
                "data": [{"id": 1, "values": [10, 20]}],
            },
        )

        # Fork child
        child = parent.fork()

        # Modify nested structures in child
        child.ctx["config"]["settings"]["timeout"] = 60
        child.ctx["config"]["features"].append("logging")
        child.ctx["data"][0]["values"].append(30)
        child.ctx["data"].append({"id": 2, "values": [40, 50]})

        # Verify parent structures are completely unchanged
        assert parent.ctx["config"]["settings"]["timeout"] == 30
        assert parent.ctx["config"]["features"] == ["auth", "cache"]
        assert parent.ctx["data"][0]["values"] == [10, 20]
        assert len(parent.ctx["data"]) == 1

        # Verify child has expected changes
        assert child.ctx["config"]["settings"]["timeout"] == 60
        assert "logging" in child.ctx["config"]["features"]
        assert 30 in child.ctx["data"][0]["values"]
        assert len(child.ctx["data"]) == 2

    def test_capability_set_isolation(self):
        """Test that capability sets are properly isolated between parent and child."""
        # Create parent with mutable capability set
        parent_caps = {"fs.read:/data/*", "net.out:*.api.com"}
        parent = create_branch(id=uuid4(), capabilities=parent_caps)

        # Fork multiple children
        child1 = parent.fork()
        child2 = parent.fork()

        # Modify capabilities independently
        child1.capabilities.add("db.query:users")
        child1.capabilities.discard("net.out:*.api.com")

        child2.capabilities.add("fs.write:/logs/*")

        # Verify complete isolation
        assert parent.capabilities == parent_caps, "Parent capabilities must be unchanged"
        assert "db.query:users" in child1.capabilities, "Child1 should have added capability"
        assert (
            "net.out:*.api.com" not in child1.capabilities
        ), "Child1 should have removed capability"
        assert (
            "db.query:users" not in child2.capabilities
        ), "Child2 should not have child1's capability"
        assert "fs.write:/logs/*" in child2.capabilities, "Child2 should have its added capability"
        assert (
            "fs.write:/logs/*" not in child1.capabilities
        ), "Child1 should not have child2's capability"

    def test_empty_branch_initialization(self):
        """Test initialization of branches with empty or None contexts and capabilities."""
        # Test empty initialization
        empty_branch = Branch(id=uuid4())
        assert empty_branch.ctx == {}, "Empty branch should have empty context dict"
        assert empty_branch.capabilities == set(), "Empty branch should have empty capability set"
        assert empty_branch.parent is None, "Empty branch should have no parent"

        # Test None initialization
        none_branch = create_branch(id=uuid4(), ctx=None, capabilities=None)
        assert none_branch.ctx == {}, "None context should default to empty dict"
        assert none_branch.capabilities == set(), "None capabilities should default to empty set"

        # Test forking from empty branch
        child = empty_branch.fork()
        assert child.ctx == {}, "Child of empty branch should have empty context"
        assert child.capabilities == set(), "Child of empty branch should have empty capabilities"
        assert child.parent is empty_branch, "Child should reference empty parent"

    def test_branch_identity_uniqueness(self):
        """Test that each branch has unique identity even with same content."""
        # Create branches with identical content
        ctx = {"data": "test"}
        caps = {"fs.read:/test"}

        branch1 = create_branch(id=uuid4(), ctx=ctx.copy(), capabilities=caps.copy())
        branch2 = create_branch(id=uuid4(), ctx=ctx.copy(), capabilities=caps.copy())

        # Verify unique identities
        assert branch1.id != branch2.id, "Branches must have unique IDs"
        assert branch1 is not branch2, "Branches must be distinct objects"

        # Verify content equality but object distinction
        assert branch1.ctx == branch2.ctx, "Context content should be equal"
        assert branch1.capabilities == branch2.capabilities, "Capabilities should be equal"

        # Verify forked branches have unique identities
        child1 = branch1.fork()
        child2 = branch1.fork()

        assert child1.id != child2.id, "Forked children must have unique IDs"
        assert child1.id != branch1.id, "Child ID must differ from parent"
        assert child1 is not child2, "Forked children must be distinct objects"
