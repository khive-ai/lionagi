"""Test suite for policy.py (Capabilities) - TDD Specification Implementation.

Focus: Adversarial testing of permission logic.
"""

import pytest

from lionagi.base.morphism import Morphism
from lionagi.base.policy import policy_check
from lionagi.base.types import create_branch


class MockMorphism(Morphism):
    """Mock morphism for policy testing."""

    def __init__(self, name: str = "test", requires: set[str] | None = None):
        self.name = name
        self.requires = requires or set()

    async def pre(self, br, **kwargs) -> bool:
        return True

    async def apply(self, br, **kwargs) -> dict:
        return {"result": "test"}

    async def post(self, br, result, **kwargs) -> bool:
        return True


class TestCapabilityEnforcement:
    """TestSuite: CapabilityEnforcement - Adversarial testing of permission logic."""

    def test_exact_match(self):
        """Test: ExactMatch

        GIVEN Branch with capabilities = {"fs.read:/data/input.txt"}
        WHEN checking Required = {"fs.read:/data/input.txt"}
        THEN policy_check should succeed.
        """
        branch = create_branch(capabilities={"fs.read:/data/input.txt"})
        morphism = MockMorphism(requires={"fs.read:/data/input.txt"})

        assert policy_check(branch, morphism), "Exact capability match must succeed"

    def test_insufficient_privilege(self):
        """Test: InsufficientPrivilege

        GIVEN Branch with capabilities = {"fs.read:/data/input.txt"}
        WHEN checking Required = {"fs.write:/data/input.txt"}
        THEN policy_check should fail.
        """
        branch = create_branch(capabilities={"fs.read:/data/input.txt"})

        # Test different action
        morphism1 = MockMorphism(requires={"fs.write:/data/input.txt"})
        assert not policy_check(branch, morphism1), "Insufficient privilege must fail validation"

        # Test different resource with same action
        morphism2 = MockMorphism(requires={"fs.read:/other/path.txt"})
        assert not policy_check(branch, morphism2), "Different path with same action must fail"

        # Test escalated action on same resource
        morphism3 = MockMorphism(requires={"fs.execute:/data/input.txt"})
        assert not policy_check(branch, morphism3), "Action escalation must fail"

    def test_wildcard_grant(self):
        """Test: WildcardGrant

        GIVEN Branch with capabilities = {"net.out:*"}
        WHEN checking Required = {"net.out:api.service.com"}
        THEN policy_check should succeed.
        """
        branch = create_branch(capabilities={"net.out:*"})

        # Test various wildcard matches
        test_cases = [
            {"net.out:api.service.com"},
            {"net.out:internal.company.net"},
            {"net.out:public.api.endpoint"},
            {"net.out:127.0.0.1:8080"},
        ]

        for required in test_cases:
            morphism = MockMorphism(requires=required)
            assert policy_check(branch, morphism), f"Wildcard should match {required}"

        # Test wildcard doesn't match different actions
        invalid_cases = [
            {"net.in:api.service.com"},  # Different action
            {"fs.read:api.service.com"},  # Different domain
            {"db.query:api.service.com"},  # Completely different
        ]

        for required in invalid_cases:
            morphism = MockMorphism(requires=required)
            assert not policy_check(branch, morphism), f"Wildcard should NOT match {required}"

    def test_scope_mismatch(self):
        """Test: ScopeMismatch

        GIVEN Branch with capabilities = {"fs.read:/data/*"}
        WHEN checking Required = {"fs.read:/other_dir/file.txt"}
        THEN policy_check should fail.
        """
        branch = create_branch(capabilities={"fs.read:/data/*"})

        # Test scope mismatches
        failing_cases = [
            {"fs.read:/other_dir/file.txt"},  # Different directory
            {"fs.read:/data_backup/file.txt"},  # Similar but different path
            {"fs.read:/file.txt"},  # Parent directory
            {"fs.read:/data/../other/file.txt"},  # Path traversal attempt
        ]

        for required in failing_cases:
            morphism = MockMorphism(requires=required)
            assert not policy_check(branch, morphism), f"Scope mismatch should fail for {required}"

        # Test valid matches within scope
        valid_cases = [
            {"fs.read:/data/file.txt"},
            {"fs.read:/data/subdir/file.txt"},
            {"fs.read:/data/deep/nested/path.txt"},
        ]

        for required in valid_cases:
            morphism = MockMorphism(requires=required)
            assert policy_check(branch, morphism), f"Valid scope should succeed for {required}"

    def test_adversarial_path_traversal(self):
        """Test: AdversarialPathTraversal (Security)

        GIVEN Branch with capabilities = {"fs.read:/sandbox/*"}
        # The policy layer checks normalized paths
        WHEN checking Required = "fs.read:/etc/passwd" (derived from input "/sandbox/../../etc/passwd")
        THEN policy_check should fail.
        """
        branch = create_branch(capabilities={"fs.read:/sandbox/*"})

        # Direct path traversal attempts
        traversal_attempts = [
            {"fs.read:/etc/passwd"},  # Direct access
            {"fs.read:/sandbox/../etc/passwd"},  # Single level up
            {"fs.read:/sandbox/../../etc/passwd"},  # Double level up
            {"fs.read:/sandbox/../../../root/.ssh"},  # Multiple levels up
            {"fs.read:/sandbox/./../../etc/shadow"},  # With current dir reference
            {"fs.read:/sandbox/subdir/../../etc/hosts"},  # From subdirectory
        ]

        for required in traversal_attempts:
            # The policy should normalize paths and detect traversal
            morphism = MockMorphism(requires=required)
            assert not policy_check(branch, morphism), f"Path traversal should fail for {required}"

        # Valid paths within sandbox
        valid_paths = [
            {"fs.read:/sandbox/file.txt"},
            {"fs.read:/sandbox/subdir/file.txt"},
            {"fs.read:/sandbox/deep/nested/file.txt"},
        ]

        for required in valid_paths:
            morphism = MockMorphism(requires=required)
            assert policy_check(
                branch, morphism
            ), f"Valid sandbox path should succeed for {required}"

    def test_multiple_capability_validation(self):
        """Test validation when multiple capabilities are required."""
        branch = create_branch(
            capabilities={
                "fs.read:/data/*",
                "fs.write:/output/*",
                "net.out:api.service.com",
                "db.query:users",
            }
        )

        # Test multiple valid requirements
        valid_multi = {
            "fs.read:/data/input.txt",
            "fs.write:/output/result.txt",
            "net.out:api.service.com",
        }
        morphism_valid = MockMorphism(requires=valid_multi)
        assert policy_check(branch, morphism_valid), "All valid capabilities should pass"

        # Test with one invalid requirement
        invalid_multi = {
            "fs.read:/data/input.txt",  # Valid
            "fs.write:/output/result.txt",  # Valid
            "net.out:malicious.site.com",  # Invalid - not in capabilities
        }
        morphism_invalid = MockMorphism(requires=invalid_multi)
        assert not policy_check(
            branch, morphism_invalid
        ), "Any invalid capability should fail entire validation"

        # Test empty requirements
        morphism_empty = MockMorphism(requires=set())
        assert policy_check(branch, morphism_empty), "Empty requirements should always pass"

    def test_complex_wildcard_patterns(self):
        """Test complex wildcard patterns and edge cases."""
        branch = create_branch(
            capabilities={
                "net.out:*.api.com",  # Subdomain wildcard
                "fs.read:/data/*/logs/*",  # Multiple wildcards
                "db.query:table_*",  # Prefix wildcard
            }
        )

        # Test subdomain wildcard
        subdomain_cases = [
            ({"net.out:v1.api.com"}, True),
            ({"net.out:beta.api.com"}, True),
            ({"net.out:api.com"}, False),  # No subdomain
            ({"net.out:api.com.evil"}, False),  # Wrong domain
            ({"net.out:v1.api.co"}, False),  # Wrong TLD
        ]

        for required, should_pass in subdomain_cases:
            morphism = MockMorphism(requires=required)
            result = policy_check(branch, morphism)
            assert (
                result == should_pass
            ), f"Subdomain wildcard test failed for {required}: expected {should_pass}, got {result}"

        # Test multiple wildcards
        multi_wildcard_cases = [
            ({"fs.read:/data/app1/logs/error.log"}, True),
            ({"fs.read:/data/app2/logs/access.log"}, True),
            ({"fs.read:/data/logs/error.log"}, False),  # Missing middle segment
            ({"fs.read:/data/app1/config.txt"}, False),  # Wrong directory
            ({"fs.read:/logs/app1/logs/error.log"}, False),  # Wrong root
        ]

        for required, should_pass in multi_wildcard_cases:
            morphism = MockMorphism(requires=required)
            result = policy_check(branch, morphism)
            assert (
                result == should_pass
            ), f"Multi-wildcard test failed for {required}: expected {should_pass}, got {result}"

    def test_branch_capability_isolation(self):
        """Test that branch capabilities are properly isolated."""
        original_caps = {"fs.read:/data/*", "net.out:api.com"}
        branch = create_branch(capabilities=original_caps)

        # Test that original set isn't modified after branch creation
        original_caps.add("malicious.capability")

        # Verify branch capabilities don't include the malicious capability
        morphism = MockMorphism(requires={"malicious.capability"})
        assert not policy_check(
            branch, morphism
        ), "Branch should not gain capabilities from external modifications"

        # Verify branch still has original capabilities
        morphism_valid = MockMorphism(requires={"fs.read:/data/test.txt"})
        assert policy_check(branch, morphism_valid), "Branch should retain original capabilities"

    def test_empty_and_none_capabilities(self):
        """Test behavior with empty or None capabilities."""
        # Empty capabilities
        empty_branch = create_branch(capabilities=set())
        morphism_any = MockMorphism(requires={"any.capability"})
        assert not policy_check(
            empty_branch, morphism_any
        ), "Empty capabilities should deny all requests"

        morphism_empty = MockMorphism(requires=set())
        assert policy_check(
            empty_branch, morphism_empty
        ), "Empty capabilities should allow empty requirements"

        # None capabilities (should default to empty)
        none_branch = create_branch(capabilities=None)
        assert not policy_check(
            none_branch, morphism_any
        ), "None capabilities should deny all requests"
        assert policy_check(
            none_branch, morphism_empty
        ), "None capabilities should allow empty requirements"

    def test_capability_normalization(self):
        """Test that capabilities are properly normalized and validated."""
        # Test case sensitivity
        branch_lower = create_branch(capabilities={"fs.read:/data/file.txt"})
        morphism_upper = MockMorphism(requires={"fs.read:/Data/file.txt"})
        assert not policy_check(
            branch_lower, morphism_upper
        ), "Case sensitivity should be preserved"

        # Test whitespace handling
        branch_spaces = create_branch(capabilities={"fs.read:/data/file with spaces.txt"})
        morphism_spaces = MockMorphism(requires={"fs.read:/data/file with spaces.txt"})
        assert policy_check(branch_spaces, morphism_spaces), "Spaces in paths should be preserved"

    def test_security_boundary_enforcement(self):
        """Test enforcement of security boundaries and fail-closed behavior."""
        branch = create_branch(capabilities={"fs.read:/public/*"})

        # Test various security boundary violations
        security_violations = [
            {"fs.read:/private/sensitive.txt"},  # Different security zone
            {"fs.write:/public/file.txt"},  # Privilege escalation
            {"fs.execute:/public/script.sh"},  # Action escalation
            {"fs.read:/public/../private/data"},  # Path traversal
            {"admin.access:system"},  # Administrative access
            {"root.privilege:*"},  # Root access attempt
        ]

        for violation in security_violations:
            morphism = MockMorphism(requires=violation)
            assert not policy_check(
                branch, morphism
            ), f"Security violation should be denied: {violation}"
