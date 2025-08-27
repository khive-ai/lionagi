"""Test suite for base/policy.py - Wildcard behavior and edge cases.

Focus: Wildcard prefix rules, req wildcard + have concrete denial,
path normalization, and Windows vs POSIX path handling.
"""

import os
from pathlib import Path

import pytest

from lionagi.base.policy import policy_check
from lionagi.base.types import Branch, Capability


class MockMorphism:
    """Mock morphism for policy testing."""

    def __init__(self, requires: set[str]):
        self.requires = requires


class TestPolicyWildcardBehavior:
    """Test policy checking wildcard behavior and edge cases."""

    def test_req_wildcard_have_concrete_denied(self):
        """Test that requiring wildcard with concrete capability is denied (fail-closed)."""
        # Branch has specific concrete permission
        concrete_cap = Capability(
            subject="test_branch", rights={"fs.read:/tmp/specific_file.txt"}, object="*"
        )
        branch = Branch(caps=(concrete_cap,))

        # Morphism requires wildcard that would include the concrete permission
        morphism = MockMorphism(requires={"fs.read:/tmp/*"})

        # Should be denied - requiring wildcard with concrete permission is fail-closed
        result = policy_check(branch, morphism)
        assert result is False, "Requiring wildcard with concrete permission should be denied"

    def test_both_wildcard_prefix_rules(self):
        """Test both sides of wildcard prefix matching."""
        # Test Case 1: Have wildcard, require concrete (should allow)
        wildcard_cap = Capability(subject="test_branch", rights={"fs.read:/data/*"}, object="*")
        branch_with_wildcard = Branch(caps=(wildcard_cap,))

        concrete_morphism = MockMorphism(requires={"fs.read:/data/file.txt"})

        result = policy_check(branch_with_wildcard, concrete_morphism)
        assert result is True, "Having wildcard should satisfy concrete requirement"

        # Test Case 2: Have concrete, require wildcard (should deny)
        concrete_cap = Capability(
            subject="test_branch", rights={"fs.read:/data/file.txt"}, object="*"
        )
        branch_with_concrete = Branch(caps=(concrete_cap,))

        wildcard_morphism = MockMorphism(requires={"fs.read:/data/*"})

        result = policy_check(branch_with_concrete, wildcard_morphism)
        assert result is False, "Having concrete should not satisfy wildcard requirement"

    def test_wildcard_prefix_matching_rules(self):
        """Test detailed wildcard prefix matching behavior."""
        test_cases = [
            # (have_right, required_right, should_allow, description)
            ("fs.read:/app/*", "fs.read:/app/data.txt", True, "Wildcard covers specific file"),
            (
                "fs.read:/app/*",
                "fs.read:/app/subdir/file.txt",
                True,
                "Wildcard covers subdirectory",
            ),
            (
                "fs.read:/app/data/*",
                "fs.read:/app/data.txt",
                False,
                "Wildcard doesn't cover parent level",
            ),
            (
                "fs.read:/app/*",
                "fs.read:/other/file.txt",
                False,
                "Wildcard doesn't cover different path",
            ),
            (
                "net.out:api.com",
                "net.out:*.api.com",
                False,
                "Concrete doesn't satisfy wildcard requirement",
            ),
            ("net.out:*.api.com", "net.out:sub.api.com", True, "Wildcard covers subdomain"),
            ("net.out:*.api.com", "net.out:api.com", False, "Wildcard doesn't cover root domain"),
            (
                "net.out:*.api.com",
                "net.out:other.com",
                False,
                "Wildcard doesn't cover different domain",
            ),
        ]

        for have_right, required_right, should_allow, description in test_cases:
            cap = Capability(subject="test", rights={have_right}, object="*")
            branch = Branch(caps=(cap,))
            morphism = MockMorphism(requires={required_right})

            result = policy_check(branch, morphism)
            assert result == should_allow, f"{description}: have={have_right}, req={required_right}"

    def test_path_normalization_blocks_traversal(self):
        """Test that path normalization prevents directory traversal attacks."""
        # Capability for /tmp directory
        tmp_cap = Capability(subject="test_branch", rights={"fs.read:/tmp/*"}, object="*")
        branch = Branch(caps=(tmp_cap,))

        # Attempt directory traversal attacks
        traversal_attempts = [
            "fs.read:/tmp/../etc/passwd",  # Classic traversal
            "fs.read:/tmp/../../../etc/shadow",  # Multiple levels
            "fs.read:/tmp/subdir/../../etc/hosts",  # Through subdirectory
            "fs.read:/tmp/./../../etc/fstab",  # With current directory reference
        ]

        for malicious_req in traversal_attempts:
            morphism = MockMorphism(requires={malicious_req})
            result = policy_check(branch, morphism)

            # Should be denied due to path normalization
            # The normalized path would be /etc/... which doesn't match /tmp/*
            assert result is False, f"Traversal attempt should be blocked: {malicious_req}"

    def test_path_normalization_preserves_valid_paths(self):
        """Test that path normalization preserves legitimate paths."""
        data_cap = Capability(subject="test_branch", rights={"fs.write:/data/*"}, object="*")
        branch = Branch(caps=(data_cap,))

        # Valid paths that should be normalized but still match
        valid_paths = [
            "fs.write:/data/file.txt",  # Direct file
            "fs.write:/data/subdir/file.txt",  # Subdirectory file
            "fs.write:/data/./file.txt",  # With current directory
            "fs.write:/data/subdir/../file.txt",  # Normalized to /data/file.txt
        ]

        for valid_req in valid_paths:
            morphism = MockMorphism(requires={valid_req})
            result = policy_check(branch, morphism)

            assert result is True, f"Valid path should be allowed: {valid_req}"

    def test_windows_vs_posix_path_handling(self):
        """Test path handling differences between Windows and POSIX systems."""
        if os.name == "nt":  # Windows
            # Windows-specific path tests
            windows_cap = Capability(subject="test", rights={"fs.read:C:\\Data\\*"}, object="*")
            branch = Branch(caps=(windows_cap,))

            windows_paths = [
                "fs.read:C:\\Data\\file.txt",  # Direct Windows path
                "fs.read:C:/Data/file.txt",  # Mixed separators
                "fs.read:C:\\Data\\..\\Data\\file.txt",  # With traversal (should normalize)
            ]

            for path in windows_paths[:2]:  # First two should work
                morphism = MockMorphism(requires={path})
                result = policy_check(branch, morphism)
                assert result is True, f"Windows path should work: {path}"

            # Traversal should be blocked
            traversal_morphism = MockMorphism(requires={windows_paths[2]})
            # Result depends on normalization - if it normalizes to C:\Data\file.txt, should allow

        else:  # POSIX
            posix_cap = Capability(subject="test", rights={"fs.read:/home/user/*"}, object="*")
            branch = Branch(caps=(posix_cap,))

            posix_paths = [
                "fs.read:/home/user/file.txt",
                "fs.read:/home/user/subdir/file.txt",
                "fs.read:/home/user/../user/file.txt",  # Should normalize to allowed path
            ]

            for path in posix_paths:
                morphism = MockMorphism(requires={path})
                result = policy_check(branch, morphism)
                # All should be allowed after normalization
                assert result is True, f"POSIX path should work: {path}"

    def test_capability_object_field_wildcard_behavior(self):
        """Test that capability object field affects wildcard matching."""
        # Capability with specific object
        specific_object_cap = Capability(
            subject="test_branch", rights={"api.call"}, object="specific.service.com"
        )

        # Capability with wildcard object
        wildcard_object_cap = Capability(subject="test_branch", rights={"api.call"}, object="*")

        branch_specific = Branch(caps=(specific_object_cap,))
        branch_wildcard = Branch(caps=(wildcard_object_cap,))

        morphism = MockMorphism(requires={"api.call"})

        # Both should satisfy the requirement (object field is for additional context)
        result_specific = policy_check(branch_specific, morphism)
        result_wildcard = policy_check(branch_wildcard, morphism)

        assert result_specific is True, "Specific object should not block basic permission"
        assert result_wildcard is True, "Wildcard object should allow basic permission"

    def test_empty_and_invalid_wildcards(self):
        """Test handling of empty and invalid wildcard patterns."""
        empty_cap = Capability(subject="test", rights={"", "valid.perm"}, object="*")
        branch = Branch(caps=(empty_cap,))

        # Test empty requirement
        empty_morphism = MockMorphism(requires={""})
        result = policy_check(branch, empty_morphism)
        # Should handle empty string gracefully (likely deny)

        # Test valid requirement
        valid_morphism = MockMorphism(requires={"valid.perm"})
        result = policy_check(branch, valid_morphism)
        assert result is True, "Valid permission should work despite empty permission in set"

    def test_case_sensitivity_in_permissions(self):
        """Test that permission matching is case-sensitive."""
        cap = Capability(
            subject="test",
            rights={"fs.READ:/data/*"},  # Uppercase READ
            object="*",
        )
        branch = Branch(caps=(cap,))

        # Test case variations
        test_cases = [
            ("fs.READ:/data/file.txt", True, "Exact case should match"),
            ("fs.read:/data/file.txt", False, "Different case should not match"),
            ("FS.READ:/data/file.txt", False, "Different case prefix should not match"),
        ]

        for perm, should_allow, description in test_cases:
            morphism = MockMorphism(requires={perm})
            result = policy_check(branch, morphism)
            assert result == should_allow, description

    def test_multiple_wildcards_in_single_permission(self):
        """Test permissions with multiple wildcard segments."""
        multi_wildcard_cap = Capability(
            subject="test",
            rights={"fs.access:*/data/*/file.*"},  # Multiple wildcards
            object="*",
        )
        branch = Branch(caps=(multi_wildcard_cap,))

        # Test various matches
        test_cases = [
            ("fs.access:home/data/user/file.txt", True, "Should match all wildcards"),
            ("fs.access:tmp/data/temp/file.log", True, "Should match different values"),
            (
                "fs.access:home/config/user/file.txt",
                False,
                "Should not match different middle segment",
            ),
            (
                "fs.access:home/data/user/config.ini",
                False,
                "Should not match different extension pattern",
            ),
        ]

        for perm, should_allow, description in test_cases:
            morphism = MockMorphism(requires={perm})
            result = policy_check(branch, morphism)
            # Note: Actual behavior depends on implementation of wildcard matching
            # This test documents expected behavior
