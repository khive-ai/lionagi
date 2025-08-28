# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for capability model and coverage algorithms.

Property-based testing with Hypothesis for capability coverage algebra, 
realistic attack scenarios, and comprehensive edge cases.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
from hypothesis.strategies import composite
from uuid import uuid4

from lionagi.services.core import CallContext
from lionagi.services.middleware import PolicyGateMW


# Strategy for generating valid capability strings
@composite
def capability_string(draw):
    """Generate realistic capability strings."""
    # Capability format: namespace:resource:action or namespace:action
    namespaces = ["fs", "net", "db", "admin", "user", "api", "stream", "auth"]
    resources = ["read", "write", "delete", "create", "update", "list", "execute"]
    targets = [
        "/data", "/secure", "/public", "/tmp", "api.openai.com", "localhost", 
        "users", "logs", "cache", "session", "config"
    ]
    
    namespace = draw(st.sampled_from(namespaces))
    
    # Sometimes just namespace:action, sometimes namespace:target:action
    if draw(st.booleans()):
        action = draw(st.sampled_from(resources))
        return f"{namespace}:{action}"
    else:
        target = draw(st.sampled_from(targets))
        action = draw(st.sampled_from(resources))
        return f"{namespace}:{target}:{action}"


@composite  
def capability_set(draw, min_size=0, max_size=10):
    """Generate a set of capabilities."""
    return set(draw(st.lists(capability_string(), min_size=min_size, max_size=max_size, unique=True)))


@composite
def wildcard_capability_set(draw, min_size=0, max_size=5):
    """Generate capabilities that include wildcards."""
    regular_caps = draw(capability_set(max_size=max_size//2))
    
    # Add some wildcard capabilities
    wildcard_prefixes = ["fs.*", "net:*", "admin:*", "api.*.com", "stream:*"]
    num_wildcards = draw(st.integers(min_value=0, max_value=min(len(wildcard_prefixes), max_size//2)))
    wildcards = set(draw(st.lists(st.sampled_from(wildcard_prefixes), min_size=0, max_size=num_wildcards, unique=True)))
    
    return regular_caps | wildcards


class TestCapabilityModelCore:
    """Core capability model tests with deterministic scenarios."""

    def test_call_context_creation_with_capabilities(self):
        """Test CallContext creation with capabilities."""
        caps = {"fs:read", "net:api.openai.com", "admin:write"}
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=caps
        )
        
        assert ctx.capabilities == caps
        assert isinstance(ctx.call_id, type(uuid4()))
        assert ctx.branch_id is not None

    def test_call_context_empty_capabilities_default(self):
        """Test CallContext defaults to empty capabilities set."""
        ctx = CallContext.new(branch_id=uuid4())
        assert ctx.capabilities == set()

    def test_service_declared_vs_request_declared_precedence(self):
        """Test that service-declared capabilities take precedence in requirements."""
        policy_gate = PolicyGateMW()
        
        # Service declares strong requirements
        service_requires = {"admin:delete", "fs:write:/critical"}
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=set(),
            service_requires=service_requires
        )
        
        # Mock request with different requirements  
        class MockRequest:
            _extra_requires = {"user:read"}  # Weaker requirements from request
        
        req = MockRequest()
        
        # Should include BOTH service and request requirements (union)
        required = policy_gate._get_required_capabilities(req, ctx)
        
        assert service_requires.issubset(required), "Service requirements must be preserved"
        assert "user:read" in required, "Request extras should be added"
        assert required == service_requires | {"user:read"}

    def test_overlapping_prefixes_capability_matching(self):
        """Test capability matching with overlapping prefixes."""
        policy_gate = PolicyGateMW()
        
        # Test overlapping but different prefixes
        test_cases = [
            # (available, required, expected_match)
            ({"fs.*"}, {"filesystem:read"}, False),  # Similar but different prefix
            ({"net:*"}, {"network:tcp"}, False),     # Similar but different prefix  
            ({"api.*"}, {"api.openai.com"}, True),   # Exact prefix match
            ({"api.*"}, {"api"}, True),              # Prefix matches exactly
            ({"prefix.*"}, {"prefix_different"}, False),  # Underscore vs dot
        ]
        
        for available, required, expected in test_cases:
            result = policy_gate._capability_covers(available, required)
            assert result == expected, f"Failed: {available} covering {required}"

    def test_malformed_capability_handling(self):
        """Test handling of malformed or edge case capabilities."""
        policy_gate = PolicyGateMW()
        
        # Test various malformed/edge cases
        edge_cases = [
            # (available_set, required, expected_result)
            ({"*"}, {"anything"}, True),      # Global wildcard
            ({".*"}, {"prefix"}, False),      # Invalid wildcard format
            ({""}, {"something"}, False),     # Empty capability
            ({"normal"}, {""}, False),        # Empty required
            ({"double**"}, {"double"}, True), # Malformed wildcard treated as prefix
            ({"ends*more"}, {"ends"}, True),  # Only prefix part matched
        ]
        
        for available, required, expected in edge_cases:
            result = policy_gate._capability_covers(available, required)
            assert result == expected, f"Edge case failed: {available} -> {required}"


class TestPropertyBasedCapabilityAlgebra:
    """Property-based tests for capability coverage algebra using Hypothesis."""

    @given(available=capability_set(), required=capability_set())
    @settings(max_examples=100)
    def test_capability_coverage_reflexivity(self, available, required):
        """Property: If required ⊆ available (exact subset), coverage should succeed."""
        policy_gate = PolicyGateMW()
        
        # If all required capabilities are exactly in available set
        if required.issubset(available):
            assert policy_gate._check_capabilities(available, required), \
                f"Exact subset should be covered: available={available}, required={required}"

    @given(available=capability_set(min_size=1), required=capability_set(min_size=1))
    @settings(max_examples=100)
    def test_capability_coverage_with_disjoint_sets(self, available, required):
        """Property: Disjoint capability sets should never match (without wildcards)."""
        assume(available.isdisjoint(required))  # Ensure no overlap
        assume(not any(cap.endswith("*") for cap in available))  # No wildcards
        
        policy_gate = PolicyGateMW()
        
        # Disjoint sets without wildcards should never match
        assert not policy_gate._check_capabilities(available, required), \
            f"Disjoint sets should not match: available={available}, required={required}"

    @given(wildcard_caps=wildcard_capability_set(min_size=1), specific_caps=capability_set(min_size=1))
    @settings(max_examples=50)
    def test_wildcard_coverage_property(self, wildcard_caps, specific_caps):
        """Property: Wildcard capabilities should cover matching prefix requirements."""
        policy_gate = PolicyGateMW()
        
        for wildcard in wildcard_caps:
            if wildcard.endswith("*"):
                prefix = wildcard[:-1]
                
                # Create a specific capability that should match the wildcard
                matching_specific = f"{prefix}suffix"
                
                # The wildcard should cover the matching specific capability
                assert policy_gate._capability_covers({wildcard}, matching_specific), \
                    f"Wildcard '{wildcard}' should cover '{matching_specific}'"

    @given(caps=capability_set())
    def test_empty_requirements_always_satisfied(self, caps):
        """Property: Empty requirements should always be satisfied by any capabilities."""
        policy_gate = PolicyGateMW()
        
        # Empty requirements should always pass
        assert policy_gate._check_capabilities(caps, set())

    @given(required=capability_set(min_size=1))
    def test_no_capabilities_fails_non_empty_requirements(self, required):
        """Property: No capabilities should fail any non-empty requirements."""
        policy_gate = PolicyGateMW()
        
        # Empty available capabilities should fail non-empty requirements
        assert not policy_gate._check_capabilities(set(), required)

    @given(caps=capability_set())
    def test_capability_coverage_symmetry_for_identical_sets(self, caps):
        """Property: Identical capability sets should always match."""
        policy_gate = PolicyGateMW()
        
        # Identical sets should always match
        assert policy_gate._check_capabilities(caps, caps)

    @given(available=wildcard_capability_set(), required=capability_set())
    @settings(max_examples=50) 
    def test_wildcard_prefix_matching_property(self, available, required):
        """Property: Wildcard matching follows consistent prefix rules."""
        policy_gate = PolicyGateMW()
        
        result = policy_gate._check_capabilities(available, required)
        
        # Manual verification of the result
        should_match = True
        for req_cap in required:
            covered = False
            # Check exact match
            if req_cap in available:
                covered = True
            else:
                # Check wildcard match
                for avail_cap in available:
                    if avail_cap.endswith("*"):
                        prefix = avail_cap[:-1]
                        if req_cap.startswith(prefix):
                            covered = True
                            break
            if not covered:
                should_match = False
                break
        
        assert result == should_match, \
            f"Coverage result mismatch: available={available}, required={required}"


class TestCapabilityEdgeCases:
    """Test edge cases and boundary conditions in capability model."""

    def test_unicode_and_special_characters_in_capabilities(self):
        """Test capabilities with unicode and special characters."""
        policy_gate = PolicyGateMW()
        
        unicode_cases = [
            # (available, required, should_match)
            ({"测试:读取"}, {"测试:读取"}, True),           # Chinese characters
            ({"café:read"}, {"café:read"}, True),         # Accented characters
            ({"fs:read/file.txt"}, {"fs:read/file.txt"}, True),  # Special chars
            ({"api:*.com"}, {"api:example.com"}, True),   # Wildcards with special chars
            ({"user@domain:*"}, {"user@domain:action"}, True),  # Email-like patterns
        ]
        
        for available, required, expected in unicode_cases:
            result = policy_gate._capability_covers(set(available), required)
            assert result == expected, f"Unicode case failed: {available} -> {required}"

    def test_very_long_capability_strings(self):
        """Test behavior with very long capability strings."""
        policy_gate = PolicyGateMW()
        
        # Very long capability string (realistic for deep filesystem paths)
        long_path = "/".join([f"level{i}" for i in range(50)])
        long_capability = f"fs:read:{long_path}"
        
        # Exact match should work
        assert policy_gate._capability_covers({long_capability}, long_capability)
        
        # Wildcard should work
        wildcard = "fs:*"
        assert policy_gate._capability_covers({wildcard}, long_capability)
        
        # Non-matching long string should fail
        different_long = f"fs:write:{long_path}"
        assert not policy_gate._capability_covers({long_capability}, different_long)

    def test_case_sensitivity_in_capabilities(self):
        """Test that capability matching is case sensitive.""" 
        policy_gate = PolicyGateMW()
        
        case_tests = [
            # (available, required, should_match) - expecting case sensitive behavior
            ({"Fs:Read"}, {"fs:read"}, False),      # Different case should not match
            ({"FS:*"}, {"fs:read"}, False),         # Case sensitive prefix
            ({"api:OPENAI.com"}, {"api:openai.com"}, False),  # Case sensitive domains
        ]
        
        for available, required, expected in case_tests:
            result = policy_gate._capability_covers(set(available), required)
            assert result == expected, f"Case sensitivity test: {available} -> {required}"

    def test_capability_set_ordering_independence(self):
        """Test that capability set ordering doesn't affect results."""
        policy_gate = PolicyGateMW()
        
        caps1 = {"fs:read", "net:write", "db:admin"} 
        caps2 = {"db:admin", "fs:read", "net:write"}  # Same caps, different order
        required = {"fs:read", "net:write"}
        
        result1 = policy_gate._check_capabilities(caps1, required)
        result2 = policy_gate._check_capabilities(caps2, required)
        
        assert result1 == result2, "Capability set ordering should not affect results"


class TestServiceRequirementSemantics:
    """Test service requirement propagation and precedence rules."""

    def test_service_requires_from_context_attrs(self):
        """Test that service requirements are properly extracted from context."""
        policy_gate = PolicyGateMW()
        
        service_reqs = {"admin:delete", "fs:write:/secure"}
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=set(),
            service_requires=service_reqs  # Service requirements in attrs
        )
        
        class MockRequest:
            _extra_requires = None
        
        req = MockRequest()
        required = policy_gate._get_required_capabilities(req, ctx)
        
        assert required == service_reqs

    def test_missing_service_requires_defaults_to_empty(self):
        """Test behavior when service_requires is missing from context."""
        policy_gate = PolicyGateMW()
        
        # Context without service_requires
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=set()
            # No service_requires provided
        )
        
        class MockRequest:
            _extra_requires = {"user:read"}
        
        req = MockRequest()
        required = policy_gate._get_required_capabilities(req, ctx)
        
        # Should only have request extras since service requires is missing
        assert required == {"user:read"}

    def test_service_requires_precedence_over_context_capabilities(self):
        """Test that service requirements are separate from available capabilities."""
        policy_gate = PolicyGateMW()
        
        # Available capabilities should not automatically become requirements
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"user:read", "fs:list"},  # Available caps
            service_requires={"admin:write"}  # Different from available caps
        )
        
        class MockRequest:
            _extra_requires = None
        
        req = MockRequest()
        required = policy_gate._get_required_capabilities(req, ctx)
        
        # Requirements should only be service_requires, not available capabilities
        assert required == {"admin:write"}
        assert "user:read" not in required
        assert "fs:list" not in required


class TestRealisticAttackScenarios:
    """Test realistic attack scenarios using property-based testing."""

    @given(
        user_caps=capability_set(max_size=5),
        admin_caps=capability_set(min_size=1, max_size=3),
        request_attempts=st.lists(capability_string(), min_size=1, max_size=5)
    )
    @settings(max_examples=30)
    def test_privilege_escalation_prevention(self, user_caps, admin_caps, request_attempts):
        """Property: Users cannot escalate to admin capabilities through request manipulation."""
        assume(user_caps.isdisjoint(admin_caps))  # User and admin caps are different
        
        policy_gate = PolicyGateMW()
        
        # User has limited capabilities
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=user_caps,
            service_requires=admin_caps  # Service requires admin caps
        )
        
        class MockRequest:
            def __init__(self, extra_reqs):
                self._extra_requires = extra_reqs
        
        # Try various request manipulations
        for attempt in request_attempts:
            req = MockRequest({attempt})
            
            # Should always fail if user doesn't have admin capabilities
            required = policy_gate._get_required_capabilities(req, ctx)
            has_required_caps = policy_gate._check_capabilities(user_caps, required)
            
            assert not has_required_caps, \
                f"Privilege escalation detected: user_caps={user_caps}, required={required}"

    @given(
        legitimate_caps=capability_set(min_size=2, max_size=8),
        malicious_patterns=st.lists(
            st.one_of(
                st.just("*"),           # Global wildcard attempt
                st.just("admin:*"),     # Admin wildcard attempt  
                st.just("root:*"),      # Root access attempt
                st.just("../*"),        # Path traversal attempt
            ),
            min_size=1, max_size=3
        )
    )
    @settings(max_examples=20)
    def test_malicious_capability_pattern_prevention(self, legitimate_caps, malicious_patterns):
        """Property: Malicious capability patterns cannot grant unauthorized access."""
        # Assume user only has legitimate capabilities (no wildcards or admin access)
        assume(not any(cap.endswith("*") or "admin" in cap or "root" in cap for cap in legitimate_caps))
        
        policy_gate = PolicyGateMW()
        
        # Service requires high-privilege access
        high_privilege_reqs = {"admin:delete", "root:execute", "system:shutdown"}
        
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=legitimate_caps,  # Only legitimate user capabilities
            service_requires=high_privilege_reqs
        )
        
        # Try to use malicious patterns in request
        class MockRequest:
            def __init__(self, patterns):
                self._extra_requires = set(patterns)
        
        req = MockRequest(malicious_patterns)
        
        # Should never succeed with legitimate caps + malicious request patterns
        required = policy_gate._get_required_capabilities(req, ctx)
        has_access = policy_gate._check_capabilities(legitimate_caps, required)
        
        assert not has_access, \
            f"Malicious pattern granted access: caps={legitimate_caps}, patterns={malicious_patterns}"


class TestCapabilityModelIntegration:
    """Integration tests between capability model and other components."""

    def test_capability_model_with_context_timeout(self):
        """Test capability model works correctly with timeout context."""
        policy_gate = PolicyGateMW()
        
        # Create context with timeout and capabilities
        ctx = CallContext.with_timeout(
            branch_id=uuid4(),
            timeout_s=5.0,
            capabilities={"fs:read", "net:api.openai.com"},
            service_requires={"fs:read"}
        )
        
        assert ctx.remaining_time is not None
        assert not ctx.is_expired
        
        class MockRequest:
            _extra_requires = None
        
        req = MockRequest()
        required = policy_gate._get_required_capabilities(req, ctx)
        
        # Capability checking should work normally with timeout context
        assert required == {"fs:read"}
        assert policy_gate._check_capabilities(ctx.capabilities, required)

    def test_capability_model_serialization_compatibility(self):
        """Test that capability model works with msgspec serialization."""
        import msgspec
        
        # Test CallContext with capabilities can be serialized
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"fs:read:/data", "net:out:api.openai.com"},
            service_requires={"fs:read:/data"}
        )
        
        # Should be serializable with msgspec (validates msgspec.Struct usage)
        encoded = msgspec.encode(ctx)
        decoded = msgspec.decode(encoded, type=CallContext)
        
        assert decoded.capabilities == ctx.capabilities
        assert decoded.attrs == ctx.attrs
        assert decoded.call_id == ctx.call_id
        assert decoded.branch_id == ctx.branch_id