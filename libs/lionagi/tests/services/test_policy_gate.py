# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for PolicyGateMW - capability-based security enforcement.

These tests validate the core security model: fail-closed behavior, capability coverage
algorithms, service-declared vs request-declared requirements, and attack vector prevention.
"""

import pytest
import anyio
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

from lionagi.errors import PolicyError
from lionagi.services.core import CallContext
from lionagi.services.endpoint import RequestModel
from lionagi.services.middleware import PolicyGateMW


class MockRequest(RequestModel):
    """Mock request for testing."""
    model: str = "test-model"
    _extra_requires: set[str] | None = None


class TestPolicyGateMWSynchronousEnforcement:
    """CRITICAL SECURITY TESTS: Fail-closed behavior validation.
    
    These tests ensure PolicyGateMW denies access when capabilities are insufficient
    and that next_call() is NEVER executed on security failures.
    """

    @pytest.mark.anyio
    async def test_fail_closed_insufficient_capabilities(self):
        """CRITICAL: PolicyGateMW must deny and prevent execution when capabilities insufficient."""
        # Arrange
        policy_gate = PolicyGateMW()
        next_call_executed = False
        
        async def mock_next_call():
            nonlocal next_call_executed
            next_call_executed = True
            return "should_not_reach_here"

        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"fs.read:/safe"},  # Available capability
            service_requires={"fs.write:/safe"}  # Required capability (not available)
        )
        
        req = MockRequest()

        # Act & Assert
        with pytest.raises(PolicyError) as exc_info:
            await policy_gate._enforce_policy(req, ctx, mock_next_call)
        
        # Verify next_call was NEVER executed (fail-closed)
        assert not next_call_executed, "CRITICAL SECURITY FAILURE: next_call() executed despite insufficient capabilities"
        
        # Verify error contains security context
        error_context = exc_info.value.context
        assert error_context["policy_check"] == "capability_enforcement"
        assert "fs.write:/safe" in error_context["missing_capabilities"]
        assert error_context["operation"] == "call"

    @pytest.mark.anyio
    async def test_fail_closed_no_capabilities_provided(self):
        """CRITICAL: Policy must deny when no capabilities provided but service requires some."""
        policy_gate = PolicyGateMW()
        next_call_executed = False
        
        async def mock_next_call():
            nonlocal next_call_executed
            next_call_executed = True
            return "unauthorized_access"

        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=set(),  # No capabilities
            service_requires={"net.out:api.openai.com"}  # Service requires network access
        )
        
        req = MockRequest()

        # Act & Assert
        with pytest.raises(PolicyError) as exc_info:
            await policy_gate._enforce_policy(req, ctx, mock_next_call)
        
        assert not next_call_executed, "CRITICAL: Unauthorized access allowed with no capabilities"
        assert "net.out:api.openai.com" in exc_info.value.context["missing_capabilities"]

    @pytest.mark.anyio
    async def test_streaming_fail_closed_enforcement(self):
        """CRITICAL: Streaming operations must also fail-closed on insufficient capabilities."""
        policy_gate = PolicyGateMW()
        stream_started = False
        
        async def mock_next_stream():
            nonlocal stream_started
            stream_started = True
            yield "unauthorized_chunk"

        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"fs.read:/public"},
            service_requires={"net.out:*"}  # Requires network access
        )
        
        req = MockRequest()

        # Act & Assert
        with pytest.raises(PolicyError) as exc_info:
            async for _ in policy_gate._enforce_policy_stream(req, ctx, mock_next_stream):
                pass
        
        assert not stream_started, "CRITICAL: Streaming started despite insufficient capabilities"
        assert exc_info.value.context["operation"] == "streaming"


class TestCapabilityUnionSemantics:
    """Test capability requirement union: service_requires + request extras."""

    @pytest.mark.anyio
    async def test_service_requires_plus_request_extras_union(self):
        """Service requirements union with request extras must all be satisfied."""
        policy_gate = PolicyGateMW()
        
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"fs.read:/data", "net.out:api.openai.com"},  # Has both
            service_requires={"fs.read:/data"}  # Service requirement
        )
        
        # Request adds additional requirement
        req = MockRequest()
        req._extra_requires = {"net.out:api.openai.com"}

        # Should succeed - both service and request requirements satisfied
        async def mock_success():
            return "success"
            
        result = await policy_gate._enforce_policy(req, ctx, mock_success)
        assert result == "success"

    @pytest.mark.anyio
    async def test_missing_service_requirement_fails(self):
        """Missing service-declared requirement must fail even if request extras satisfied."""
        policy_gate = PolicyGateMW()
        executed = False
        
        async def should_not_execute():
            nonlocal executed
            executed = True
            return "breach"

        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"net.out:api.openai.com"},  # Only has request extra capability
            service_requires={"fs.write:/secure"}  # Missing service requirement
        )
        
        req = MockRequest()
        req._extra_requires = {"net.out:api.openai.com"}  # Request extra is satisfied

        with pytest.raises(PolicyError):
            await policy_gate._enforce_policy(req, ctx, should_not_execute)
        
        assert not executed, "Service requirement bypass attempted"

    @pytest.mark.anyio
    async def test_missing_request_extra_fails(self):
        """Missing request extra requirement must fail even if service requirements satisfied."""
        policy_gate = PolicyGateMW()
        executed = False
        
        async def should_not_execute():
            nonlocal executed
            executed = True
            return "breach"

        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"fs.read:/data"},  # Only has service capability
            service_requires={"fs.read:/data"}  # Service requirement satisfied
        )
        
        req = MockRequest()
        req._extra_requires = {"db.write:sensitive"}  # Missing request extra

        with pytest.raises(PolicyError):
            await policy_gate._enforce_policy(req, ctx, should_not_execute)
        
        assert not executed, "Request extra requirement bypass attempted"


class TestWildcardCapabilityMatching:
    """Test wildcard capability matching - only available side can use wildcards."""

    @pytest.mark.anyio
    async def test_wildcard_available_matches_specific_required(self):
        """Available wildcard capability should match specific required capability."""
        policy_gate = PolicyGateMW()
        
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"net.out:*"},  # Wildcard available
            service_requires={"net.out:api.openai.com"}  # Specific required
        )
        
        req = MockRequest()

        async def mock_success():
            return "wildcard_match_success"
            
        result = await policy_gate._enforce_policy(req, ctx, mock_success)
        assert result == "wildcard_match_success"

    @pytest.mark.anyio  
    async def test_wildcard_prefix_matching_validation(self):
        """Wildcard matching should validate prefix correctly."""
        policy_gate = PolicyGateMW()
        
        # Test various wildcard scenarios
        test_cases = [
            # (available, required, should_match)
            ({"fs:*"}, {"fs:read"}, True),
            ({"fs:*"}, {"fs:read:file.txt"}, True),
            ({"net.out:api.*"}, {"net.out:api.openai.com"}, True),
            ({"net.out:api.*"}, {"net.out:different.com"}, False),  # Prefix mismatch
            ({"fs.read:*"}, {"fs.write:file"}, False),  # Different operation
            ({"db.*"}, {"database.read"}, False),  # Prefix doesn't match 'database'
        ]
        
        for available_caps, required_caps, should_match in test_cases:
            ctx = CallContext.new(
                branch_id=uuid4(),
                capabilities=available_caps,
                service_requires=required_caps
            )
            
            req = MockRequest()
            executed = False
            
            async def track_execution():
                nonlocal executed
                executed = True
                return "executed"
            
            if should_match:
                await policy_gate._enforce_policy(req, ctx, track_execution)
                assert executed, f"Should have matched: {available_caps} -> {required_caps}"
            else:
                with pytest.raises(PolicyError):
                    await policy_gate._enforce_policy(req, ctx, track_execution)
                assert not executed, f"Should not have matched: {available_caps} -> {required_caps}"
            
            executed = False  # Reset for next test

    @pytest.mark.anyio
    async def test_required_wildcard_not_allowed(self):
        """Required capabilities cannot use wildcards - only available side."""
        policy_gate = PolicyGateMW()
        executed = False
        
        async def should_not_execute():
            nonlocal executed
            executed = True
            return "wildcard_exploit"

        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"fs.read:specific_file.txt"},  # Specific available
            service_requires={"fs.read:*"}  # Wildcard in required (not allowed)
        )
        
        req = MockRequest()

        # This should fail because wildcard is only allowed on available side
        with pytest.raises(PolicyError):
            await policy_gate._enforce_policy(req, ctx, should_not_execute)
        
        assert not executed, "Wildcard in required capabilities should not be allowed"


class TestCapabilityOverrideProtection:
    """Test that requests cannot weaken or replace service-declared requirements."""

    def test_get_required_capabilities_union_behavior(self):
        """Test that _get_required_capabilities implements proper union semantics."""
        policy_gate = PolicyGateMW()
        
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=set(),
            service_requires={"fs.read:/data", "net.out:api.openai.com"}
        )
        
        req = MockRequest()
        req._extra_requires = {"db.write:logs", "net.out:api.openai.com"}  # Overlap with service
        
        required = policy_gate._get_required_capabilities(req, ctx)
        
        # Should be union of service + request requirements
        expected = {"fs.read:/data", "net.out:api.openai.com", "db.write:logs"}
        assert required == expected

    def test_service_requirements_cannot_be_replaced(self):
        """Verify that request cannot replace service requirements, only add to them."""
        policy_gate = PolicyGateMW()
        
        # Service declares requirements
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=set(),
            service_requires={"fs.write:/secure", "auth:admin"}
        )
        
        # Request attempts to "override" by providing different requirements
        req = MockRequest()
        req._extra_requires = {"fs.read:/public"}  # Different, weaker requirements
        
        required = policy_gate._get_required_capabilities(req, ctx)
        
        # Service requirements must still be present (cannot be weakened)
        assert "fs.write:/secure" in required
        assert "auth:admin" in required
        assert "fs.read:/public" in required  # Request extra added, not replaced

    def test_empty_request_extras_preserves_service_requirements(self):
        """Empty or None request extras should preserve service requirements."""
        policy_gate = PolicyGateMW()
        
        service_reqs = {"critical:admin", "fs.write:/sensitive"}
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities=set(),
            service_requires=service_reqs
        )
        
        # Test with None extra requirements
        req = MockRequest()
        req._extra_requires = None
        
        required = policy_gate._get_required_capabilities(req, ctx)
        assert required == service_reqs
        
        # Test with empty set
        req._extra_requires = set()
        required = policy_gate._get_required_capabilities(req, ctx)
        assert required == service_reqs


class TestCapabilityCoverageAlgorithm:
    """Test the core capability coverage algorithm with edge cases."""

    def test_capability_covers_exact_matching(self):
        """Test exact capability matching."""
        policy_gate = PolicyGateMW()
        
        available = {"fs.read:/data", "net.out:api.openai.com", "db.read:users"}
        
        # Exact matches should work
        assert policy_gate._capability_covers(available, "fs.read:/data")
        assert policy_gate._capability_covers(available, "net.out:api.openai.com") 
        assert policy_gate._capability_covers(available, "db.read:users")
        
        # Non-matches should fail
        assert not policy_gate._capability_covers(available, "fs.write:/data")
        assert not policy_gate._capability_covers(available, "net.out:api.anthropic.com")
        assert not policy_gate._capability_covers(available, "db.write:users")

    def test_capability_covers_wildcard_edge_cases(self):
        """Test wildcard matching edge cases and malformed patterns."""
        policy_gate = PolicyGateMW()
        
        # Edge cases for wildcard matching
        test_cases = [
            # (available_set, required, expected_result)
            ({"*"}, {"anything"}, True),  # Global wildcard
            ({"fs.*"}, {"fs."}, True),    # Empty suffix after prefix
            ({"fs.*"}, {"fs"}, False),    # Prefix without separator doesn't match
            ({"fs.read.*"}, {"fs.read.file.txt"}, True),  # Multiple levels
            ({"prefix*"}, {"prefix"}, True),  # Exact prefix match
            ({"prefix*"}, {"prefi"}, False),  # Partial prefix doesn't match
            ({"*suffix"}, {"something"}, False),  # Suffix wildcards not supported (only prefix)
            ({"middle*end"}, {"middle"}, True),  # Only prefix part of wildcard pattern used
        ]
        
        for available_set, required, expected in test_cases:
            result = policy_gate._capability_covers(available_set, required)
            assert result == expected, f"Failed for {available_set} -> {required} (expected {expected})"

    def test_check_capabilities_empty_requirements(self):
        """Empty requirements should always pass."""
        policy_gate = PolicyGateMW()
        
        # Any available capabilities should satisfy empty requirements
        assert policy_gate._check_capabilities(set(), set())
        assert policy_gate._check_capabilities({"some:capability"}, set())
        assert policy_gate._check_capabilities({"many", "different", "caps"}, set())

    def test_check_capabilities_comprehensive_scenarios(self):
        """Test comprehensive capability checking scenarios."""
        policy_gate = PolicyGateMW()
        
        scenarios = [
            # (available, required, should_pass, description)
            (
                {"fs:*", "net.out:api.openai.com"}, 
                {"fs:read", "net.out:api.openai.com"}, 
                True, 
                "Mixed exact and wildcard should work"
            ),
            (
                {"admin:*"}, 
                {"user:read", "admin:write"}, 
                False, 
                "Wildcard doesn't cover different prefix"
            ),
            (
                {"fs.read:*", "fs.write:*"}, 
                {"fs.read:file1", "fs.write:file2", "fs.delete:file3"}, 
                False, 
                "Multiple wildcards, one requirement not covered"
            ),
            (
                {"net:*", "db:*", "fs:*"}, 
                {"net:tcp", "db:select", "fs:read", "net:udp"}, 
                True, 
                "Multiple wildcards covering all requirements"
            ),
        ]
        
        for available, required, should_pass, description in scenarios:
            result = policy_gate._check_capabilities(available, required)
            assert result == should_pass, f"Failed scenario: {description}"


class TestAttackVectorPrevention:
    """Test prevention of various attack vectors against the capability system."""

    @pytest.mark.anyio
    async def test_capability_injection_prevention(self):
        """Test that malformed capabilities cannot be injected through request data."""
        policy_gate = PolicyGateMW()
        executed = False
        
        async def should_not_execute():
            nonlocal executed
            executed = True
            return "injection_successful"

        # Attempt to inject capabilities through various request fields
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"fs.read:/safe"},
            service_requires={"admin:root"}  # High privilege requirement
        )
        
        # Malicious request attempting capability injection
        req = MockRequest()
        req.model = "admin:root"  # Attempt to inject through model field
        req._extra_requires = {"admin:root"}  # Even if extra requirement matches, still need available capability
        
        with pytest.raises(PolicyError):
            await policy_gate._enforce_policy(req, ctx, should_not_execute)
        
        assert not executed, "Capability injection attack prevented"

    @pytest.mark.anyio
    async def test_service_requirement_bypass_prevention(self):
        """Test that service requirements cannot be bypassed through context manipulation."""
        policy_gate = PolicyGateMW()
        executed = False
        
        async def should_not_execute():
            nonlocal executed  
            executed = True
            return "bypass_successful"

        # Service requires admin access
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"user:read"},  # Only user-level access
            service_requires={"admin:write"}  # Requires admin
        )
        
        # Various bypass attempts
        req = MockRequest()
        
        # Attempt 1: Try to add admin to available capabilities via request
        # (This shouldn't work as capabilities come from context, not request)
        req._extra_requires = set()  # Even empty extras shouldn't help
        
        with pytest.raises(PolicyError):
            await policy_gate._enforce_policy(req, ctx, should_not_execute)
        
        assert not executed, "Service requirement bypass prevented"

    @pytest.mark.anyio  
    async def test_wildcard_abuse_prevention(self):
        """Test that wildcard patterns cannot be abused for privilege escalation."""
        policy_gate = PolicyGateMW()
        executed = False
        
        async def should_not_execute():
            nonlocal executed
            executed = True
            return "wildcard_abuse_successful"

        # User has broad but limited wildcard access
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"user:*"},  # User namespace wildcard
            service_requires={"admin:delete"}  # Admin operation required
        )
        
        req = MockRequest()
        
        # Should fail - user:* doesn't cover admin:delete
        with pytest.raises(PolicyError):
            await policy_gate._enforce_policy(req, ctx, should_not_execute)
        
        assert not executed, "Wildcard privilege escalation prevented"


class TestErrorContextAndObservability:
    """Test that security failures provide proper error context for observability."""

    @pytest.mark.anyio
    async def test_policy_error_contains_security_audit_context(self):
        """Policy errors must contain comprehensive context for security auditing."""
        policy_gate = PolicyGateMW()
        
        call_id = uuid4()
        branch_id = uuid4()
        
        ctx = CallContext.new(
            branch_id=branch_id,
            capabilities={"fs.read:/public", "net.out:example.com"},
            service_requires={"admin:write", "fs.delete:/critical"}
        )
        ctx.call_id = call_id  # Set specific call_id for testing
        
        req = MockRequest()
        req._extra_requires = {"db.admin:delete"}

        async def never_called():
            return "security_breach"

        with pytest.raises(PolicyError) as exc_info:
            await policy_gate._enforce_policy(req, ctx, never_called)
        
        error_context = exc_info.value.context
        
        # Verify comprehensive security context
        assert error_context["call_id"] == str(call_id)
        assert error_context["branch_id"] == str(branch_id)
        assert error_context["operation"] == "call"
        assert error_context["policy_check"] == "capability_enforcement"
        
        # Verify capability analysis
        available = set(error_context["available_capabilities"])
        required = set(error_context["required_capabilities"])  
        missing = set(error_context["missing_capabilities"])
        
        assert available == {"fs.read:/public", "net.out:example.com"}
        assert required == {"admin:write", "fs.delete:/critical", "db.admin:delete"}
        assert missing == {"admin:write", "fs.delete:/critical", "db.admin:delete"}
        
        # Verify missing capabilities are correctly calculated
        assert missing == required - available

    @pytest.mark.anyio
    async def test_streaming_error_context_completeness(self):
        """Streaming policy errors must also provide complete security context."""
        policy_gate = PolicyGateMW()
        
        ctx = CallContext.new(
            branch_id=uuid4(),
            capabilities={"stream:read"},
            service_requires={"stream:write", "admin:manage"}
        )
        
        req = MockRequest()
        
        async def never_streams():
            yield "unauthorized_data"

        with pytest.raises(PolicyError) as exc_info:
            async for _ in policy_gate._enforce_policy_stream(req, ctx, never_streams):
                pass
        
        error_context = exc_info.value.context
        assert error_context["operation"] == "streaming"
        assert "stream:write" in error_context["missing_capabilities"]
        assert "admin:manage" in error_context["missing_capabilities"]


@pytest.mark.anyio
async def test_policy_gate_success_no_interference():
    """Test that PolicyGateMW doesn't interfere when capabilities are sufficient."""
    policy_gate = PolicyGateMW()
    
    ctx = CallContext.new(
        branch_id=uuid4(),
        capabilities={"fs.read:/data", "net.out:api.openai.com", "admin:write"},
        service_requires={"fs.read:/data", "net.out:api.openai.com"}
    )
    
    req = MockRequest()
    req._extra_requires = {"admin:write"}
    
    expected_result = {"status": "success", "data": "test_result"}
    
    async def successful_call():
        return expected_result
    
    # Should pass through without modification
    result = await policy_gate._enforce_policy(req, ctx, successful_call)
    assert result == expected_result
    
    # Test streaming success as well
    expected_chunks = ["chunk1", "chunk2", "chunk3"]
    
    async def successful_stream():
        for chunk in expected_chunks:
            yield chunk
    
    received_chunks = []
    async for chunk in policy_gate._enforce_policy_stream(req, ctx, successful_stream):
        received_chunks.append(chunk)
    
    assert received_chunks == expected_chunks