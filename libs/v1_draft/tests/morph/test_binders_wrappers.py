"""Test suite for morph/binders.py and wrappers.py - Policy Surface Propagation.

Focus: BoundOp and OpThenPatch must properly propagate policy attributes
for security invariants to work correctly.
"""

import pytest

from lionagi.base.types import Branch, Capability
from lionagi.morph.binders import BoundOp
from lionagi.morph.wrappers import OpThenPatch


class MockMorphism:
    """Mock morphism with full policy surface for testing."""

    def __init__(self, **policy_attrs):
        self.name = "mock_inner"
        self.requires = {"mock.permission"}
        self.io = True
        # Policy surface attributes
        self.ctx_writes = policy_attrs.get("ctx_writes", {"test_key"})
        self.result_schema = policy_attrs.get("result_schema", "test_schema")
        self.result_keys = policy_attrs.get("result_keys", {"result"})
        self.result_bytes_limit = policy_attrs.get("result_bytes_limit", 1000)
        self.latency_budget_ms = policy_attrs.get("latency_budget_ms", 5000)

    async def pre(self, br: Branch, **kw) -> bool:
        return True

    async def apply(self, br: Branch, **kw) -> dict:
        return {"result": "test_value"}

    async def post(self, br: Branch, res: dict) -> bool:
        return True


class TestBoundOpPolicyPropagation:
    """Test BoundOp properly propagates policy surface attributes."""

    def test_boundop_propagates_ctx_writes(self):
        """Test that BoundOp propagates ctx_writes from inner morphism."""
        inner = MockMorphism(ctx_writes={"sensitive_key", "audit_log"})
        bound = BoundOp(inner, bind={}, defaults={})

        assert bound.ctx_writes == {"sensitive_key", "audit_log"}

    def test_boundop_propagates_result_schema(self):
        """Test that BoundOp propagates result_schema from inner morphism."""
        inner = MockMorphism(result_schema="complex_schema")
        bound = BoundOp(inner, bind={}, defaults={})

        assert bound.result_schema == "complex_schema"

    def test_boundop_propagates_result_keys(self):
        """Test that BoundOp propagates result_keys from inner morphism."""
        inner = MockMorphism(result_keys={"output", "metadata"})
        bound = BoundOp(inner, bind={}, defaults={})

        assert bound.result_keys == {"output", "metadata"}

    def test_boundop_propagates_result_bytes_limit(self):
        """Test that BoundOp propagates result_bytes_limit from inner morphism."""
        inner = MockMorphism(result_bytes_limit=5000000)
        bound = BoundOp(inner, bind={}, defaults={})

        assert bound.result_bytes_limit == 5000000

    def test_boundop_propagates_latency_budget_ms(self):
        """Test that BoundOp propagates latency_budget_ms from inner morphism."""
        inner = MockMorphism(latency_budget_ms=2500)
        bound = BoundOp(inner, bind={}, defaults={})

        assert bound.latency_budget_ms == 2500

    def test_boundop_propagates_all_policy_surface(self):
        """Test that BoundOp propagates entire policy surface from inner morphism."""
        inner = MockMorphism(
            ctx_writes={"key1", "key2"},
            result_schema="full_schema",
            result_keys={"out1", "out2"},
            result_bytes_limit=10000,
            latency_budget_ms=3000,
        )
        bound = BoundOp(inner, bind={}, defaults={})

        # Verify all policy attributes are propagated
        assert bound.ctx_writes == {"key1", "key2"}
        assert bound.result_schema == "full_schema"
        assert bound.result_keys == {"out1", "out2"}
        assert bound.result_bytes_limit == 10000
        assert bound.latency_budget_ms == 3000

        # Verify existing attributes are still propagated
        assert bound.requires == {"mock.permission"}
        assert bound.io is True

    def test_boundop_handles_missing_policy_attributes(self):
        """Test BoundOp handles inner morphisms without policy attributes gracefully."""

        class MinimalMorphism:
            name = "minimal"
            requires = {"minimal.perm"}
            io = False

            async def pre(self, br, **kw):
                return True

            async def apply(self, br, **kw):
                return {}

            async def post(self, br, res):
                return True

        minimal = MinimalMorphism()
        bound = BoundOp(minimal, bind={}, defaults={})

        # Should not crash, should handle None gracefully
        assert bound.ctx_writes is None
        assert bound.result_schema is None
        assert bound.result_keys is None
        assert bound.result_bytes_limit is None
        assert bound.latency_budget_ms is None


class TestOpThenPatchCtxWrites:
    """Test OpThenPatch properly declares ctx_writes for patch targets."""

    def test_opthenpatch_declares_ctx_writes_for_patch_targets(self):
        """Test that OpThenPatch declares ctx_writes including patch targets."""
        inner = MockMorphism(ctx_writes={"inner_key"})
        wrapper = OpThenPatch(inner, patch={"result": "context_target"})

        # Should include both inner ctx_writes and patch targets
        assert "inner_key" in wrapper.ctx_writes
        assert "context_target" in wrapper.ctx_writes

    def test_opthenpatch_handles_no_inner_ctx_writes(self):
        """Test OpThenPatch when inner has no ctx_writes declared."""

        class NoCtxWritesMorphism:
            name = "no_ctx"
            requires = set()
            io = False

            async def pre(self, br, **kw):
                return True

            async def apply(self, br, **kw):
                return {"result": "value"}

            async def post(self, br, res):
                return True

        inner = NoCtxWritesMorphism()
        wrapper = OpThenPatch(inner, patch={"result": "target_key"})

        # Should only include patch targets
        assert wrapper.ctx_writes == {"target_key"}

    def test_opthenpatch_handles_empty_patch(self):
        """Test OpThenPatch with empty patch mapping."""
        inner = MockMorphism(ctx_writes={"inner_key"})
        wrapper = OpThenPatch(inner, patch={})

        # Should only have inner ctx_writes
        assert wrapper.ctx_writes == {"inner_key"}

    def test_opthenpatch_propagates_other_policy_attributes(self):
        """Test that OpThenPatch propagates all policy attributes like BoundOp."""
        inner = MockMorphism(
            ctx_writes={"inner_writes"},
            result_schema="test_schema",
            result_keys={"result"},
            result_bytes_limit=8000,
            latency_budget_ms=4000,
        )
        wrapper = OpThenPatch(inner, patch={"result": "patch_target"})

        # Verify ctx_writes includes both inner and patch targets
        assert "inner_writes" in wrapper.ctx_writes
        assert "patch_target" in wrapper.ctx_writes

        # Verify other policy attributes are propagated
        assert wrapper.result_schema == "test_schema"
        assert wrapper.result_keys == {"result"}
        assert wrapper.result_bytes_limit == 8000
        assert wrapper.latency_budget_ms == 4000

    def test_opthenpatch_multiple_patch_targets(self):
        """Test OpThenPatch with multiple patch targets."""
        inner = MockMorphism(ctx_writes={"inner1", "inner2"})
        wrapper = OpThenPatch(
            inner, patch={"out1": "ctx_key1", "out2": "ctx_key2", "out3": "ctx_key3"}
        )

        expected_ctx_writes = {"inner1", "inner2", "ctx_key1", "ctx_key2", "ctx_key3"}
        assert wrapper.ctx_writes == expected_ctx_writes


class TestPolicyPropagationIntegration:
    """Integration tests for policy propagation through wrapper chains."""

    def test_nested_wrappers_propagate_policy(self):
        """Test policy propagation through nested wrappers (BoundOp -> OpThenPatch)."""
        inner = MockMorphism(
            ctx_writes={"base_write"},
            result_schema="base_schema",
            result_keys={"base_result"},
            result_bytes_limit=6000,
            latency_budget_ms=3500,
        )

        # First wrap with BoundOp
        bound = BoundOp(inner, bind={}, defaults={})

        # Then wrap with OpThenPatch
        patched = OpThenPatch(bound, patch={"base_result": "final_output"})

        # Verify full policy propagation through chain
        assert "base_write" in patched.ctx_writes
        assert "final_output" in patched.ctx_writes
        assert patched.result_schema == "base_schema"
        assert patched.result_keys == {"base_result"}
        assert patched.result_bytes_limit == 6000
        assert patched.latency_budget_ms == 3500

    def test_wrapper_policy_surface_enables_ipu_checks(self):
        """Test that wrappers with proper policy surface can be checked by IPU."""
        inner = MockMorphism(ctx_writes={"audit_log"}, result_bytes_limit=1000)

        wrapper = OpThenPatch(inner, patch={"result": "processed_data"})

        # Mock IPU check - should be able to access policy attributes
        assert hasattr(wrapper, "ctx_writes")
        assert hasattr(wrapper, "result_bytes_limit")
        assert "audit_log" in wrapper.ctx_writes
        assert "processed_data" in wrapper.ctx_writes
        assert wrapper.result_bytes_limit == 1000

        # This validates that IPU invariants can work with wrapped operations
