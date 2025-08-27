# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for morph/binders.py and morph/wrappers.py - Binding and patching operations.

Focus: Parameter precedence rules, policy surface propagation, and ctx operations.
"""

import pytest

from lionagi.base.types import create_branch
from lionagi.morph.binders import BoundOp
from lionagi.morph.wrappers import OpThenPatch


class EchoOp:
    """Test operation that returns all received kwargs."""

    name = "echo"
    requires = {"test.op"}
    io = False
    ctx_writes = {"echo_key"}
    result_schema = None
    result_keys = {"result"}
    result_bytes_limit = 1000
    latency_budget_ms = 500

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return dict(kw)

    async def post(self, br, res):
        return True


class MinimalOp:
    """Minimal operation with no extra policy attributes."""

    name = "minimal"
    requires = set()
    io = True

    async def pre(self, br, **kw):
        return True

    async def apply(self, br, **kw):
        return {"result": "success", "input": kw.get("input", "default")}

    async def post(self, br, res):
        return True


@pytest.mark.anyio
async def test_boundop_parameter_precedence(anyio_backend):
    """Test BoundOp parameter precedence: ctx -> defaults -> runtime (runtime wins)."""
    br = create_branch()
    br.ctx.update({"ctx_val": "from_ctx", "shared": "ctx_version"})

    op = BoundOp(
        EchoOp(),
        bind={"bound_param": "ctx_val", "shared_param": "shared"},
        defaults={"default_param": "default_value", "shared_param": "default_version"},
    )

    # Call with runtime parameters
    result = await op.apply(
        br,
        shared_param="runtime_version",  # Should override default and ctx
        runtime_param="runtime_value",
    )

    # Verify precedence:
    # - bound_param comes from ctx
    # - shared_param: runtime wins over defaults and ctx
    # - default_param comes from defaults
    # - runtime_param comes from runtime
    assert result["bound_param"] == "from_ctx"
    assert result["shared_param"] == "runtime_version"  # Runtime wins
    assert result["default_param"] == "default_value"
    assert result["runtime_param"] == "runtime_value"


@pytest.mark.anyio
async def test_boundop_missing_ctx_keys(anyio_backend):
    """Test BoundOp behavior when bound ctx keys are missing."""
    br = create_branch()
    # ctx is empty - no bound values available

    op = BoundOp(
        EchoOp(),
        bind={"missing_param": "nonexistent_ctx_key"},
        defaults={"backup_param": "backup_value"},
    )

    result = await op.apply(br, runtime_param="runtime")

    # missing_param should not appear (no ctx value, no default)
    assert "missing_param" not in result
    # backup_param should come from defaults
    assert result["backup_param"] == "backup_value"
    # runtime_param should come from runtime
    assert result["runtime_param"] == "runtime"


@pytest.mark.anyio
async def test_boundop_policy_surface_propagation(anyio_backend):
    """Test that BoundOp preserves all policy surface attributes."""
    inner_op = EchoOp()
    bound_op = BoundOp(inner_op, bind={"x": "ctx_x"})

    # Verify all policy attributes are propagated
    assert bound_op.requires == inner_op.requires
    assert bound_op.io == inner_op.io
    assert bound_op.ctx_writes == inner_op.ctx_writes
    assert bound_op.result_schema == inner_op.result_schema
    assert bound_op.result_keys == inner_op.result_keys
    assert bound_op.result_bytes_limit == inner_op.result_bytes_limit
    assert bound_op.latency_budget_ms == inner_op.latency_budget_ms


@pytest.mark.anyio
async def test_boundop_minimal_inner_attributes(anyio_backend):
    """Test BoundOp with minimal inner operation (missing optional attributes)."""
    inner_op = MinimalOp()
    bound_op = BoundOp(inner_op)

    # Should handle missing optional attributes gracefully
    assert bound_op.requires == set()
    assert bound_op.io is True
    assert bound_op.ctx_writes is None  # Not defined in MinimalOp
    assert bound_op.result_schema is None
    assert bound_op.result_keys is None
    assert bound_op.result_bytes_limit is None
    assert bound_op.latency_budget_ms is None


@pytest.mark.anyio
async def test_opthen_patch_identity_mapping(anyio_backend):
    """Test OpThenPatch with identity mapping (copy result keys to same ctx keys)."""
    br = create_branch()

    # Identity mapping: result keys -> same ctx keys
    op = OpThenPatch(MinimalOp(), ["result", "missing_key"])

    result = await op.apply(br, input="test_input")

    # Verify result returned unchanged
    assert result["result"] == "success"
    assert result["input"] == "test_input"

    # Verify ctx patching - only existing keys are copied
    assert br.ctx["result"] == "success"
    assert "missing_key" not in br.ctx  # Missing keys are ignored


@pytest.mark.anyio
async def test_opthen_patch_mapping_dict(anyio_backend):
    """Test OpThenPatch with source -> destination mapping."""
    br = create_branch()

    # Map result keys to different ctx keys
    op = OpThenPatch(
        MinimalOp(), {"result": "ctx_result", "input": "ctx_input", "nonexistent": "ctx_missing"}
    )

    result = await op.apply(br, input="test_data")

    # Verify result unchanged
    assert result["result"] == "success"
    assert result["input"] == "test_data"

    # Verify ctx mapping - only existing source keys are copied
    assert br.ctx["ctx_result"] == "success"
    assert br.ctx["ctx_input"] == "test_data"
    assert "ctx_missing" not in br.ctx  # nonexistent source key ignored


@pytest.mark.anyio
async def test_opthen_patch_ctx_writes_declaration(anyio_backend):
    """Test that OpThenPatch correctly declares ctx_writes."""
    inner_op = EchoOp()  # Has ctx_writes = {"echo_key"}

    # Test with identity mapping
    identity_op = OpThenPatch(inner_op, ["patch_key1", "patch_key2"])
    expected_writes = {"echo_key", "patch_key1", "patch_key2"}
    assert identity_op.ctx_writes == expected_writes

    # Test with dict mapping
    dict_op = OpThenPatch(inner_op, {"src": "dst_key"})
    expected_writes = {"echo_key", "dst_key"}
    assert dict_op.ctx_writes == expected_writes


@pytest.mark.anyio
async def test_opthen_patch_no_inner_ctx_writes(anyio_backend):
    """Test OpThenPatch when inner operation has no ctx_writes."""
    inner_op = MinimalOp()  # No ctx_writes attribute

    op = OpThenPatch(inner_op, {"result": "ctx_result"})

    # Should only include patch targets
    assert op.ctx_writes == {"ctx_result"}


@pytest.mark.anyio
async def test_opthen_patch_policy_surface_propagation(anyio_backend):
    """Test that OpThenPatch preserves policy surface attributes."""
    inner_op = EchoOp()
    patch_op = OpThenPatch(inner_op, {"result": "ctx_result"})

    # Verify basic attributes propagated
    assert patch_op.requires == inner_op.requires
    assert patch_op.io == inner_op.io

    # Verify other policy attributes propagated
    assert patch_op.result_schema == inner_op.result_schema
    assert patch_op.result_keys == inner_op.result_keys
    assert patch_op.result_bytes_limit == inner_op.result_bytes_limit
    assert patch_op.latency_budget_ms == inner_op.latency_budget_ms

    # ctx_writes should be augmented, not just copied
    expected_writes = set(inner_op.ctx_writes) | {"ctx_result"}
    assert patch_op.ctx_writes == expected_writes


@pytest.mark.anyio
async def test_boundop_pre_post_delegation(anyio_backend):
    """Test that BoundOp correctly delegates pre/post with transformed kwargs."""

    class ValidatingOp:
        name = "validator"
        requires = set()
        io = False

        async def pre(self, br, **kw):
            # Require specific parameter
            return "required_param" in kw and kw["required_param"] == "expected_value"

        async def apply(self, br, **kw):
            return {"validated": True}

        async def post(self, br, res):
            return "validated" in res

    br = create_branch()
    br.ctx["ctx_required"] = "expected_value"

    # Bind required parameter from context
    op = BoundOp(ValidatingOp(), bind={"required_param": "ctx_required"})

    # pre() should see the bound parameter
    assert await op.pre(br)

    # apply() should work
    result = await op.apply(br)

    # post() should validate result
    assert await op.post(br, result)


@pytest.mark.anyio
async def test_opthen_patch_preserves_delegation(anyio_backend):
    """Test that OpThenPatch delegates pre/post without interference."""

    class StrictOp:
        name = "strict"
        requires = set()
        io = False

        async def pre(self, br, **kw):
            return "input" in kw and len(kw["input"]) > 3

        async def apply(self, br, **kw):
            return {"processed": kw["input"].upper()}

        async def post(self, br, res):
            return "processed" in res and res["processed"].isupper()

    br = create_branch()
    op = OpThenPatch(StrictOp(), {"processed": "ctx_processed"})

    # Should pass through validation
    assert await op.pre(br, input="test")  # Length > 3
    assert not await op.pre(br, input="hi")  # Length <= 3

    # Should execute and patch
    result = await op.apply(br, input="hello")
    assert result["processed"] == "HELLO"
    assert br.ctx["ctx_processed"] == "HELLO"

    # Should validate result
    assert await op.post(br, result)
