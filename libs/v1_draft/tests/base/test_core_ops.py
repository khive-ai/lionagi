# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for ops/core.py - All core operations with edge cases.

Focus: Policy surface propagation, capability validation, schema enforcement,
and structured concurrency patterns.
"""

import anyio
import msgspec
import pytest

from lionagi.base.graph import OpGraph, OpNode
from lionagi.base.ipu import StrictIPU, default_invariants
from lionagi.base.runner import Runner
from lionagi.base.types import Capability, create_branch
from lionagi.ops.core import (
    CtxSet,
    FSRead,
    HttpClient,
    HTTPGet,
    InMemoryKV,
    KVGet,
    KVSet,
    LLMGenerate,
    LLMProvider,
    SubgraphRun,
    TextOut,
    WithRetry,
    WithTimeout,
)


class StubLLMProvider:
    """Test LLM provider that uppercases input."""

    async def generate(self, prompt: str) -> str:
        return prompt.upper()


class StubHTTPClient:
    """Test HTTP client with predictable responses."""

    async def get(self, url: str) -> tuple[int, str]:
        return (200, f"Response for {url}")


class FailingLLMProvider:
    """LLM provider that always fails for retry testing."""

    async def generate(self, prompt: str) -> str:
        raise TimeoutError("Simulated timeout")


class FlakySingleRetryProvider:
    """Provider that fails once then succeeds."""

    def __init__(self):
        self.call_count = 0

    async def generate(self, prompt: str) -> str:
        self.call_count += 1
        if self.call_count == 1:
            raise TimeoutError("First call fails")
        return prompt.upper()


class SlowLLMProvider:
    """Provider that takes longer than timeout for timeout testing."""

    async def generate(self, prompt: str) -> str:
        await anyio.sleep(0.1)  # 100ms delay
        return prompt.upper()


@pytest.mark.anyio
async def test_llm_generate_basic(anyio_backend):
    """Test LLMGenerate basic functionality and schema validation."""
    br = create_branch(capabilities={"net.out:*"})
    op = LLMGenerate(StubLLMProvider())

    # Validate pre-condition
    assert await op.pre(br, prompt="test")
    assert not await op.pre(br)  # Missing prompt
    assert not await op.pre(br, prompt=123)  # Non-string prompt

    # Test apply
    res = await op.apply(br, prompt="hello")

    # Validate post-condition
    assert await op.post(br, res)

    # Validate schema compliance
    text_out = msgspec.json.decode(msgspec.json.encode(res), type=TextOut)
    assert text_out.text == "HELLO"

    # Validate ctx writes
    assert br.ctx["last_llm"] == "HELLO"


@pytest.mark.anyio
async def test_llm_generate_schema_lock(anyio_backend):
    """Test that LLMGenerate enforces TextOut schema strictly."""

    class BadProvider:
        async def generate(self, prompt: str) -> dict:
            return {"not": "text"}  # Wrong return type

    br = create_branch(capabilities={"net.out:*"})
    op = LLMGenerate(BadProvider())

    # This should fail at runtime when trying to validate schema
    with pytest.raises((TypeError, KeyError, msgspec.ValidationError)):
        res = await op.apply(br, prompt="test")
        # If it doesn't fail in apply, it should fail in post
        await op.post(br, res)


@pytest.mark.anyio
async def test_httpget_required_rights_parsing(anyio_backend):
    """Test HTTPGet host parsing including edge cases."""
    op = HTTPGet(StubHTTPClient(), host="*")

    # Normal host parsing
    rights = op.required_rights(url="https://api.example.com/path")
    assert rights == {"net.out:api.example.com"}

    # IPv6 with port
    rights = op.required_rights(url="http://[::1]:8080/test")
    assert rights == {"net.out:[::1]:8080"}

    # Port without IPv6
    rights = op.required_rights(url="https://api.example.com:443/path")
    assert rights == {"net.out:api.example.com:443"}

    # Invalid URL fallback
    rights = op.required_rights(url="not a valid url")
    assert rights == {"net.out:*"}  # Falls back to default

    # Missing URL parameter
    rights = op.required_rights()
    assert rights == {"net.out:*"}  # Falls back to default


@pytest.mark.anyio
async def test_httpget_basic_operation(anyio_backend):
    """Test HTTPGet basic functionality."""
    br = create_branch(capabilities={"net.out:api.example.com"})
    op = HTTPGet(StubHTTPClient())

    # Test pre-condition
    assert await op.pre(br, url="https://api.example.com/test")
    assert not await op.pre(br)  # Missing URL
    assert not await op.pre(br, url=123)  # Non-string URL

    # Test apply
    res = await op.apply(br, url="https://api.example.com/test")

    # Validate result structure
    assert await op.post(br, res)
    assert res["status"] == 200
    assert "api.example.com" in res["body"]


@pytest.mark.anyio
async def test_fsread_reads_file(anyio_backend, tmp_path):
    """Test FSRead with real file I/O."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!", encoding="utf-8")

    # Create operation with appropriate pattern
    op = FSRead(allow_pattern=str(tmp_path / "*"))
    br = create_branch(capabilities={f"fs.read:{test_file}"})

    # Test pre-condition
    assert await op.pre(br, path=str(test_file))
    assert not await op.pre(br)  # Missing path
    assert not await op.pre(br, path=123)  # Non-string path

    # Test apply
    res = await op.apply(br, path=str(test_file))

    # Validate result
    assert res["data"] == "Hello, World!"
    assert str(test_file) in res["path"]


@pytest.mark.anyio
async def test_fsread_required_rights_normalization(anyio_backend, tmp_path):
    """Test FSRead path normalization and resolution."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("content", encoding="utf-8")

    op = FSRead()

    # Test path resolution with tilde expansion
    # Note: This test assumes the test runs in an environment where ~/ is valid
    rights = op.required_rights(path=str(test_file))
    assert f"fs.read:{test_file.resolve()}" in rights

    # Test handling of invalid paths
    rights = op.required_rights(path=None)
    assert rights == {"fs.read:/*"}  # Falls back to default pattern


@pytest.mark.anyio
async def test_kv_operations(anyio_backend):
    """Test KV set/get operations."""
    kv = InMemoryKV()
    set_op = KVSet(kv, "test_ns")
    get_op = KVGet(kv, "test_ns")
    br = create_branch()

    # Test KVSet pre-conditions
    assert await set_op.pre(br, key="test", value=42)
    assert not await set_op.pre(br, key="test")  # Missing value
    assert not await set_op.pre(br, value=42)  # Missing key

    # Test KVGet pre-conditions
    assert await get_op.pre(br, key="test")
    assert not await get_op.pre(br)  # Missing key

    # Test set operation
    set_result = await set_op.apply(br, key="test", value=42)
    assert set_result["key"] == "test"
    assert set_result["ok"] is True

    # Test get operation
    get_result = await get_op.apply(br, key="test")
    assert get_result["key"] == "test"
    assert get_result["value"] == 42

    # Test get non-existent key
    get_result = await get_op.apply(br, key="missing")
    assert get_result["value"] is None


@pytest.mark.anyio
async def test_ctxset_allows_only_declared_keys(anyio_backend):
    """Test CtxSet capability enforcement."""
    br = create_branch()

    # Valid operation - keys are subset of allowed
    good_op = CtxSet({"x": 1, "y": 2}, allowed_keys={"x", "y", "z"})
    assert await good_op.pre(br)

    result = await good_op.apply(br)
    assert await good_op.post(br, result)
    assert br.ctx["x"] == 1
    assert br.ctx["y"] == 2

    # Invalid operation - key not in allowed set
    bad_op = CtxSet({"forbidden": 3}, allowed_keys={"x", "y"})
    assert not await bad_op.pre(br)


@pytest.mark.anyio
async def test_withretry_retry_paths(anyio_backend):
    """Test WithRetry behavior including jitter settings."""
    # Test retries with failing provider
    failing_op = WithRetry(LLMGenerate(FailingLLMProvider()), retries=2, jitter=False)
    br = create_branch(capabilities={"net.out:*"})

    with pytest.raises(TimeoutError):
        await failing_op.apply(br, prompt="test")

    # Test successful retry after one failure
    flaky_provider = FlakySingleRetryProvider()
    retry_op = WithRetry(LLMGenerate(flaky_provider), retries=2, jitter=False)

    result = await retry_op.apply(br, prompt="test")
    assert result["text"] == "TEST"
    assert flaky_provider.call_count == 2  # Failed once, succeeded second time


@pytest.mark.anyio
async def test_withretry_preserves_policy_surface(anyio_backend):
    """Test that WithRetry preserves all policy surface attributes."""
    inner_op = LLMGenerate(StubLLMProvider(), host="api.test.com", latency_budget_ms=1500)
    retry_op = WithRetry(inner_op, retries=1)

    # Verify policy surface preservation
    assert retry_op.requires == inner_op.requires
    assert retry_op.io == inner_op.io
    assert retry_op.ctx_writes == inner_op.ctx_writes
    assert retry_op.result_schema == inner_op.result_schema
    assert retry_op.latency_budget_ms == inner_op.latency_budget_ms


@pytest.mark.anyio
async def test_withtimeout_enforces_deadline(anyio_backend):
    """Test WithTimeout deadline enforcement."""
    slow_op = WithTimeout(LLMGenerate(SlowLLMProvider()), timeout_ms=50)  # 50ms timeout
    br = create_branch(capabilities={"net.out:*"})

    # Should timeout before SlowProvider completes (100ms delay)
    with pytest.raises(TimeoutError):
        await slow_op.apply(br, prompt="test")


@pytest.mark.anyio
async def test_withtimeout_preserves_policy_surface(anyio_backend):
    """Test that WithTimeout preserves all policy surface attributes."""
    inner_op = HTTPGet(StubHTTPClient(), host="api.test.com", latency_budget_ms=2000)
    timeout_op = WithTimeout(inner_op, timeout_ms=1000)

    # Verify policy surface preservation
    assert timeout_op.requires == inner_op.requires
    assert timeout_op.io == inner_op.io
    assert timeout_op.result_keys == inner_op.result_keys
    assert timeout_op.result_bytes_limit == inner_op.result_bytes_limit
    assert timeout_op.latency_budget_ms == inner_op.latency_budget_ms


@pytest.mark.anyio
async def test_subgraph_run_executes_nested_graph(anyio_backend):
    """Test SubgraphRun with minimal nested operation."""

    class NoopOp:
        name = "noop"
        requires = set()
        io = False

        async def pre(self, br, **kw):
            return True

        async def apply(self, br, **kw):
            br.ctx["nested_key"] = "nested_value"
            return {"ok": True}

        async def post(self, br, res):
            return res.get("ok", False)

    # Create subgraph
    graph = OpGraph()
    node = OpNode(m=NoopOp())
    graph.nodes[node.id] = node
    graph.roots = {node.id}

    # Create and run SubgraphRun operation
    br = create_branch()
    subgraph_op = SubgraphRun(graph)

    assert await subgraph_op.pre(br)
    result = await subgraph_op.apply(br)

    # Verify nested operation executed
    assert result["ok"] is True
    assert br.ctx["nested_key"] == "nested_value"


@pytest.mark.anyio
async def test_subgraph_run_requires_opgraph(anyio_backend):
    """Test SubgraphRun validation of graph parameter."""
    with pytest.raises(ValueError, match="SubgraphRun requires an OpGraph instance"):
        SubgraphRun("not a graph")


# Additional edge case tests for comprehensive coverage


@pytest.mark.anyio
async def test_baseop_default_behavior(anyio_backend):
    """Test BaseOp default implementations."""
    from lionagi.ops.core import BaseOp

    op = BaseOp()
    br = create_branch()

    # Default pre/post should return True
    assert await op.pre(br) is True
    assert await op.post(br, {}) is True

    # Check default attributes
    assert op.name == "base"
    assert op.requires == set()
    assert op.io is False
    assert op.latency_budget_ms is None


@pytest.mark.anyio
async def test_inmemory_kv_namespace_isolation(anyio_backend):
    """Test that KV namespaces are properly isolated."""
    kv = InMemoryKV()

    # Set values in different namespaces
    set_op1 = KVSet(kv, "ns1")
    set_op2 = KVSet(kv, "ns2")
    get_op1 = KVGet(kv, "ns1")
    get_op2 = KVGet(kv, "ns2")

    br = create_branch()

    # Set same key in different namespaces
    await set_op1.apply(br, key="shared", value="value1")
    await set_op2.apply(br, key="shared", value="value2")

    # Values should be isolated
    result1 = await get_op1.apply(br, key="shared")
    result2 = await get_op2.apply(br, key="shared")

    assert result1["value"] == "value1"
    assert result2["value"] == "value2"


@pytest.mark.anyio
async def test_textout_msgspec_struct_behavior(anyio_backend):
    """Test TextOut msgspec struct validation."""
    # Valid construction
    valid_out = TextOut(text="Hello")
    assert valid_out.text == "Hello"

    # Serialization round-trip
    json_data = msgspec.json.encode(valid_out)
    decoded = msgspec.json.decode(json_data, type=TextOut)
    assert decoded.text == "Hello"

    # Invalid construction should raise
    with pytest.raises(TypeError):
        TextOut()  # Missing required text field
