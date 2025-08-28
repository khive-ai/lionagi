# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for msgspec vs traditional serialization approaches.

Tests demonstrate measurable gains from v1 architectural decision to use msgspec
over Pydantic/dataclasses for core service structures.
"""

import json
import time
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import msgspec
import pytest
from pydantic import BaseModel

# Import the msgspec-based structures
from lionagi.services.core import CallContext
from lionagi.services.endpoint import ChatRequestModel, RequestModel
from lionagi.services.executor import CallStatus, ExecutorConfig, ServiceCall


# Create Pydantic equivalents for comparison
class PydanticCallContext(BaseModel):
    """Pydantic version of CallContext for benchmark comparison."""

    call_id: UUID
    branch_id: UUID
    deadline_s: float | None = None
    capabilities: set[str] = set()
    attrs: dict[str, Any] = {}


class PydanticServiceCall(BaseModel):
    """Pydantic version of ServiceCall for benchmark comparison."""
    
    model_config = {"arbitrary_types_allowed": True}

    id: UUID
    service: Any  # Mock service
    request: Any  # Mock request
    context: Any  # Mock context
    status: str = "pending"
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: Exception | None = None
    token_estimate: int = 0


class PydanticExecutorConfig(BaseModel):
    """Pydantic version of ExecutorConfig for benchmark comparison."""

    queue_capacity: int = 100
    capacity_refresh_time: float = 60.0
    interval: float | None = None
    limit_requests: int | None = None
    limit_tokens: int | None = None
    concurrency_limit: int | None = None


@pytest.fixture
def sample_data():
    """Generate realistic test data for benchmarks."""
    branch_id = uuid4()
    call_id = uuid4()

    # Realistic capabilities set
    capabilities = {
        "net.out:api.openai.com",
        "net.out:api.anthropic.com",
        "fs.read:/workspace/*",
        "compute.gpu:nvidia.com/*",
    }

    # Realistic context attributes
    attrs = {
        "trace_id": "abc123def456",
        "span_id": "789xyz",
        "user_id": "user_12345",
        "request_id": "req_abcdef123456",
        "experiment_flags": ["flag_a", "flag_b"],
        "model_config": {"temperature": 0.7, "max_tokens": 4096},
    }

    return {
        "branch_id": branch_id,
        "call_id": call_id,
        "capabilities": capabilities,
        "attrs": attrs,
        "deadline_s": time.time() + 300.0,
    }


class TestCoreStructSerialization:
    """Benchmark msgspec vs Pydantic for core service structures."""

    def test_call_context_serialization_msgspec(self, benchmark, sample_data):
        """Benchmark msgspec CallContext serialization performance."""
        ctx = CallContext(
            call_id=sample_data["call_id"],
            branch_id=sample_data["branch_id"],
            deadline_s=sample_data["deadline_s"],
            capabilities=sample_data["capabilities"],
            attrs=sample_data["attrs"],
        )

        def serialize_msgspec():
            encoded = msgspec.encode(ctx)
            decoded = msgspec.decode(encoded, type=CallContext)
            return decoded

        result = benchmark(serialize_msgspec)
        assert result.call_id == sample_data["call_id"]
        assert result.capabilities == sample_data["capabilities"]

    def test_call_context_serialization_pydantic(self, benchmark, sample_data):
        """Benchmark Pydantic CallContext serialization performance."""
        ctx = PydanticCallContext(
            call_id=sample_data["call_id"],
            branch_id=sample_data["branch_id"],
            deadline_s=sample_data["deadline_s"],
            capabilities=sample_data["capabilities"],
            attrs=sample_data["attrs"],
        )

        def serialize_pydantic():
            json_str = ctx.model_dump_json()
            decoded = PydanticCallContext.model_validate_json(json_str)
            return decoded

        result = benchmark(serialize_pydantic)
        assert result.call_id == sample_data["call_id"]
        assert result.capabilities == sample_data["capabilities"]

    def test_executor_config_serialization_msgspec(self, benchmark):
        """Benchmark msgspec ExecutorConfig serialization."""
        config = ExecutorConfig(
            queue_capacity=1000,
            capacity_refresh_time=30.0,
            interval=1.0,
            limit_requests=100,
            limit_tokens=50000,
            concurrency_limit=10,
        )

        def serialize_msgspec():
            encoded = msgspec.encode(config)
            decoded = msgspec.decode(encoded, type=ExecutorConfig)
            return decoded

        result = benchmark(serialize_msgspec)
        assert result.queue_capacity == 1000
        assert result.concurrency_limit == 10

    def test_executor_config_serialization_pydantic(self, benchmark):
        """Benchmark Pydantic ExecutorConfig serialization."""
        config = PydanticExecutorConfig(
            queue_capacity=1000,
            capacity_refresh_time=30.0,
            interval=1.0,
            limit_requests=100,
            limit_tokens=50000,
            concurrency_limit=10,
        )

        def serialize_pydantic():
            json_str = config.model_dump_json()
            decoded = PydanticExecutorConfig.model_validate_json(json_str)
            return decoded

        result = benchmark(serialize_pydantic)
        assert result.queue_capacity == 1000
        assert result.concurrency_limit == 10


class TestTransportJsonParsing:
    """Benchmark msgspec.json vs standard json for transport-level parsing."""

    @pytest.fixture
    def response_payloads(self):
        """Generate realistic API response payloads of various sizes."""
        small_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
        }

        # 4KB response - typical chat completion
        medium_response = small_response.copy()
        medium_response["choices"][0]["message"]["content"] = "A" * 3500  # ~4KB

        # 64KB response - large completion or multi-turn
        large_response = small_response.copy()
        large_response["choices"] = [
            {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": "B" * 1000,  # 1KB per choice
                },
                "finish_reason": "stop",
            }
            for i in range(60)  # ~64KB total
        ]

        # 1MB response - streaming or batch responses
        xlarge_response = small_response.copy()
        xlarge_response["choices"][0]["message"]["content"] = "C" * (1024 * 1024)  # 1MB

        return {
            "small": json.dumps(small_response),
            "medium": json.dumps(medium_response),
            "large": json.dumps(large_response),
            "xlarge": json.dumps(xlarge_response),
        }

    def test_small_payload_msgspec_parsing(self, benchmark, response_payloads):
        """Benchmark msgspec parsing for small payloads (~1KB)."""
        payload = response_payloads["small"].encode("utf-8")

        def parse_msgspec():
            return msgspec.json.decode(payload)

        result = benchmark(parse_msgspec)
        assert result["model"] == "gpt-4"
        assert "choices" in result

    def test_small_payload_stdlib_parsing(self, benchmark, response_payloads):
        """Benchmark stdlib json parsing for small payloads (~1KB)."""
        payload = response_payloads["small"]

        def parse_stdlib():
            return json.loads(payload)

        result = benchmark(parse_stdlib)
        assert result["model"] == "gpt-4"
        assert "choices" in result

    def test_medium_payload_msgspec_parsing(self, benchmark, response_payloads):
        """Benchmark msgspec parsing for medium payloads (~4KB)."""
        payload = response_payloads["medium"].encode("utf-8")

        def parse_msgspec():
            return msgspec.json.decode(payload)

        result = benchmark(parse_msgspec)
        assert result["model"] == "gpt-4"
        assert len(result["choices"][0]["message"]["content"]) > 3000

    def test_medium_payload_stdlib_parsing(self, benchmark, response_payloads):
        """Benchmark stdlib json parsing for medium payloads (~4KB)."""
        payload = response_payloads["medium"]

        def parse_stdlib():
            return json.loads(payload)

        result = benchmark(parse_stdlib)
        assert result["model"] == "gpt-4"
        assert len(result["choices"][0]["message"]["content"]) > 3000

    def test_large_payload_msgspec_parsing(self, benchmark, response_payloads):
        """Benchmark msgspec parsing for large payloads (~64KB)."""
        payload = response_payloads["large"].encode("utf-8")

        def parse_msgspec():
            return msgspec.json.decode(payload)

        result = benchmark(parse_msgspec)
        assert result["model"] == "gpt-4"
        assert len(result["choices"]) == 60

    def test_large_payload_stdlib_parsing(self, benchmark, response_payloads):
        """Benchmark stdlib json parsing for large payloads (~64KB)."""
        payload = response_payloads["large"]

        def parse_stdlib():
            return json.loads(payload)

        result = benchmark(parse_stdlib)
        assert result["model"] == "gpt-4"
        assert len(result["choices"]) == 60

    def test_xlarge_payload_msgspec_parsing(self, benchmark, response_payloads):
        """Benchmark msgspec parsing for extra large payloads (~1MB)."""
        payload = response_payloads["xlarge"].encode("utf-8")

        def parse_msgspec():
            return msgspec.json.decode(payload)

        result = benchmark(parse_msgspec)
        assert result["model"] == "gpt-4"
        assert len(result["choices"][0]["message"]["content"]) > 1000000

    def test_xlarge_payload_stdlib_parsing(self, benchmark, response_payloads):
        """Benchmark stdlib json parsing for extra large payloads (~1MB)."""
        payload = response_payloads["xlarge"]

        def parse_stdlib():
            return json.loads(payload)

        result = benchmark(parse_stdlib)
        assert result["model"] == "gpt-4"
        assert len(result["choices"][0]["message"]["content"]) > 1000000


class TestBatchSerialization:
    """Test serialization performance under batch processing scenarios."""

    def test_batch_call_context_msgspec(self, benchmark):
        """Benchmark batch serialization of CallContext instances with msgspec."""
        contexts = []
        for i in range(100):
            contexts.append(
                CallContext(
                    call_id=uuid4(),
                    branch_id=uuid4(),
                    deadline_s=time.time() + 300.0,
                    capabilities={f"net.out:api.provider{i % 5}.com"},
                    attrs={
                        "batch_id": f"batch_{i}",
                        "priority": i % 3,
                        "metadata": {"key": f"value_{i}"},
                    },
                )
            )

        def batch_serialize_msgspec():
            encoded_batch = [msgspec.encode(ctx) for ctx in contexts]
            decoded_batch = [msgspec.decode(data, type=CallContext) for data in encoded_batch]
            return decoded_batch

        result = benchmark(batch_serialize_msgspec)
        assert len(result) == 100
        assert all(ctx.call_id for ctx in result)

    def test_batch_call_context_pydantic(self, benchmark):
        """Benchmark batch serialization of CallContext instances with Pydantic."""
        contexts = []
        for i in range(100):
            contexts.append(
                PydanticCallContext(
                    call_id=uuid4(),
                    branch_id=uuid4(),
                    deadline_s=time.time() + 300.0,
                    capabilities={f"net.out:api.provider{i % 5}.com"},
                    attrs={
                        "batch_id": f"batch_{i}",
                        "priority": i % 3,
                        "metadata": {"key": f"value_{i}"},
                    },
                )
            )

        def batch_serialize_pydantic():
            json_batch = [ctx.model_dump_json() for ctx in contexts]
            decoded_batch = [PydanticCallContext.model_validate_json(data) for data in json_batch]
            return decoded_batch

        result = benchmark(batch_serialize_pydantic)
        assert len(result) == 100
        assert all(ctx.call_id for ctx in result)


class TestMemoryUsage:
    """Test memory efficiency of msgspec vs alternatives."""

    def test_memory_usage_msgspec_vs_pydantic(self):
        """Compare memory usage between msgspec and Pydantic structures."""
        import sys

        # Create many instances to see memory difference
        msgspec_contexts = []
        pydantic_contexts = []

        base_data = {
            "call_id": uuid4(),
            "branch_id": uuid4(),
            "deadline_s": time.time() + 300.0,
            "capabilities": {"net.out:api.openai.com", "fs.read:/workspace/*"},
            "attrs": {"trace_id": "abc123", "priority": 1},
        }

        # Create 1000 instances of each
        for i in range(1000):
            msgspec_contexts.append(CallContext(**base_data))
            pydantic_contexts.append(PydanticCallContext(**base_data))

        msgspec_size = sys.getsizeof(msgspec_contexts) + sum(
            sys.getsizeof(ctx) for ctx in msgspec_contexts
        )
        pydantic_size = sys.getsizeof(pydantic_contexts) + sum(
            sys.getsizeof(ctx) for ctx in pydantic_contexts
        )

        # msgspec should use significantly less memory
        memory_improvement = (pydantic_size - msgspec_size) / pydantic_size

        # Assert at least 20% memory improvement (conservative estimate)
        assert (
            memory_improvement > 0.2
        ), f"Memory improvement was only {memory_improvement:.1%}, expected > 20%"


class TestValidationPerformance:
    """Test validation performance differences."""

    def test_validation_msgspec(self, benchmark):
        """Benchmark msgspec validation performance."""
        valid_data = {
            "call_id": uuid4(),
            "branch_id": uuid4(),
            "deadline_s": time.time() + 300.0,
            "capabilities": {"net.out:api.openai.com"},
            "attrs": {"priority": 1},
        }

        def validate_msgspec():
            return CallContext(**valid_data)

        result = benchmark(validate_msgspec)
        assert result.call_id == valid_data["call_id"]

    def test_validation_pydantic(self, benchmark):
        """Benchmark Pydantic validation performance."""
        valid_data = {
            "call_id": uuid4(),
            "branch_id": uuid4(),
            "deadline_s": time.time() + 300.0,
            "capabilities": {"net.out:api.openai.com"},
            "attrs": {"priority": 1},
        }

        def validate_pydantic():
            return PydanticCallContext(**valid_data)

        result = benchmark(validate_pydantic)
        assert result.call_id == valid_data["call_id"]


# Benchmark configuration for pytest-benchmark
def pytest_benchmark_group_stats(groups, key):
    """Custom grouping for performance comparison reporting."""
    return groups


# Performance assertions - these validate that msgspec shows measurable improvement
class TestPerformanceAssertions:
    """Validate that msgspec provides measurable performance gains."""

    def test_msgspec_serialization_faster_than_pydantic(self):
        """Integration test ensuring msgspec is consistently faster than Pydantic."""
        import time

        # Create test data
        test_contexts = []
        for i in range(100):
            test_contexts.append(
                {
                    "call_id": uuid4(),
                    "branch_id": uuid4(),
                    "deadline_s": time.time() + 300.0,
                    "capabilities": {f"net.out:api.provider{i % 3}.com"},
                    "attrs": {"batch_id": f"batch_{i}"},
                }
            )

        # Time msgspec serialization
        start_time = time.perf_counter()
        msgspec_results = []
        for data in test_contexts:
            ctx = CallContext(**data)
            encoded = msgspec.encode(ctx)
            decoded = msgspec.decode(encoded, type=CallContext)
            msgspec_results.append(decoded)
        msgspec_time = time.perf_counter() - start_time

        # Time Pydantic serialization
        start_time = time.perf_counter()
        pydantic_results = []
        for data in test_contexts:
            ctx = PydanticCallContext(**data)
            json_str = ctx.model_dump_json()
            decoded = PydanticCallContext.model_validate_json(json_str)
            pydantic_results.append(decoded)
        pydantic_time = time.perf_counter() - start_time

        # Assert msgspec is at least 2x faster (conservative)
        improvement = pydantic_time / msgspec_time
        assert improvement >= 2.0, f"msgspec was only {improvement:.1f}x faster, expected >= 2x"

        # Verify results are equivalent
        assert len(msgspec_results) == len(pydantic_results) == 100
