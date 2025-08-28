# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration and fixtures for lionagi services integration tests.

Provides shared fixtures, configuration, and test utilities for comprehensive
Agent Kernel readiness testing through iModel integration.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import anyio
import pytest
from anyio.testing import MockClock

from lionagi.services import CallContext, ChatRequestModel, RequestModel, Service


# Pytest Configuration
def pytest_configure(config):
    """Configure pytest for services tests."""
    # Register custom markers
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "critical_flaw: tests for critical flaws")
    config.addinivalue_line("markers", "observability: observability and logging tests")
    config.addinivalue_line("markers", "performance: performance benchmark tests")


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure AnyIO backend for structured concurrency tests."""
    return "asyncio"


@pytest.fixture
def mock_clock():
    """Provide mock clock for deterministic time testing."""
    return MockClock()


@pytest.fixture
async def mock_environment():
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key-123",
        "ANTHROPIC_API_KEY": "test-anthropic-key-456",
        "GROQ_API_KEY": "test-groq-key-789",
    }

    with patch.dict("os.environ", env_vars):
        yield env_vars


@pytest.fixture
def temp_log_file():
    """Create temporary log file for log capture testing."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as f:
        yield f.name


@pytest.fixture
def structured_logger(temp_log_file):
    """Create structured logger for observability testing."""
    logger = logging.getLogger("test.structured")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Add file handler with JSON formatting
    handler = logging.FileHandler(temp_log_file)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    yield logger

    # Cleanup
    logger.handlers.clear()


# Service Fixtures


class TestService(Service):
    """Configurable test service for integration testing."""

    name = "test-service"
    requires = {"test:capability"}

    def __init__(
        self,
        delay_s: float = 0.01,
        should_fail: bool = False,
        fail_with: type[Exception] = Exception,
        fail_after_calls: int = 0,
    ):
        self.delay_s = delay_s
        self.should_fail = should_fail
        self.fail_with = fail_with
        self.fail_after_calls = fail_after_calls
        self.call_count = 0
        self.stream_count = 0
        self.last_request = None
        self.last_context = None

    async def call(self, req: RequestModel, *, ctx: CallContext) -> dict[str, Any]:
        self.call_count += 1
        self.last_request = req
        self.last_context = ctx

        if self.delay_s > 0:
            await anyio.sleep(self.delay_s)

        if self.should_fail and self.call_count > self.fail_after_calls:
            raise self.fail_with(f"Test service failure on call {self.call_count}")

        return {
            "service": self.name,
            "call_id": str(ctx.call_id),
            "call_count": self.call_count,
            "model": getattr(req, "model", "test-model"),
            "success": True,
        }

    async def stream(self, req: RequestModel, *, ctx: CallContext) -> AsyncIterator[dict[str, Any]]:
        self.stream_count += 1
        self.last_request = req
        self.last_context = ctx

        chunk_count = getattr(req, "max_tokens", 3)

        for i in range(chunk_count):
            if self.delay_s > 0:
                await anyio.sleep(self.delay_s)

            if self.should_fail and i == (chunk_count - 1):  # Fail on last chunk
                raise self.fail_with(f"Test stream failure on chunk {i}")

            yield {
                "chunk": i,
                "service": self.name,
                "call_id": str(ctx.call_id),
                "stream_count": self.stream_count,
                "total_chunks": chunk_count,
            }


@pytest.fixture
def test_service():
    """Basic test service fixture."""
    return TestService()


@pytest.fixture
def slow_service():
    """Slow service for timeout testing."""
    return TestService(delay_s=1.0)


@pytest.fixture
def failing_service():
    """Service that always fails."""
    return TestService(should_fail=True, fail_with=Exception)


@pytest.fixture
def flaky_service():
    """Service that fails intermittently."""
    return TestService(should_fail=True, fail_after_calls=2, fail_with=Exception)


# Context and Request Fixtures


@pytest.fixture
def test_context():
    """Basic test context with common capabilities."""
    return CallContext.with_timeout(
        branch_id=uuid4(),
        timeout_s=30.0,
        capabilities={"test:capability"},
        user_id="test-user",
        trace_id="test-trace",
    )


@pytest.fixture
def privileged_context():
    """Context with elevated capabilities."""
    return CallContext.with_timeout(
        branch_id=uuid4(),
        timeout_s=30.0,
        capabilities={"test:capability", "net.out:*", "data.read:*", "admin:access"},
        admin_user=True,
    )


@pytest.fixture
def unprivileged_context():
    """Context with minimal capabilities."""
    return CallContext.with_timeout(
        branch_id=uuid4(),
        timeout_s=30.0,
        capabilities=set(),  # No capabilities
        user_id="limited-user",
    )


@pytest.fixture
def chat_request():
    """Basic chat request."""
    return ChatRequestModel(
        model="test-model",
        messages=[{"role": "user", "content": "Test message for integration testing"}],
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def streaming_request(chat_request):
    """Chat request configured for streaming."""
    return chat_request.model_copy(update={"stream": True})


# Mock Provider Fixtures


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for provider testing."""
    mock_client = Mock()

    # Mock completion response
    mock_response = Mock()
    mock_response.model = "gpt-4o-mini"
    mock_response.choices = [Mock(message=Mock(role="assistant", content="Test response"))]
    mock_response.usage = Mock(total_tokens=50)

    mock_client.chat.completions.create = Mock(return_value=mock_response)

    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for provider testing."""
    mock_client = Mock()

    # Mock message response
    mock_response = Mock()
    mock_response.id = "msg_test123"
    mock_response.content = [Mock(text="Test Claude response")]
    mock_response.usage = Mock(input_tokens=25, output_tokens=25)

    mock_client.messages.create = Mock(return_value=mock_response)

    return mock_client


# Observability Fixtures


@pytest.fixture
def hook_event_collector():
    """Collect hook events for testing."""
    events = []

    async def collect_hook(event):
        events.append(
            {
                "type": (
                    event.hook_type.value
                    if hasattr(event.hook_type, "value")
                    else str(event.hook_type)
                ),
                "call_id": str(event.call_id),
                "branch_id": str(event.branch_id),
                "service": event.service_name,
                "timestamp": event.timestamp,
                "metadata": event.metadata.copy() if event.metadata else {},
            }
        )

    collect_hook.events = events
    return collect_hook


@pytest.fixture
def metrics_collector():
    """Collect metrics for testing."""
    metrics = []

    def collect_metric(metric_data):
        metrics.append(metric_data.copy())

    collect_metric.metrics = metrics
    return collect_metric


# Async Test Helpers


@pytest.fixture
async def async_test_context():
    """Async context manager for test setup/teardown."""

    class AsyncTestContext:
        def __init__(self):
            self.resources = []

        async def add_resource(self, resource):
            """Add resource for cleanup."""
            self.resources.append(resource)
            return resource

        async def cleanup(self):
            """Clean up all resources."""
            for resource in reversed(self.resources):
                if hasattr(resource, "stop"):
                    await resource.stop()
                elif hasattr(resource, "close"):
                    await resource.close()
                elif hasattr(resource, "__aexit__"):
                    await resource.__aexit__(None, None, None)

    ctx = AsyncTestContext()

    try:
        yield ctx
    finally:
        await ctx.cleanup()


# Performance Testing Fixtures


@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import os
    import time

    import psutil

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.end_memory = None
            self.process = psutil.Process(os.getpid())

        def start(self):
            self.start_time = time.perf_counter()
            self.start_memory = self.process.memory_info().rss

        def stop(self):
            self.end_time = time.perf_counter()
            self.end_memory = self.process.memory_info().rss

        @property
        def duration_s(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None

        @property
        def memory_delta_mb(self):
            if self.start_memory and self.end_memory:
                return (self.end_memory - self.start_memory) / 1024 / 1024
            return None

    return PerformanceMonitor()


# Parametrized Fixtures for Multiple Test Scenarios


@pytest.fixture(
    params=[
        {"provider": "openai", "model": "gpt-4o-mini"},
        {"provider": "anthropic", "model": "claude-3-sonnet-20240229"},
    ]
)
def provider_config(request):
    """Parametrized provider configurations."""
    return request.param


@pytest.fixture(params=[0.01, 0.1, 0.5])
def service_delays(request):
    """Different service delay configurations for performance testing."""
    return request.param


@pytest.fixture(
    params=[
        {"queue_capacity": 10, "limit_requests": 5},
        {"queue_capacity": 50, "limit_requests": 20},
        {"queue_capacity": 100, "limit_requests": None},
    ]
)
def executor_configs(request):
    """Different executor configurations for load testing."""
    return request.param


# Test Markers and Utilities


def pytest_runtest_setup(item):
    """Setup for individual tests."""
    # Mark tests based on name patterns
    if "critical_flaw" in item.nodeid:
        item.add_marker(pytest.mark.critical_flaw)
    if "observability" in item.nodeid or "log" in item.nodeid:
        item.add_marker(pytest.mark.observability)
    if "performance" in item.nodeid or "benchmark" in item.nodeid:
        item.add_marker(pytest.mark.performance)


# Custom Assertions


def assert_valid_call_context(context: CallContext):
    """Assert CallContext has required fields."""
    assert context.call_id is not None
    assert context.branch_id is not None
    assert isinstance(context.capabilities, set)
    assert isinstance(context.attrs, dict)


def assert_service_call_metrics(metrics: dict, expected_status: str = "success"):
    """Assert service call metrics have expected structure."""
    required_fields = ["call_id", "duration_s", "status"]
    for field in required_fields:
        assert field in metrics, f"Missing required metric field: {field}"

    assert metrics["status"] == expected_status
    assert isinstance(metrics["duration_s"], (int, float))
    assert metrics["duration_s"] >= 0


def assert_hook_event_structure(event: dict):
    """Assert hook event has required structure."""
    required_fields = ["type", "call_id", "service", "timestamp"]
    for field in required_fields:
        assert field in event, f"Missing required hook event field: {field}"

    assert isinstance(event["timestamp"], (int, float))
    assert event["timestamp"] > 0
