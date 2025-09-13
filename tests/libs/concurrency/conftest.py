"""Enhanced fixtures for lionagi concurrency testing.

Provides backend-neutral testing utilities, performance monitoring,
and resource leak detection for comprehensive test coverage.
"""

import asyncio
import time
import tracemalloc
import weakref
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import anyio
import pytest


def pytest_generate_tests(metafunc):
    """Parametrize async tests to run on both asyncio and trio backends."""
    # Only parametrize if the test doesn't already use @pytest.mark.anyio
    # (which handles backend parametrization automatically)
    if "anyio_backend" in metafunc.fixturenames:
        # Check if the test has an anyio marker
        has_anyio_marker = False
        for mark in metafunc.definition.iter_markers():
            if mark.name == "anyio":
                has_anyio_marker = True
                break

        if not has_anyio_marker:
            metafunc.parametrize("anyio_backend", ["asyncio", "trio"])


@pytest.fixture
def deadline():
    """Default timeout for tests to prevent hangs."""
    return 10.0


@pytest.fixture
def cancel_guard(deadline):
    """Context manager that ensures tests don't exceed deadline."""

    @asynccontextmanager
    async def _guard():
        with anyio.move_on_after(deadline) as scope:
            yield scope
            assert (
                not scope.cancel_called
            ), f"Test exceeded {deadline}s deadline"

    return _guard


@pytest.fixture
def mem_tracer():
    """Memory profiling context manager using tracemalloc."""

    class Tracer:
        def __init__(self):
            self.start_snapshot = None
            self.end_snapshot = None
            self.stats = None

        def __enter__(self):
            tracemalloc.start()
            self.start_snapshot = tracemalloc.take_snapshot()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_snapshot = tracemalloc.take_snapshot()
            if self.start_snapshot:
                self.stats = self.end_snapshot.compare_to(
                    self.start_snapshot, "lineno"
                )
            tracemalloc.stop()

        def total_kib(self):
            """Get total memory difference in KiB."""
            if not self.stats:
                return 0
            return sum(s.size_diff for s in self.stats) / 1024.0

        def peak_kib(self):
            """Get peak memory usage in KiB."""
            if not self.end_snapshot:
                return 0
            return (
                sum(s.size for s in self.end_snapshot.statistics("lineno"))
                / 1024.0
            )

    return Tracer


@pytest.fixture
def monotonic():
    """High-resolution timer for performance measurements."""
    return time.perf_counter


@pytest.fixture
def resource_tracker():
    """Track resources for leak detection."""

    class ResourceTracker:
        def __init__(self):
            self.resources = []
            self.weak_refs = []

        def track(self, resource: Any, name: str = None):
            """Track a resource for cleanup verification."""
            self.resources.append(
                (resource, name or f"resource_{len(self.resources)}")
            )
            self.weak_refs.append(weakref.ref(resource))

        def verify_cleanup(self):
            """Verify all tracked resources have been cleaned up."""
            import gc

            gc.collect()  # Force garbage collection

            alive_resources = []
            for (resource, name), weak_ref in zip(
                self.resources, self.weak_refs
            ):
                if weak_ref() is not None:
                    alive_resources.append(name)

            if alive_resources:
                pytest.fail(f"Resources not cleaned up: {alive_resources}")

        def clear(self):
            """Clear tracked resources."""
            self.resources.clear()
            self.weak_refs.clear()

    return ResourceTracker()


@pytest.fixture
async def concurrency_probe():
    """Utility for tracking concurrent execution."""

    class ConcurrencyProbe:
        def __init__(self):
            self.current_running = 0
            self.max_running = 0
            self.lock = anyio.Lock()

        async def enter_task(self):
            """Call when a task starts."""
            async with self.lock:
                self.current_running += 1
                self.max_running = max(self.max_running, self.current_running)

        async def exit_task(self):
            """Call when a task finishes."""
            async with self.lock:
                self.current_running -= 1

        @asynccontextmanager
        async def track_task(self):
            """Context manager to track task lifecycle."""
            await self.enter_task()
            try:
                yield
            finally:
                await self.exit_task()

        def reset(self):
            """Reset counters."""
            self.current_running = 0
            self.max_running = 0

    return ConcurrencyProbe()


@pytest.fixture
def task_factory():
    """Factory for creating test tasks with various behaviors."""

    class TaskFactory:
        @staticmethod
        async def io_task(item, delay=0.001):
            """Simulate I/O bound work."""
            await anyio.sleep(delay)
            return item * 2

        @staticmethod
        async def failing_task(item, fail_on=None):
            """Task that fails on specific values."""
            if fail_on is not None and item in fail_on:
                raise ValueError(f"Task failed on {item}")
            await anyio.sleep(0.001)
            return item * 3

        @staticmethod
        async def cancellable_task(item, resource_tracker=None):
            """Task that can be cancelled and tracks cleanup."""
            resource = f"resource_{item}"
            if resource_tracker:
                resource_tracker.append(resource)

            try:
                await anyio.sleep(0.1)  # Long enough to be cancelled
                return item
            finally:
                # Cleanup resource
                if resource_tracker and resource in resource_tracker:
                    resource_tracker.remove(resource)

        @staticmethod
        def sync_task(item, multiplier=2):
            """Synchronous task for thread pool testing."""
            return item * multiplier

        @staticmethod
        def sync_kwargs_task(a, b=0, *, c=0):
            """Sync task with complex kwargs (tests the kwargs fix)."""
            return a + b + c

    return TaskFactory()
