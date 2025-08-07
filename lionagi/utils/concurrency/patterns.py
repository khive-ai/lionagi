"""Common concurrency patterns for structured concurrency."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, TypeVar

import anyio

from .cancel import move_on_after
from .primitives import CapacityLimiter, Lock
from .resource_tracker import track_resource, untrack_resource
from .task import create_task_group

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")
Response = TypeVar("Response")


class ConnectionPool:
    """A pool of reusable connections."""

    def __init__(
        self,
        max_connections: int,
        connection_factory: Callable[[], Awaitable[T]],
    ):
        """Initialize a new connection pool."""
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if not callable(connection_factory):
            raise ValueError("connection_factory must be callable")

        self._connection_factory = connection_factory
        self._limiter = CapacityLimiter(max_connections)
        self._connections: list[T] = []
        self._lock = Lock()

        track_resource(self, f"ConnectionPool-{id(self)}", "ConnectionPool")

    def __del__(self):
        """Clean up resource tracking."""
        try:
            untrack_resource(self)
        except Exception:
            pass

    async def acquire(self) -> T:
        """Acquire a connection from the pool."""
        await self._limiter.acquire()

        try:
            async with self._lock:
                if self._connections:
                    return self._connections.pop()

            # No pooled connection available, create new one
            return await self._connection_factory()
        except Exception:
            self._limiter.release()
            raise

    async def release(self, connection: T) -> None:
        """Release a connection back to the pool."""
        try:
            async with self._lock:
                self._connections.append(connection)
        finally:
            self._limiter.release()

    async def __aenter__(self) -> ConnectionPool[T]:
        """Enter the connection pool context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the connection pool context."""
        # Clean up any remaining connections
        async with self._lock:
            self._connections.clear()


async def parallel_requests(
    inputs: list[str],
    func: Callable[[str], Awaitable[Response]],
    max_concurrency: int = 10,
) -> list[Response]:
    """Execute requests in parallel with controlled concurrency.

    Args:
        inputs: List of inputs
        fetch_func: Async function
        max_concurrency: Maximum number of concurrent requests

    Returns:
        List of responses in the same order as inputs
    """
    if not inputs:
        return []

    results: list[Response | None] = [None] * len(inputs)

    async def bounded_fetch(
        semaphore: anyio.Semaphore, idx: int, url: str
    ) -> None:
        async with semaphore:
            results[idx] = await func(url)

    try:
        async with create_task_group() as tg:
            semaphore = anyio.Semaphore(max_concurrency)

            for i, inp in enumerate(inputs):
                await tg.start_soon(bounded_fetch, semaphore, i, inp)
    except BaseException as e:
        # Re-raise the first exception directly instead of ExceptionGroup
        if hasattr(e, "exceptions") and e.exceptions:
            raise e.exceptions[0]
        else:
            raise

    return results  # type: ignore


async def retry_with_timeout(
    func: Callable[[], Awaitable[T]],
    max_retries: int = 3,
    timeout: float = 30.0,
    backoff_factor: float = 1.0,
) -> T:
    """Retry an async function with exponential backoff and timeout.

    Args:
        func: The async function to retry
        max_retries: Maximum number of retries
        timeout: Timeout for each attempt
        backoff_factor: Multiplier for exponential backoff

    Returns:
        The result of the successful function call

    Raises:
        Exception: The last exception raised by the function
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            with move_on_after(timeout) as cancel_scope:
                result = await func()
                if not cancel_scope.cancelled_caught:
                    return result
                else:
                    raise TimeoutError(f"Function timed out after {timeout}s")
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                delay = backoff_factor * (2**attempt)
                await anyio.sleep(delay)
            continue

    if last_exception:
        raise last_exception
    else:
        raise RuntimeError("Retry failed without capturing exception")


class WorkerPool:
    """A pool of worker tasks that process items from a queue."""

    def __init__(
        self, num_workers: int, worker_func: Callable[[Any], Awaitable[None]]
    ):
        """Initialize a new worker pool."""
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if not callable(worker_func):
            raise ValueError("worker_func must be callable")

        self._num_workers = num_workers
        self._worker_func = worker_func
        self._queue = anyio.create_memory_object_stream(1000)
        self._task_group = None

        track_resource(self, f"WorkerPool-{id(self)}", "WorkerPool")

    def __del__(self):
        """Clean up resource tracking."""
        try:
            untrack_resource(self)
        except Exception:
            pass

    async def start(self) -> None:
        """Start the worker pool."""
        if self._task_group is not None:
            raise RuntimeError("Worker pool is already started")

        self._task_group = create_task_group()
        await self._task_group.__aenter__()

        # Start worker tasks
        for i in range(self._num_workers):
            await self._task_group.start_soon(self._worker_loop)

    async def stop(self) -> None:
        """Stop the worker pool."""
        if self._task_group is None:
            return

        # Close the queue to signal workers to stop
        await self._queue[0].aclose()

        # Wait for all workers to finish
        try:
            await self._task_group.__aexit__(None, None, None)
        finally:
            self._task_group = None

    async def submit(self, item: Any) -> None:
        """Submit an item for processing."""
        if self._task_group is None:
            raise RuntimeError("Worker pool is not started")
        await self._queue[0].send(item)

    async def _worker_loop(self) -> None:
        """Main loop for worker tasks."""
        try:
            async with self._queue[1]:
                async for item in self._queue[1]:
                    try:
                        await self._worker_func(item)
                    except Exception as e:
                        logger.error(f"Worker error processing item: {e}")
        except anyio.ClosedResourceError:
            # Queue was closed, worker should exit gracefully
            pass

    async def __aenter__(self) -> WorkerPool:
        """Enter the worker pool context."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the worker pool context."""
        await self.stop()
