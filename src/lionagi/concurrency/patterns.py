"""Common concurrency patterns for structured concurrency."""

import math
from collections.abc import Awaitable
from types import TracebackType
from typing import Any, Callable, Optional, TypeVar

import anyio

from pynector.concurrency.cancel import move_on_after
from pynector.concurrency.primitives import CapacityLimiter, Lock
from pynector.concurrency.task import create_task_group

T = TypeVar("T")
R = TypeVar("R")
Response = TypeVar("Response")


class ConnectionPool:
    """A pool of reusable connections."""

    def __init__(
        self, max_connections: int, connection_factory: Callable[[], Awaitable[T]]
    ):
        """Initialize a new connection pool.

        Args:
            max_connections: The maximum number of connections in the pool
            connection_factory: A factory function that creates new connections
        """
        self._connection_factory = connection_factory
        self._limiter = CapacityLimiter(max_connections)
        self._connections: list[T] = []
        self._lock = Lock()

    async def acquire(self) -> T:
        """Acquire a connection from the pool.

        Returns:
            A connection from the pool, or a new connection if the pool is empty.
        """
        async with self._limiter:
            async with self._lock:
                if self._connections:
                    return self._connections.pop()

            # No connections available, create a new one
            return await self._connection_factory()

    async def release(self, connection: T) -> None:
        """Release a connection back to the pool.

        Args:
            connection: The connection to release
        """
        async with self._lock:
            self._connections.append(connection)

    async def __aenter__(self) -> "ConnectionPool":
        """Enter the connection pool context.

        Returns:
            The connection pool instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the connection pool context, closing all connections."""
        async with self._lock:
            for connection in self._connections:
                if hasattr(connection, "close"):
                    await connection.close()
                elif hasattr(connection, "disconnect"):
                    await connection.disconnect()
            self._connections.clear()


async def parallel_requests(
    urls: list[str],
    fetch_func: Callable[[str], Awaitable[Response]],
    max_concurrency: int = 10,
) -> list[Response]:
    """Fetch multiple URLs in parallel with limited concurrency.

    Args:
        urls: The URLs to fetch
        fetch_func: The function to use for fetching
        max_concurrency: The maximum number of concurrent requests

    Returns:
        A list of responses in the same order as the URLs
    """
    limiter = CapacityLimiter(max_concurrency)
    results: list[Optional[Response]] = [None] * len(urls)
    exceptions: list[Optional[Exception]] = [None] * len(urls)

    async def fetch_with_limit(index: int, url: str) -> None:
        async with limiter:
            try:
                results[index] = await fetch_func(url)
            except Exception as exc:
                exceptions[index] = exc

    async with create_task_group() as tg:
        for i, url in enumerate(urls):
            await tg.start_soon(fetch_with_limit, i, url)

    # Check for exceptions
    for i, exc in enumerate(exceptions):
        if exc is not None:
            raise exc

    return results  # type: ignore


async def retry_with_timeout(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    timeout: float = 5.0,
    retry_exceptions: Optional[list[type[Exception]]] = None,
    **kwargs: Any,
) -> T:
    """Execute a function with retry logic and timeout.

    Args:
        func: The function to call
        *args: Positional arguments to pass to the function
        max_retries: The maximum number of retry attempts
        timeout: The timeout for each attempt in seconds
        retry_exceptions: List of exception types to retry on, or None to retry on any exception
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The return value of the function

    Raises:
        TimeoutError: If all retry attempts time out
        Exception: If the function raises an exception after all retry attempts
    """
    retry_exceptions = retry_exceptions or [Exception]
    last_exception = None

    for attempt in range(max_retries):
        try:
            timed_out = False
            with move_on_after(timeout) as scope:
                result = await func(*args, **kwargs)
                if not scope.cancelled_caught:
                    return result
                timed_out = True

            # If we get here, the operation timed out
            if timed_out:
                if attempt == max_retries - 1:
                    raise TimeoutError(
                        f"Operation timed out after {max_retries} attempts"
                    )

                # Wait before retrying (exponential backoff)
                await anyio.sleep(2**attempt)

        except tuple(retry_exceptions) as exc:
            last_exception = exc
            if attempt == max_retries - 1:
                raise

            # Wait before retrying (exponential backoff)
            await anyio.sleep(2**attempt)

    # This should never be reached, but makes the type checker happy
    if last_exception:
        raise last_exception
    raise RuntimeError("Unreachable code")


class WorkerPool:
    """A pool of worker tasks that process items from a queue."""

    def __init__(self, num_workers: int, worker_func: Callable[[Any], Awaitable[None]]):
        """Initialize a new worker pool.

        Args:
            num_workers: The number of worker tasks to create
            worker_func: The function that each worker will run
        """
        self._num_workers = num_workers
        self._worker_func = worker_func
        self._queue = anyio.create_memory_object_stream(math.inf)
        self._task_group = None

    async def start(self) -> None:
        """Start the worker pool."""
        if self._task_group is not None:
            raise RuntimeError("Worker pool already started")

        self._task_group = create_task_group()

        async with self._task_group as tg:
            for _ in range(self._num_workers):
                tg.start_soon(self._worker_loop)

    async def stop(self) -> None:
        """Stop the worker pool."""
        if self._task_group is None:
            return

        # Signal workers to stop
        for _ in range(self._num_workers):
            await self._queue[0].send(None)

        # Wait for workers to finish
        await self._task_group.__aexit__(None, None, None)
        self._task_group = None

    async def submit(self, item: Any) -> None:
        """Submit an item to be processed by a worker.

        Args:
            item: The item to process
        """
        if self._task_group is None:
            raise RuntimeError("Worker pool not started")

        await self._queue[0].send(item)

    async def _worker_loop(self) -> None:
        """The main loop for each worker task."""
        while True:
            try:
                item = await self._queue[1].receive()

                # None is a signal to stop
                if item is None:
                    break

                try:
                    await self._worker_func(item)
                except Exception as exc:
                    # Log the exception but keep the worker running
                    print(f"Worker error: {exc}")
            except anyio.EndOfStream:
                break
