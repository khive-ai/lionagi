# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Rate-limited API executor for sophisticated call management."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, AsyncIterator
from uuid import UUID

import anyio
import msgspec

from lionagi.ln.concurrency import (
    CapacityLimiter,
    Event,
    Lock,
    TaskGroup,
    create_task_group,
    effective_deadline,
    fail_after,
    get_cancelled_exc_class,
    is_cancelled,
)

from .core import CallContext, Service, ServiceError
from .endpoint import RequestModel

logger = logging.getLogger(__name__)


class CallStatus(Enum):
    """Status of API call execution."""

    PENDING = "pending"
    QUEUED = "queued"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceCall(msgspec.Struct, kw_only=True):
    """Represents a service call with its context and lifecycle using msgspec."""

    id: UUID
    service: Service
    request: RequestModel
    context: CallContext
    status: CallStatus = CallStatus.PENDING
    created_at: float = msgspec.field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: Any = None
    error: Exception | None = None
    token_estimate: int = 0

    # Event for completion notification using lionagi's Event
    _completion_event: Event = msgspec.field(default_factory=Event)

    def mark_queued(self) -> None:
        """Mark call as queued."""
        self.status = CallStatus.QUEUED

    def mark_executing(self) -> None:
        """Mark call as executing."""
        self.status = CallStatus.EXECUTING
        self.started_at = time.time()

    def mark_completed(self, result: Any) -> None:
        """Mark call as completed with result."""
        self.status = CallStatus.COMPLETED
        self.completed_at = time.time()
        self.result = result
        self._completion_event.set()

    def mark_failed(self, error: Exception) -> None:
        """Mark call as failed with error."""
        self.status = CallStatus.FAILED
        self.completed_at = time.time()
        self.error = error
        self._completion_event.set()

    def mark_cancelled(self) -> None:
        """Mark call as cancelled."""
        self.status = CallStatus.CANCELLED
        self.completed_at = time.time()
        self._completion_event.set()

    async def wait_completion(self) -> Any:
        """Wait for call completion and return result."""
        await self._completion_event.wait()

        if self.status == CallStatus.COMPLETED:
            return self.result
        elif self.status == CallStatus.FAILED:
            raise self.error
        elif self.status == CallStatus.CANCELLED:
            raise get_cancelled_exc_class()("Service call was cancelled")
        else:
            raise ServiceError(f"Call completed with unexpected status: {self.status}")


class StreamingCall(msgspec.Struct, kw_only=True):
    """Represents a streaming service call using msgspec."""

    id: UUID
    service: Service
    request: RequestModel
    context: CallContext
    status: CallStatus = CallStatus.PENDING
    created_at: float = msgspec.field(default_factory=time.time)
    started_at: float | None = None
    chunk_count: int = 0
    total_bytes: int = 0

    def mark_executing(self) -> None:
        self.status = CallStatus.EXECUTING
        self.started_at = time.time()


class ExecutorConfig(msgspec.Struct, kw_only=True):
    """Configuration for the rate-limited executor using msgspec."""

    queue_capacity: int = 100
    capacity_refresh_time: float = 60.0
    interval: float | None = None
    limit_requests: int | None = None
    limit_tokens: int | None = None
    concurrency_limit: int | None = None


class RateLimitedExecutor:
    """Sophisticated rate-limited executor for API calls with structured concurrency.

    Features:
    - Token and request rate limiting with proper locking
    - Memory object streams for async-native queuing
    - Structured concurrency with TaskGroup lifecycle management
    - Concurrency control for streaming calls
    - Statistics and monitoring
    - Graceful shutdown and cleanup
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config

        # Queue management using memory object streams (async-native)
        send_stream, receive_stream = anyio.create_memory_object_stream[ServiceCall](
            max_buffer_size=config.queue_capacity
        )
        self._queue_send: anyio.abc.ObjectSendStream[ServiceCall] = send_stream
        self._queue_receive: anyio.abc.ObjectReceiveStream[ServiceCall] = receive_stream
        
        self.active_calls: dict[UUID, ServiceCall] = {}
        self.completed_calls: dict[UUID, ServiceCall] = {}  # History for debugging

        # Rate limiting state with proper locking
        self._rate_lock = Lock()
        self.request_count = 0
        self.token_count = 0
        self.last_refresh = time.time()

        # Concurrency control
        self.concurrency_limiter = None
        if config.concurrency_limit:
            self.concurrency_limiter = CapacityLimiter(config.concurrency_limit)

        # Processing state with structured concurrency
        self._task_group: TaskGroup | None = None
        self._shutdown_event = Event()
        self._stats = {
            "calls_queued": 0,
            "calls_completed": 0,
            "calls_failed": 0,
            "calls_cancelled": 0,
            "total_tokens": 0,
            "queue_wait_times": [],
        }

    async def start(self) -> None:
        """Start the executor processor with structured concurrency."""
        if self._task_group is None:
            # Create task group and start processor
            self._task_group = await create_task_group().__aenter__()
            self._task_group.start_soon(self._processor_loop)
            logger.info("RateLimitedExecutor started with structured concurrency")

    async def stop(self) -> None:
        """Stop the executor with proper cleanup."""
        self._shutdown_event.set()
        
        # Close the queue send stream to signal no more items
        await self._queue_send.aclose()

        # Wait for task group to finish
        if self._task_group:
            try:
                await self._task_group.aclose()
            except Exception:
                pass

        # Cancel any remaining active calls
        for call in self.active_calls.values():
            call.mark_cancelled()

        logger.info(f"RateLimitedExecutor stopped. Stats: {self._stats}")

    async def submit_call(
        self, service: Service, request: RequestModel, context: CallContext
    ) -> ServiceCall:
        """Submit a service call for execution."""
        from uuid import uuid4

        call = ServiceCall(
            id=uuid4(),
            service=service,
            request=request,
            context=context,
            token_estimate=self._estimate_tokens(request),
        )

        # Add to queue using memory object stream
        try:
            self._queue_send.send_nowait(call)  # Will raise if at capacity
            call.mark_queued()
            self._stats["calls_queued"] += 1
            logger.debug(f"Call {call.id} queued")
        except anyio.WouldBlock:
            raise ServiceError("Executor queue at capacity")

        # Start processor if not running
        await self.start()

        return call

    async def submit_stream(
        self, service: Service, request: RequestModel, context: CallContext
    ) -> AsyncIterator[Any]:
        """Submit a streaming service call."""
        from uuid import uuid4

        call = StreamingCall(
            id=uuid4(),
            service=service,
            request=request,
            context=context,
        )

        call.mark_executing()

        try:
            if self.concurrency_limiter:
                async with self.concurrency_limiter:
                    async for chunk in service.stream(request, ctx=context):
                        call.chunk_count += 1
                        if isinstance(chunk, (str, bytes)):
                            call.total_bytes += len(chunk)
                        yield chunk
            else:
                async for chunk in service.stream(request, ctx=context):
                    call.chunk_count += 1
                    if isinstance(chunk, (str, bytes)):
                        call.total_bytes += len(chunk)
                    yield chunk

            logger.debug(
                f"Stream {call.id} completed. Chunks: {call.chunk_count}, Bytes: {call.total_bytes}"
            )

        except Exception as e:
            logger.error(f"Stream {call.id} failed: {e}")
            raise

    async def _processor_loop(self) -> None:
        """Main processor loop using async streams - no polling!"""
        async with create_task_group() as tg:
            # Start refresh task
            tg.start_soon(self._refresh_limits_loop)
            
            # Process calls as they arrive - no polling
            async with self._queue_receive:
                async for call in self._queue_receive:
                    if self._shutdown_event.is_set():
                        break
                    
                    # Wait until we can process this call
                    await self._wait_for_capacity(call)
                    
                    # Process call in task group
                    self.active_calls[call.id] = call
                    tg.start_soon(self._execute_call, call)
                    
                    # Periodically cleanup completed
                    if len(self.completed_calls) > 100:
                        tg.start_soon(self._cleanup_completed)

    async def _refresh_limits_loop(self) -> None:
        """Continuously refresh rate limits at intervals."""
        while not self._shutdown_event.is_set():
            await anyio.sleep(self.config.capacity_refresh_time)
            async with self._rate_lock:
                self.request_count = 0
                self.token_count = 0
                self.last_refresh = time.time()
                logger.debug("Rate limits refreshed")

    async def _wait_for_capacity(self, call: ServiceCall) -> None:
        """Wait until we have capacity to process a call."""
        while True:
            async with self._rate_lock:
                if self._can_process_call(call):
                    # Reserve capacity
                    self.request_count += 1
                    self.token_count += call.token_estimate
                    return
            
            # Wait a bit before checking again
            await anyio.sleep(0.1)

    def _can_process_call(self, call: ServiceCall) -> bool:
        """Check if we can process a call given current rate limits.
        
        Must be called while holding _rate_lock.
        """
        if self.config.limit_requests and self.request_count >= self.config.limit_requests:
            return False

        if (
            self.config.limit_tokens
            and (self.token_count + call.token_estimate) > self.config.limit_tokens
        ):
            return False

        return True

    async def _execute_call(self, call: ServiceCall) -> None:
        """Execute a single service call."""
        call.mark_executing()

        try:
            # Execute the call (counters already updated in _wait_for_capacity)
            result = await call.service.call(call.request, ctx=call.context)

            # Mark as completed
            call.mark_completed(result)
            self._stats["calls_completed"] += 1

            # Calculate queue wait time
            if call.started_at:
                wait_time = call.started_at - call.created_at
                self._stats["queue_wait_times"].append(wait_time)

            logger.debug(f"Call {call.id} completed successfully")

        except Exception as e:
            call.mark_failed(e)
            self._stats["calls_failed"] += 1
            logger.error(f"Call {call.id} failed: {e}")

        finally:
            # Move from active to completed
            if call.id in self.active_calls:
                del self.active_calls[call.id]
                self.completed_calls[call.id] = call

    async def _cleanup_completed(self) -> None:
        """Clean up old completed calls to prevent memory leaks."""
        if len(self.completed_calls) > 1000:  # Keep last 1000 for debugging
            # Remove oldest 500 calls
            oldest_ids = sorted(
                self.completed_calls.keys(), key=lambda k: self.completed_calls[k].completed_at or 0
            )[:500]

            for call_id in oldest_ids:
                del self.completed_calls[call_id]

            logger.debug(f"Cleaned up {len(oldest_ids)} completed calls")

    def _estimate_tokens(self, request: RequestModel) -> int:
        """Estimate token usage for a request."""
        # Simple heuristic - could be improved with actual tokenizers
        if hasattr(request, "messages") and request.messages:
            total_chars = sum(
                len(msg.get("content", ""))
                for msg in request.messages
                if isinstance(msg.get("content"), str)
            )
            return total_chars // 4  # Rough chars-to-tokens conversion

        return 100  # Default estimate

    @property
    def stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        stats = self._stats.copy()
        stats.update(
            {
                "active_calls": len(self.active_calls),
                "completed_calls": len(self.completed_calls),
                "request_count": self.request_count,
                "token_count": self.token_count,
            }
        )

        if stats["queue_wait_times"]:
            stats["avg_queue_wait"] = sum(stats["queue_wait_times"]) / len(
                stats["queue_wait_times"]
            )

        return stats
