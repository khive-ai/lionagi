# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Resilience middleware for retries, circuit breakers, and deadline enforcement."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import Enum
from typing import Any

import anyio
import msgspec

from ..errors import NonRetryableError, RetryableError, ServiceError, TimeoutError
from ..ln import Lock
from .core import CallContext
from .endpoint import RequestModel

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryConfig(msgspec.Struct, kw_only=True):
    """Retry policy configuration using msgspec for v1 performance."""

    max_attempts: int = 3
    base_delay: float = 0.1  # seconds
    max_delay: float = 10.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True


class CircuitBreakerConfig(msgspec.Struct, kw_only=True):
    """Circuit breaker configuration using msgspec for v1 performance."""

    failure_threshold: int = 5  # failures to trip breaker
    success_threshold: int = 2  # successes to close breaker
    timeout: float = 60.0  # seconds before half-open retry
    failure_ratio_threshold: float = 0.5  # ratio of failures to trip


class CircuitBreaker:
    """Circuit breaker for fast-fail behavior with proper concurrency protection."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.total_requests = 0
        self._lock = Lock()  # Protect state modifications using lionagi's Lock

    async def call(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Execute operation through circuit breaker with proper state protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.config.timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise ServiceError("Circuit breaker is OPEN")

            self.total_requests += 1

        try:
            result = await operation()
            await self._on_success()
            return result
        except (RetryableError, TimeoutError) as e:
            await self._on_failure()
            raise
        except NonRetryableError:
            # Don't count non-retryable errors toward circuit breaking
            raise

    async def _on_success(self):
        """Handle successful operation with concurrency protection."""
        async with self._lock:
            self.failure_count = 0

            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logger.info("Circuit breaker closed after successful recovery")

    async def _on_failure(self):
        """Handle failed operation with concurrency protection."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning("Circuit breaker reopened after failure in HALF_OPEN")
            elif (
                self.state == CircuitState.CLOSED
                and self.failure_count >= self.config.failure_threshold
            ):
                self.state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryMW:
    """Retry middleware with exponential backoff and jitter.

    Only retries on retryable errors (network, 5xx, 429). Respects
    call context deadline to prevent retry storms.
    """

    def __init__(self, config: RetryConfig):
        self.config = config

    async def __call__(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Apply retry logic to call operations."""
        last_error = None

        for attempt in range(self.config.max_attempts):
            try:
                return await next_call()
            except NonRetryableError:
                # Don't retry client errors (except 429 which is RetryableError)
                raise
            except (RetryableError, TimeoutError) as e:
                last_error = e

                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break

                # Check if we have time left for retry
                delay = self._compute_delay(attempt)
                if not self._can_retry(ctx, delay):
                    logger.warning(
                        f"Skipping retry due to deadline: remaining={ctx.remaining_time}s, delay={delay}s",
                        extra={"call_id": str(ctx.call_id)},
                    )
                    break

                logger.info(
                    f"Retrying after {delay:.2f}s (attempt {attempt + 1}/{self.config.max_attempts})",
                    extra={
                        "call_id": str(ctx.call_id),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )

                await anyio.sleep(delay)

        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise ServiceError(f"All {self.config.max_attempts} retry attempts failed")

    async def stream(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_stream: Callable[[], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Apply retry logic to streaming operations.

        Note: Streaming retries are more complex because we may have already
        started yielding chunks. For now, we only retry if the stream fails
        before yielding any chunks.
        """
        last_error = None

        for attempt in range(self.config.max_attempts):
            try:
                chunk_yielded = False
                async for chunk in next_stream():
                    chunk_yielded = True
                    yield chunk
                return  # Stream completed successfully

            except NonRetryableError:
                raise
            except (RetryableError, TimeoutError) as e:
                last_error = e

                # If we already yielded chunks, we can't retry
                if chunk_yielded:
                    logger.warning(
                        "Stream failed after yielding chunks - cannot retry",
                        extra={"call_id": str(ctx.call_id), "error": str(e)},
                    )
                    raise

                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break

                # Check if we have time for retry
                delay = self._compute_delay(attempt)
                if not self._can_retry(ctx, delay):
                    break

                logger.info(
                    f"Retrying stream after {delay:.2f}s (attempt {attempt + 1}/{self.config.max_attempts})",
                    extra={"call_id": str(ctx.call_id), "error": str(e)},
                )

                await anyio.sleep(delay)

        if last_error:
            raise last_error
        else:
            raise ServiceError(f"All {self.config.max_attempts} stream retry attempts failed")

    def _compute_delay(self, attempt: int) -> float:
        """Compute retry delay with exponential backoff and jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base**attempt),
            self.config.max_delay,
        )

        if self.config.jitter:
            # Full jitter: delay * random(0, 1)
            delay *= random.random()

        return delay

    def _can_retry(self, ctx: CallContext, delay: float) -> bool:
        """Check if we have enough time left to retry."""
        if ctx.deadline_s is None:
            return True

        remaining = ctx.remaining_time
        if remaining is None:
            return True

        # Leave some buffer time for the actual retry
        return remaining > (delay + 1.0)


class CircuitBreakerMW:
    """Circuit breaker middleware for fast-fail behavior."""

    def __init__(self, config: CircuitBreakerConfig):
        self.breaker = CircuitBreaker(config)

    async def __call__(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Apply circuit breaker to call operations."""
        return await self.breaker.call(next_call)

    async def stream(
        self,
        req: RequestModel,
        ctx: CallContext,
        next_stream: Callable[[], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:
        """Apply circuit breaker to streaming operations with pass-through semantics.

        No buffering - streams pass through while tracking state for future calls.
        """
        # Check breaker state before starting stream
        async with self.breaker._lock:
            if self.breaker.state == CircuitState.OPEN:
                if time.time() - self.breaker.last_failure_time > self.breaker.config.timeout:
                    self.breaker.state = CircuitState.HALF_OPEN
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise ServiceError("Circuit breaker is OPEN")

            self.breaker.total_requests += 1

        success = False
        first_chunk_seen = False

        try:
            async for chunk in next_stream():
                # First successful chunk in HALF_OPEN proves recovery
                if not first_chunk_seen:
                    first_chunk_seen = True
                    if self.breaker.state == CircuitState.HALF_OPEN:
                        await self.breaker._on_success()

                yield chunk  # Pass through immediately - no buffering

            success = True

        except (RetryableError, TimeoutError) as e:
            await self.breaker._on_failure()
            raise
        except NonRetryableError:
            # Don't count non-retryable errors toward circuit breaking
            raise
        finally:
            if success:
                await self.breaker._on_success()


# Combined resilience middleware factory
def create_resilience_mw(
    *,
    retry_config: RetryConfig | None = None,
    circuit_config: CircuitBreakerConfig | None = None,
) -> tuple[Any, ...]:
    """Create resilience middleware stack.

    Returns tuple of middleware in correct order:
    1. Circuit breaker (outermost - fast fail)
    2. Retry logic (inner - handles retryable failures)
    """
    middleware = []

    if circuit_config:
        middleware.append(CircuitBreakerMW(circuit_config))

    if retry_config:
        middleware.append(RetryMW(retry_config))

    return tuple(middleware)
