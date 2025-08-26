# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any

from typing_extensions import Self, override

from lionagi.libs.concurrency import CapacityLimiter, Lock, move_on_after
from lionagi.protocols.types import Executor, Processor

from .connections.api_calling import APICalling

__all__ = (
    "RateLimitedAPIProcessor",
    "RateLimitedAPIExecutor",
)


class RateLimitedAPIProcessor(Processor):
    event_type = APICalling

    def __init__(
        self,
        queue_capacity: int,
        capacity_refresh_time: float,
        interval: float | None = None,
        limit_requests: int = None,
        limit_tokens: int = None,
        concurrency_limit: int | None = None,
    ):
        super().__init__(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            concurrency_limit=concurrency_limit,
        )
        self.limit_tokens = limit_tokens
        self.limit_requests = limit_requests
        self.interval = interval or self.capacity_refresh_time
        self.available_request = self.limit_requests
        self.available_token = self.limit_tokens
        self._rate_limit_replenisher_task: asyncio.Task | None = None
        self._lock = Lock()

        # Use CapacityLimiter for better token management
        if self.limit_tokens:
            self._token_limiter = CapacityLimiter(self.limit_tokens)
        else:
            self._token_limiter = None

        if self.limit_requests:
            self._request_limiter = CapacityLimiter(self.limit_requests)
        else:
            self._request_limiter = None

    async def start_replenishing(self):
        """Start replenishing rate limit capacities at regular intervals."""
        await self.start()
        try:
            while not self.is_stopped():
                await asyncio.sleep(self.interval)

                # Reset capacity limiters to their original values
                if self._request_limiter and self.limit_requests:
                    # Adjust total tokens to reset capacity
                    current_borrowed = self._request_limiter.borrowed_tokens
                    if current_borrowed < self.limit_requests:
                        self._request_limiter.total_tokens = (
                            self.limit_requests
                        )

                if self._token_limiter and self.limit_tokens:
                    # Reset token limiter capacity
                    current_borrowed = self._token_limiter.borrowed_tokens
                    if current_borrowed < self.limit_tokens:
                        self._token_limiter.total_tokens = self.limit_tokens

        except asyncio.CancelledError:
            logging.info("Rate limit replenisher task cancelled.")
        except Exception as e:
            logging.error(f"Error in rate limit replenisher: {e}")

    @override
    async def stop(self) -> None:
        """Stop the replenishment task."""
        if self._rate_limit_replenisher_task:
            self._rate_limit_replenisher_task.cancel()
            await self._rate_limit_replenisher_task
        await super().stop()

    @override
    @classmethod
    async def create(
        cls,
        queue_capacity: int,
        capacity_refresh_time: float,
        interval: float | None = None,
        limit_requests: int = None,
        limit_tokens: int = None,
        concurrency_limit: int | None = None,
    ) -> Self:
        self = cls(
            interval=interval,
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            limit_requests=limit_requests,
            limit_tokens=limit_tokens,
            concurrency_limit=concurrency_limit,
        )
        self._rate_limit_replenisher_task = asyncio.create_task(
            self.start_replenishing()
        )
        return self

    @override
    async def request_permission(
        self, required_tokens: int = None, **kwargs: Any
    ) -> bool:
        # No limits configured, just check queue capacity
        if self._request_limiter is None and self._token_limiter is None:
            return self.queue.qsize() < self.queue_capacity

        # Check request limit
        if self._request_limiter:
            # Try to acquire with timeout
            with move_on_after(0.1) as scope:
                await self._request_limiter.acquire()
                if scope.cancelled_caught:
                    return False

        # Check token limit if required
        if self._token_limiter and required_tokens:
            # For token-based limiting, we need to acquire multiple tokens
            # This is a simplified approach - in production you might want
            # a more sophisticated token bucket algorithm
            if self._token_limiter.available_tokens < required_tokens:
                if self._request_limiter:
                    self._request_limiter.release()
                return False

        return True


class RateLimitedAPIExecutor(Executor):
    processor_type = RateLimitedAPIProcessor

    def __init__(
        self,
        queue_capacity: int,
        capacity_refresh_time: float,
        interval: float | None = None,
        limit_requests: int = None,
        limit_tokens: int = None,
        strict_event_type: bool = False,
        concurrency_limit: int | None = None,
    ):
        config = {
            "queue_capacity": queue_capacity,
            "capacity_refresh_time": capacity_refresh_time,
            "interval": interval,
            "limit_requests": limit_requests,
            "limit_tokens": limit_tokens,
            "concurrency_limit": concurrency_limit,
        }
        super().__init__(
            processor_config=config, strict_event_type=strict_event_type
        )
        self.config = config
        self.interval = interval
        self.limit_requests = limit_requests
        self.limit_tokens = limit_tokens
        self.concurrency_limit = concurrency_limit or queue_capacity
