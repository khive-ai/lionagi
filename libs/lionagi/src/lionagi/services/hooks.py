# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Hook system for service lifecycle events - adapted from v0 to v1 architecture."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar
from uuid import UUID

import anyio

from .core import CallContext, ServiceError
from .endpoint import RequestModel

logger = logging.getLogger(__name__)

Req = TypeVar("Req", bound=RequestModel)
Res = TypeVar("Res")
Chunk = TypeVar("Chunk")


class HookType(Enum):
    """Types of hooks in the service lifecycle."""

    PRE_CALL = "pre_call"  # Before service call starts
    POST_CALL = "post_call"  # After service call completes
    PRE_STREAM = "pre_stream"  # Before streaming starts
    POST_STREAM = "post_stream"  # After streaming completes
    STREAM_CHUNK = "stream_chunk"  # For each streaming chunk
    CALL_ERROR = "call_error"  # On call errors
    STREAM_ERROR = "stream_error"  # On streaming errors
    RATE_LIMITED = "rate_limited"  # When rate limited
    QUEUE_FULL = "queue_full"  # When queue is full


@dataclass(slots=True)
class HookEvent:
    """Event data passed to hooks."""

    hook_type: HookType
    call_id: UUID
    branch_id: UUID
    service_name: str
    request: RequestModel | None = None
    context: CallContext | None = None
    result: Any = None
    error: Exception | None = None
    chunk: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=anyio.current_time)


# Hook function types
HookFunc = Callable[[HookEvent], Awaitable[None]]
SyncHookFunc = Callable[[HookEvent], None]
StreamHookFunc = Callable[[HookEvent, Any], Awaitable[Any]]  # Can transform chunks


class HookRegistry:
    """Registry for managing service lifecycle hooks."""

    def __init__(self):
        self._hooks: dict[HookType, list[HookFunc | SyncHookFunc]] = {}
        self._stream_hooks: dict[HookType, list[StreamHookFunc]] = {}
        self._timeout: float = 30.0  # Default hook timeout

    def register(
        self,
        hook_type: HookType,
        hook_func: HookFunc | SyncHookFunc,
        *,
        timeout: float | None = None,
    ) -> None:
        """Register a hook function for a specific event type."""
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []

        self._hooks[hook_type].append(hook_func)

        if timeout:
            # Store timeout metadata - could be per-hook in future
            self._timeout = timeout

        logger.debug(f"Registered hook for {hook_type.value}")

    def register_stream_hook(
        self, hook_type: HookType, hook_func: StreamHookFunc, *, timeout: float | None = None
    ) -> None:
        """Register a streaming hook that can transform chunks."""
        if hook_type not in self._stream_hooks:
            self._stream_hooks[hook_type] = []

        self._stream_hooks[hook_type].append(hook_func)

        if timeout:
            self._timeout = timeout

        logger.debug(f"Registered stream hook for {hook_type.value}")

    def unregister(self, hook_type: HookType, hook_func: HookFunc | SyncHookFunc) -> None:
        """Unregister a hook function."""
        if hook_type in self._hooks:
            try:
                self._hooks[hook_type].remove(hook_func)
                logger.debug(f"Unregistered hook for {hook_type.value}")
            except ValueError:
                pass

    def has_hooks(self, hook_type: HookType) -> bool:
        """Check if there are any hooks registered for this type."""
        return (hook_type in self._hooks and len(self._hooks[hook_type]) > 0) or (
            hook_type in self._stream_hooks and len(self._stream_hooks[hook_type]) > 0
        )

    async def emit(self, event: HookEvent) -> None:
        """Emit an event to all registered hooks."""
        if event.hook_type not in self._hooks:
            return

        # Execute all hooks concurrently with proper timeout enforcement
        async def _run() -> None:
            from lionagi.ln.concurrency import create_task_group
            async with create_task_group() as tg:
                for hook_func in self._hooks[event.hook_type]:
                    tg.start_soon(self._execute_hook, hook_func, event)

        try:
            from lionagi.ln.concurrency import fail_after
            with fail_after(self._timeout):
                await _run()
        except TimeoutError:
            logger.warning(f"Hook execution timed out for {event.hook_type.value}")
        except Exception as e:
            logger.error(f"Hook execution failed for {event.hook_type.value}: {e}", exc_info=True)

    async def emit_stream_chunk(self, event: HookEvent, chunk: Any) -> Any:
        """Emit a streaming chunk event and allow transformation."""
        if event.hook_type not in self._stream_hooks:
            return chunk

        result = chunk

        # Apply stream hooks in sequence with proper timeout enforcement per hook
        from lionagi.ln.concurrency import fail_after
        
        for hook_func in self._stream_hooks[event.hook_type]:
            try:
                with fail_after(self._timeout):
                    result = await hook_func(event, result)
            except TimeoutError:
                logger.warning(f"Stream hook timed out for {event.hook_type.value}")
                break
            except Exception as e:
                logger.error(f"Stream hook failed for {event.hook_type.value}: {e}", exc_info=True)
                break

        return result

    async def _execute_hook(self, hook_func: HookFunc | SyncHookFunc, event: HookEvent) -> None:
        """Execute a single hook function."""
        from lionagi.ln.concurrency import is_coro_func
        try:
            if is_coro_func(hook_func):
                await hook_func(event)
            else:
                hook_func(event)
        except Exception as e:
            logger.error(f"Hook function failed: {e}", exc_info=True)


    def clear(self) -> None:
        """Clear all registered hooks."""
        self._hooks.clear()
        self._stream_hooks.clear()
        logger.debug("Cleared all hooks")


class HookedMiddleware:
    """Middleware that integrates hooks with the v1 middleware chain."""

    def __init__(self, registry: HookRegistry):
        self.registry = registry

    async def __call__(
        self, req: RequestModel, ctx: CallContext, next_call: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Apply hooks around call execution."""
        service_name = getattr(ctx.attrs, "service_name", "unknown")

        # Pre-call hook
        if self.registry.has_hooks(HookType.PRE_CALL):
            event = HookEvent(
                hook_type=HookType.PRE_CALL,
                call_id=ctx.call_id,
                branch_id=ctx.branch_id,
                service_name=service_name,
                request=req,
                context=ctx,
            )
            await self.registry.emit(event)

        try:
            result = await next_call()

            # Post-call hook
            if self.registry.has_hooks(HookType.POST_CALL):
                event = HookEvent(
                    hook_type=HookType.POST_CALL,
                    call_id=ctx.call_id,
                    branch_id=ctx.branch_id,
                    service_name=service_name,
                    request=req,
                    context=ctx,
                    result=result,
                )
                await self.registry.emit(event)

            return result

        except Exception as e:
            # Error hook
            if self.registry.has_hooks(HookType.CALL_ERROR):
                event = HookEvent(
                    hook_type=HookType.CALL_ERROR,
                    call_id=ctx.call_id,
                    branch_id=ctx.branch_id,
                    service_name=service_name,
                    request=req,
                    context=ctx,
                    error=e,
                )
                await self.registry.emit(event)

            raise

    async def stream(
        self, req: RequestModel, ctx: CallContext, next_stream: Callable[[], AsyncIterator[Any]]
    ) -> AsyncIterator[Any]:
        """Apply hooks around streaming execution."""
        service_name = getattr(ctx.attrs, "service_name", "unknown")

        # Pre-stream hook
        if self.registry.has_hooks(HookType.PRE_STREAM):
            event = HookEvent(
                hook_type=HookType.PRE_STREAM,
                call_id=ctx.call_id,
                branch_id=ctx.branch_id,
                service_name=service_name,
                request=req,
                context=ctx,
            )
            await self.registry.emit(event)

        try:
            chunk_count = 0
            async for chunk in next_stream():
                chunk_count += 1

                # Stream chunk hook (can transform chunk)
                if self.registry.has_hooks(HookType.STREAM_CHUNK):
                    event = HookEvent(
                        hook_type=HookType.STREAM_CHUNK,
                        call_id=ctx.call_id,
                        branch_id=ctx.branch_id,
                        service_name=service_name,
                        request=req,
                        context=ctx,
                        chunk=chunk,
                        metadata={"chunk_count": chunk_count},
                    )
                    chunk = await self.registry.emit_stream_chunk(event, chunk)

                yield chunk

            # Post-stream hook
            if self.registry.has_hooks(HookType.POST_STREAM):
                event = HookEvent(
                    hook_type=HookType.POST_STREAM,
                    call_id=ctx.call_id,
                    branch_id=ctx.branch_id,
                    service_name=service_name,
                    request=req,
                    context=ctx,
                    metadata={"chunk_count": chunk_count},
                )
                await self.registry.emit(event)

        except Exception as e:
            # Stream error hook
            if self.registry.has_hooks(HookType.STREAM_ERROR):
                event = HookEvent(
                    hook_type=HookType.STREAM_ERROR,
                    call_id=ctx.call_id,
                    branch_id=ctx.branch_id,
                    service_name=service_name,
                    request=req,
                    context=ctx,
                    error=e,
                )
                await self.registry.emit(event)

            raise


# Global hook registry for convenience
_global_registry = HookRegistry()


def get_global_hooks() -> HookRegistry:
    """Get the global hook registry."""
    return _global_registry


# Decorator for easy hook registration
def hook(hook_type: HookType, *, timeout: float | None = None):
    """Decorator to register a function as a hook.

    Usage:
        @hook(HookType.PRE_CALL)
        async def my_hook(event: HookEvent):
            print(f"Call starting: {event.call_id}")
    """

    def decorator(func):
        _global_registry.register(hook_type, func, timeout=timeout)
        return func

    return decorator


def stream_hook(hook_type: HookType, *, timeout: float | None = None):
    """Decorator to register a streaming hook that can transform chunks.

    Usage:
        @stream_hook(HookType.STREAM_CHUNK)
        async def my_stream_hook(event: HookEvent, chunk: Any) -> Any:
            # Transform chunk if needed
            return chunk
    """

    def decorator(func):
        _global_registry.register_stream_hook(hook_type, func, timeout=timeout)
        return func

    return decorator


# Example hooks for common use cases


@hook(HookType.PRE_CALL)
async def log_call_start(event: HookEvent):
    """Log when a call starts."""
    logger.info(f"Call {event.call_id} starting on service {event.service_name}")


@hook(HookType.POST_CALL)
async def log_call_complete(event: HookEvent):
    """Log when a call completes."""
    logger.info(f"Call {event.call_id} completed successfully")


@hook(HookType.CALL_ERROR)
async def log_call_error(event: HookEvent):
    """Log when a call fails."""
    logger.error(f"Call {event.call_id} failed: {event.error}")


@stream_hook(HookType.STREAM_CHUNK)
async def count_stream_chunks(event: HookEvent, chunk: Any) -> Any:
    """Count streaming chunks."""
    count = event.metadata.get("chunk_count", 0)
    if count % 100 == 0:  # Log every 100 chunks
        logger.debug(f"Stream {event.call_id} processed {count} chunks")
    return chunk
