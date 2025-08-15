"""Structured concurrency primitives for pynector.

This module provides structured concurrency primitives using AnyIO,
which allows for consistent behavior across asyncio and trio backends.
"""

from .cancel import CancelScope, fail_after, move_on_after
from .errors import get_cancelled_exc_class, shield
from .patterns import (
    ConnectionPool,
    WorkerPool,
    parallel_requests,
    retry_with_timeout,
)
from .primitives import CapacityLimiter, Condition, Event, Lock, Semaphore
from .resource_tracker import (
    ResourceTracker,
    cleanup_check,
    get_global_tracker,
    resource_leak_detector,
    track_resource,
    untrack_resource,
)
from .task import TaskGroup, create_task_group
from .utils import is_coro_func

__all__ = (
    "TaskGroup",
    "create_task_group",
    "CancelScope",
    "move_on_after",
    "fail_after",
    "ConnectionPool",
    "WorkerPool",
    "parallel_requests",
    "retry_with_timeout",
    "Lock",
    "Semaphore",
    "CapacityLimiter",
    "Event",
    "Condition",
    "get_cancelled_exc_class",
    "shield",
    "ResourceTracker",
    "resource_leak_detector",
    "track_resource",
    "untrack_resource",
    "cleanup_check",
    "get_global_tracker",
    "is_coro_func",
)
