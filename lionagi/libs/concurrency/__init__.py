"""Structured concurrency primitives.

This module provides structured concurrency primitives using AnyIO,
which allows for consistent behavior across asyncio and trio backends.
"""

from .cancel import CancelScope, fail_after, move_on_after
from .errors import get_cancelled_exc_class, shield
from .primitives import (
    CapacityLimiter,
    Condition,
    Event,
    Lock,
    Semaphore,
)
from .task import TaskGroup, create_task_group

__all__ = [
    "TaskGroup",
    "create_task_group",
    "CancelScope",
    "move_on_after",
    "fail_after",
    "Lock",
    "Semaphore",
    "CapacityLimiter",
    "Event",
    "Condition",
    "get_cancelled_exc_class",
    "shield",
]
