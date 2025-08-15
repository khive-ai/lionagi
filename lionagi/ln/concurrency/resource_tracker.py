"""Resource tracking utilities for concurrency primitives.

This module provides lightweight resource leak detection and lifecycle tracking
to address the security vulnerabilities identified in the hardening tests.
"""

import weakref
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ResourceInfo:
    """Information about a tracked resource."""

    name: str
    creation_time: datetime
    resource_type: str


class ResourceTracker:
    """Lightweight resource lifecycle tracking for leak detection.

    This addresses the over-engineering concerns by providing simple,
    practical resource management without complex abstraction layers.
    """

    def __init__(self):
        """Initialize a new resource tracker."""
        self._active_resources: dict[int, ResourceInfo] = {}
        self._weak_refs: weakref.WeakKeyDictionary = (
            weakref.WeakKeyDictionary()
        )

    def track(
        self, resource: Any, name: str, resource_type: str | None = None
    ) -> None:
        """Track a resource for leak detection.

        Args:
            resource: The resource to track
            name: Human-readable name for the resource
            resource_type: Optional type classification
        """
        if resource_type is None:
            resource_type = type(resource).__name__

        resource_info = ResourceInfo(
            name=name,
            creation_time=datetime.now(),
            resource_type=resource_type,
        )

        # Use weak reference to avoid interfering with garbage collection
        self._weak_refs[resource] = resource_info
        self._active_resources[id(resource)] = resource_info

    def untrack(self, resource: Any) -> None:
        """Manually untrack a resource.

        Args:
            resource: The resource to stop tracking
        """
        resource_id = id(resource)
        self._active_resources.pop(resource_id, None)
        self._weak_refs.pop(resource, None)

    def cleanup_check(self) -> list[ResourceInfo]:
        """Check for potentially leaked resources.

        Returns:
            List of resource info for resources that may have leaked
        """
        # Clean up references to garbage collected objects
        current_resources = []
        for resource, info in list(self._weak_refs.items()):
            current_resources.append(info)

        return current_resources

    def get_active_count(self) -> int:
        """Get the number of currently tracked resources.

        Returns:
            Number of active tracked resources
        """
        return len(self._weak_refs)

    def get_resource_summary(self) -> dict[str, int]:
        """Get a summary of tracked resources by type.

        Returns:
            Dictionary mapping resource types to counts
        """
        summary = {}
        for info in self._weak_refs.values():
            resource_type = info.resource_type
            summary[resource_type] = summary.get(resource_type, 0) + 1
        return summary


# Global tracker instance for convenience
_global_tracker = ResourceTracker()


def track_resource(
    resource: Any, name: str, resource_type: str | None = None
) -> None:
    """Track a resource using the global tracker.

    Args:
        resource: The resource to track
        name: Human-readable name for the resource
        resource_type: Optional type classification
    """
    _global_tracker.track(resource, name, resource_type)


def untrack_resource(resource: Any) -> None:
    """Untrack a resource using the global tracker.

    Args:
        resource: The resource to stop tracking
    """
    _global_tracker.untrack(resource)


def get_global_tracker() -> ResourceTracker:
    """Get the global resource tracker instance.

    Returns:
        The global ResourceTracker instance
    """
    return _global_tracker


def cleanup_check() -> list[ResourceInfo]:
    """Check for potentially leaked resources using global tracker.

    Returns:
        List of resource info for resources that may have leaked
    """
    return _global_tracker.cleanup_check()


class resource_leak_detector:
    """Context manager for resource leak detection in tests and production.

    Example:
        async with resource_leak_detector() as tracker:
            lock = Lock()
            tracker.track(lock, "test_lock")
            # ... use lock
        # Automatically checks for leaks on exit
    """

    def __init__(self, raise_on_leak: bool = False):
        """Initialize the leak detector.

        Args:
            raise_on_leak: Whether to raise an exception if leaks are detected
        """
        self.raise_on_leak = raise_on_leak
        self.tracker = ResourceTracker()
        self._initial_count = 0

    async def __aenter__(self) -> ResourceTracker:
        """Enter the context and return the tracker."""
        self._initial_count = self.tracker.get_active_count()
        return self.tracker

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context and check for leaks."""
        leaked_resources = self.tracker.cleanup_check()

        if leaked_resources and self.raise_on_leak:
            resource_summary = self.tracker.get_resource_summary()
            raise RuntimeError(
                f"Resource leak detected: {len(leaked_resources)} resources "
                f"still active. Summary: {resource_summary}"
            )
