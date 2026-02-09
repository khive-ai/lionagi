# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe lazy initialization utility."""

import threading
from collections.abc import Callable

__all__ = ("LazyInit",)


class LazyInit:
    """Thread-safe lazy initialization helper using double-checked locking.

    Defers expensive imports/setup until first use. Guarantees init_func
    runs exactly once even under concurrent access.

    Example:
        _lazy = LazyInit()
        _MODEL_LIKE = None

        def _do_init():
            global _MODEL_LIKE
            from pydantic import BaseModel
            _MODEL_LIKE = (BaseModel,)

        def my_function(x):
            _lazy.ensure(_do_init)
            # _MODEL_LIKE is now initialized

    Attributes:
        initialized: True after ensure() has completed successfully.
    """

    __slots__ = ("_initialized", "_lock")

    def __init__(self) -> None:
        """Create uninitialized LazyInit instance."""
        self._initialized = False
        self._lock = threading.RLock()

    @property
    def initialized(self) -> bool:
        """Check if initialization has completed."""
        return self._initialized

    def ensure(self, init_func: Callable[[], None]) -> None:
        """Execute init_func exactly once, thread-safely.

        Uses double-checked locking: fast path (no lock) when already
        initialized, lock acquisition only on first call.

        Args:
            init_func: Initialization function. Should be idempotent
                as a safety measure (though only called once).
        """
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            init_func()
            self._initialized = True
