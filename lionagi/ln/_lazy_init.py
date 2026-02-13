# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe lazy initialization utility."""

import threading
from collections.abc import Callable

__all__ = ("LazyInit", "lazy_import")


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


def lazy_import(
    name: str,
    module_map: dict[str, tuple[str, str | None]],
    package: str,
    globs: dict,
) -> object:
    """Registry-based lazy import for module ``__getattr__``.

    Looks up *name* in *module_map*, imports the object, and caches it
    in *globs* so that subsequent access bypasses ``__getattr__``
    entirely.

    Args:
        name: Attribute name being looked up.
        module_map: ``{attr: (dotted_module, import_name | None)}``.
            When *import_name* is ``None``, the whole module is
            imported and *name* is taken from it via ``getattr``.
        package: The package name (``__name__`` of the caller).
        globs: The caller's ``globals()`` dict for caching.

    Returns:
        The imported object.

    Raises:
        AttributeError: If *name* is not in *module_map*.

    Example::

        _MAP = {
            "Session": ("session.session", "Session"),
            "iModel":  ("service.imodel", "iModel"),
        }

        def __getattr__(name: str):
            return lazy_import(name, _MAP, __name__, globals())
    """
    if name not in module_map:
        raise AttributeError(f"module '{package}' has no attribute '{name}'")
    module_path, import_name = module_map[name]
    import importlib

    # Resolve the parent package for relative imports.
    # If *package* is "lionagi.operations.fields" (a module, not a package),
    # we need "lionagi" as the anchor for relative resolution.
    pkg = globs.get("__package__") or package.rpartition(".")[0]
    mod = importlib.import_module(f".{module_path}", pkg)
    obj = getattr(mod, import_name or name)
    globs[name] = obj
    return obj
