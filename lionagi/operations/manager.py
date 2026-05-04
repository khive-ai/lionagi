from collections.abc import Callable

from lionagi.protocols._concepts import Manager
from lionagi.utils import is_coro_func

"""
experimental
"""


class OperationManager(Manager):
    def __init__(self):
        super().__init__()
        self.registry: dict[str, Callable] = {}

    def register(self, operation: str, func: Callable, update: bool = False):
        if operation in self.registry and not update:
            raise ValueError(f"Operation '{operation}' is already registered.")
        if not is_coro_func(func):
            raise ValueError(f"Operation '{operation}' must be an async function.")
        self.registry[operation] = func

    def unregister(self, operation: str, func: Callable | None = None) -> bool:
        """Remove an operation registration.

        If ``func`` is provided, the registration is removed only when it still
        points to that callable. This lets temporary owners clean up without
        deleting a newer registration for the same name.
        """
        registered = self.registry.get(operation)
        if registered is None:
            return False
        if func is not None and registered is not func:
            return False
        self.registry.pop(operation, None)
        return True
