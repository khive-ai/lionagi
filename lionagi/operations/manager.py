"""experimental"""

from collections.abc import Callable
from typing import Coroutine

from lionagi.protocols._concepts import Manager
from lionagi.utils import is_coro_func


class OperationManager(Manager):

    def __init__(self):
        self.registry: dict[str, Callable] = {}

    def register_operations(self, name: str, func, /, *, update=False) -> None:
        """Register async operations with the manager"""

        if not is_coro_func(func):
            raise ValueError(
                f"Function {name} is not a coroutine function. Operations must be async."
            )

        ...

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.registry: dict[str, Callable] = {}
        self.register_operations(*args, **kwargs)

    def register_operations(self, *args, **kwargs) -> None:
        operations = {}
        if args:
            operations = {i.__name__ for i in args if hasattr(i, "__name__")}
        operations.update(kwargs)
        self.registry.update(operations)
