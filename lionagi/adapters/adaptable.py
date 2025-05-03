"""Mixin that endows a domain class with *adapt-from / adapt-to* helpers."""

from __future__ import annotations

from typing import Any, ClassVar, TypeVar

from .adapter import Adapter, AdapterRegistry

T = TypeVar("T", bound="Adaptable")


class Adaptable:
    """Attach to any pydantic model (or plain dataclass) to make it adaptable.

    Each *concrete* subclass maintains its own :class:`AdapterRegistry`
    so that ``User.adapt_from(..., obj_key="json")`` is independent of
    ``Trade.adapt_from(..., obj_key="json")``.
    """

    _adapter_registry: ClassVar[AdapterRegistry] | None = None

    # --------------------------------------------------------------------- #
    # registry management                                                   #
    # --------------------------------------------------------------------- #
    @classmethod
    def _registry(cls) -> AdapterRegistry:
        if cls._adapter_registry is None:
            cls._adapter_registry = AdapterRegistry()
        return cls._adapter_registry

    @classmethod
    def register_adapter(cls, adapter_cls: type[Adapter]) -> None:
        """Attach an adapter (class) to *this* subject."""
        cls._registry().register(adapter_cls)

    # --------------------------------------------------------------------- #
    # high-level helpers                                                    #
    # --------------------------------------------------------------------- #
    @classmethod
    def adapt_from(
        cls: type[T],
        obj: Any,
        /,
        *,
        obj_key: str,
        many: bool = False,
        **kwargs,
    ) -> T | list[T]:
        """Create one or many ``cls`` from external object."""
        return cls._registry().adapt_from(
            cls, obj, obj_key=obj_key, many=many, **kwargs
        )

    def adapt_to(
        self: T, /, *, obj_key: str, many: bool = False, **kwargs
    ) -> Any:
        """Convert *self* (or list thereof) to external form."""
        return self._registry().adapt_to(
            self, obj_key=obj_key, many=many, **kwargs
        )
