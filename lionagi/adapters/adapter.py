"""Polished *Adapter* abstraction and *AdapterRegistry*.

An *adapter* converts between an internal domain object and some external
representation (JSON string, CSV file, pandas.DataFrame, ...).

Design goals
------------
* **Stateless** - all conversion logic is expressed as *class methods*.
* **Pluggable** - register new adapters at runtime.
* **Type-safe** - uses a *single* type parameter `T` representing the subject.
* **Minimal surface** - exactly two conversion hooks: `from_obj` and `to_obj`.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")  # internal subject type


# --------------------------------------------------------------------------- #
# Adapter protocol                                                            #
# --------------------------------------------------------------------------- #
@runtime_checkable
class Adapter(Protocol[T]):
    """A stateless conversion helper between *subject* and *external* form.

    Each concrete adapter must declare a unique :pyattr:`obj_key`
    (file-extension-style string, e.g. ``"json"``, ``".csv"``).

    Implementations *must* provide two classmethods:

    * :py:meth:`from_obj` - build **one or many** instances of ``T`` from
      external object
    * :py:meth:`to_obj`   - convert **one or many** ``T`` into external object
    """

    # unique identifier (e.g. "json", ".csv", "pd_dataframe")
    obj_key: ClassVar[str]

    # ---------- required conversion hooks ----------
    @classmethod
    def from_obj(
        cls, subj_cls: type[T], obj: Any, /, *, many: bool = False, **kwargs
    ) -> T | list[T]: ...

    @classmethod
    def to_obj(
        cls, subj: T | list[T], /, *, many: bool = False, **kwargs
    ) -> Any: ...


# --------------------------------------------------------------------------- #
# Adapter registry                                                            #
# --------------------------------------------------------------------------- #
class AdapterRegistry:
    """Keeps a mapping ``obj_key -> adapter_cls``."""

    def __init__(self) -> None:
        self._reg: dict[str, type[Adapter]] = {}

    # --------------------------------------------------------------------- #
    # public API                                                            #
    # --------------------------------------------------------------------- #
    def register(self, adapter_cls: type[Adapter]) -> None:
        key = getattr(adapter_cls, "obj_key", None)
        if not key:
            raise AttributeError(
                "Adapter class must define 'obj_key' attribute"
            )
        if key in self._reg:
            logging.warning(
                "Adapter for '%s' replaced: %s -> %s",
                key,
                self._reg[key],
                adapter_cls,
            )
        self._reg[key] = adapter_cls

    def get(self, obj_key: str) -> type[Adapter]:
        try:
            return self._reg[obj_key]
        except KeyError as exc:
            raise KeyError(f"No adapter registered for '{obj_key}'") from exc

    # convenience shortcuts
    def adapt_from(
        self,
        subj_cls: type[T],
        obj: Any,
        /,
        *,
        obj_key: str,
        many: bool = False,
        **kwargs,
    ) -> T | list[T]:
        return self.get(obj_key).from_obj(subj_cls, obj, many=many, **kwargs)

    def adapt_to(
        self,
        subj: T | list[T],
        /,
        *,
        obj_key: str,
        many: bool = False,
        **kwargs,
    ) -> Any:
        return self.get(obj_key).to_obj(subj, many=many, **kwargs)
