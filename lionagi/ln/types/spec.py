import contextlib
import inspect
import os
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from typing_extensions import Annotated, OrderedDict

from lionagi.ln.concurrency.utils import is_coro_func

from .._to_list import to_list
from ._sentinel import MaybeUndefined, Undefined, is_sentinel, not_sentinel
from .base import Meta

# Global cache for annotated types with bounded size
_MAX_CACHE_SIZE = int(os.environ.get("LIONAGI_FIELD_CACHE_SIZE", "10000"))
_annotated_cache: OrderedDict[tuple[type, tuple[Meta, ...]], type] = (
    OrderedDict()
)
_cache_lock = threading.RLock()  # Thread-safe access to cache


__all__ = ("Spec",)


class CommonMeta(Enum):

    NAME = "name"
    NULLABLE = "nullable"
    LISTABLE = "listable"
    VALIDATOR = "validator"
    DEFAULT = "default"
    DEFAULT_FACTORY = "default_factory"

    @classmethod
    def allowed(cls) -> set[str]:
        return {i.value for i in cls}

    @classmethod
    def _validate_common_metas(cls, **kw):
        if kw.get("default") and kw.get("default_factory"):
            raise ValueError(
                "Cannot provide both 'default' and 'default_factory'"
            )
        if _df := kw.get("default_factory"):
            if not inspect.isfunction(_df):
                raise ValueError(
                    "'default_factory' must be a callable function"
                )
        if _val := kw.get("validator"):
            _val = [_val] if not isinstance(_val, list) else _val
            if not all(callable(v) for v in _val):
                raise ValueError(
                    "Validators must be a list of functions or a function"
                )

    @classmethod
    def prepare(
        cls, *args: Meta, metadata: tuple[Meta, ...] = None, **kw: Any
    ) -> tuple[Meta, ...]:
        _metas = list(metadata) if metadata else []
        _args = (
            to_list(args, flatten=True, flatten_tuple_set=True, dropna=True)
            if args
            else []
        )
        _kw = [Meta(k, v) for k, v in kw.items()]
        metas = _metas + _args + _kw
        meta_dict = {meta.key: meta.value for meta in metas}
        if len(meta_dict) != len(metas):
            raise ValueError("Duplicate metadata keys found in Spec metadata.")
        cls._validate_common_metas(**meta_dict)
        return tuple(metas)


@dataclass(frozen=True, slots=True, init=False)
class Spec:
    base_type: type
    metadata: tuple[Meta, ...]
    sha256: str | None = None

    def __init__(
        self,
        base_type: type = None,
        *args,
        metadata: tuple[Meta, ...] = None,
        **kw,
    ) -> None:
        metas = CommonMeta.prepare(*args, metadata=metadata, **kw)
        if not_sentinel(base_type, True):
            import types

            is_valid_type = (
                isinstance(base_type, type)
                or hasattr(base_type, "__origin__")
                or isinstance(base_type, types.UnionType)
                or str(type(base_type)) == "<class 'types.UnionType'>"
            )
            if not is_valid_type:
                raise ValueError(
                    f"base_type must be a type or type annotation, got {base_type}"
                )
        object.__setattr__(self, "base_type", base_type)
        object.__setattr__(self, "metadata", metas)

    def __getitem__(self, key: str) -> Any:
        for meta in self.metadata:
            if meta.key == key:
                return meta.value
        raise KeyError(f"Metadata key '{key}' undefined in Spec.")

    def get(self, key: str, default: Any = Undefined) -> Any:
        with contextlib.suppress(KeyError):
            return self[key]
        return default

    @property
    def name(self) -> MaybeUndefined[str]:
        return self.get(CommonMeta.NAME.value)

    @property
    def is_nullable(self) -> bool:
        return self.get(CommonMeta.NULLABLE.value) is True

    @property
    def is_listable(self) -> bool:
        return self.get(CommonMeta.LISTABLE.value) is True

    @property
    def default(self) -> MaybeUndefined[Any]:
        return self.get(
            CommonMeta.DEFAULT.value,
            self.get(CommonMeta.DEFAULT_FACTORY.value),
        )

    @property
    def has_default_factory(self) -> bool:
        return _is_factory(self.get(CommonMeta.DEFAULT_FACTORY.value))[0]

    @property
    def has_async_default_factory(self) -> bool:
        return _is_factory(self.get(CommonMeta.DEFAULT_FACTORY.value))[1]

    def create_default_value(self) -> Any:
        if self.default is Undefined:
            raise ValueError("No default value or factory defined in Spec.")
        if self.has_async_default_factory:
            raise ValueError(
                "Default factory is asynchronous; cannot create default synchronously."
                "use 'await spec.create_default_value_async()' instead."
            )
        if self.has_default_factory:
            return self.default()
        return self.default

    async def acreate_default_value(self) -> Any:
        if self.has_async_default_factory:
            return await self.default()
        return self.create_default_value()

    def with_updates(self, **kw):
        _filtered = [meta for meta in self.metadata if meta.key not in kw]
        for k, v in kw.items():
            if not_sentinel(v):
                _filtered.append(Meta(k, v))
        _metas = tuple(_filtered)
        return type(self)(self.base_type, metadata=_metas)

    def as_nullable(self) -> "Spec":
        return self.with_updates(nullable=True)

    def as_listable(self) -> "Spec":
        return self.with_updates(listable=True)

    def with_default(self, default: Any) -> "Spec":
        if inspect.isfunction(default):
            return self.with_updates(default_factory=default)
        return self.with_updates(default=default)

    def with_validator(
        self, validator: Callable[..., Any] | list[Callable[..., Any]]
    ) -> "Spec":
        return self.with_updates(validator=validator)

    @property
    def annotation(self) -> type[Any]:
        """A plain type annotation representing base type, nullable, and listable."""
        if is_sentinel(self.base_type, True):
            return Any
        t_ = self.base_type
        if self.is_listable:
            t_ = list[t_]
        if self.is_nullable:
            return t_ | None
        return t_

    def annotated(self) -> type[Any]:
        """Materialize this template into an Annotated type.

        This method is cached to ensure repeated calls return the same
        type object for performance and identity checks. The cache is bounded
        using LRU eviction to prevent unbounded memory growth.

        Returns:
            Annotated type with all metadata attached
        """
        # Check cache first with thread safety
        cache_key = (self.base_type, self.metadata)

        with _cache_lock:
            if cache_key in _annotated_cache:
                # Move to end to mark as recently used
                _annotated_cache.move_to_end(cache_key)
                return _annotated_cache[cache_key]

            # Handle nullable case - wrap in Optional-like union
            actual_type = (
                Any
                if is_sentinel(self.base_type, none_as_sentinel=True)
                else self.base_type
            )
            current_metadata = (
                ()
                if is_sentinel(self.metadata, none_as_sentinel=True)
                else self.metadata
            )

            if any(m.key == "nullable" and m.value for m in current_metadata):
                # Use union syntax for nullable
                actual_type = actual_type | None  # type: ignore

            if current_metadata:
                args = [actual_type] + list(current_metadata)
                result = Annotated.__class_getitem__(tuple(args))  # type: ignore
            else:
                result = actual_type  # type: ignore[misc]

            # Cache the result with LRU eviction
            _annotated_cache[cache_key] = result  # type: ignore[assignment]

            # Evict oldest if cache is too large (guard against empty cache)
            while len(_annotated_cache) > _MAX_CACHE_SIZE:
                try:
                    _annotated_cache.popitem(last=False)  # Remove oldest
                except KeyError:
                    # Cache became empty during race, safe to continue
                    break

        return result  # type: ignore[return-value]

    def metadict(
        self, exclude: set[str] | None = None, exclude_common: bool = False
    ) -> dict[str, Any]:
        if exclude is None:
            exclude = set()
        if exclude_common:
            exclude = exclude | CommonMeta.allowed()
        return {
            meta.key: meta.value
            for meta in self.metadata
            if meta.key not in exclude
        }


def _is_factory(obj: Any) -> tuple[bool, bool]:
    """returns (is_factory, is_async)"""
    if not callable(obj):
        return (False, False)
    if is_coro_func(obj):
        return (True, True)
    return (True, False)
