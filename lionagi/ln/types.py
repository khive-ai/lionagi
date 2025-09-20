from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum as _Enum
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    OrderedDict,
    TypeVar,
    Union,
)

from typing_extensions import Self, TypedDict, override

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

__all__ = (
    "Undefined",
    "Unset",
    "MaybeUndefined",
    "MaybeUnset",
    "MaybeSentinel",
    "SingletonType",
    "UndefinedType",
    "UnsetType",
    "KeysDict",
    "T",
    "Enum",
    "is_sentinel",
    "not_sentinel",
    "Params",
    "DataClass",
    "KeysLike",
    "Meta",
    "FieldTemplate",
)

T = TypeVar("T")


class _SingletonMeta(type):
    """Metaclass that guarantees exactly one instance per subclass.

    This ensures that sentinel values maintain identity across the entire application,
    allowing safe identity checks with 'is' operator.
    """

    _cache: dict[type, SingletonType] = {}

    def __call__(cls, *a, **kw):
        if cls not in cls._cache:
            cls._cache[cls] = super().__call__(*a, **kw)
        return cls._cache[cls]


class SingletonType(metaclass=_SingletonMeta):
    """Base class for singleton sentinel types.

    Provides consistent interface for sentinel values with:
    - Identity preservation across deepcopy
    - Falsy boolean evaluation
    - Clear string representation
    """

    __slots__: tuple[str, ...] = ()

    def __deepcopy__(self, memo):  # copy & deepcopy both noop
        return self

    def __copy__(self):
        return self

    # concrete classes *must* override the two methods below
    def __bool__(self) -> bool: ...
    def __repr__(self) -> str: ...


class UndefinedType(SingletonType):
    """Sentinel for a key or field entirely missing from a namespace.

    Use this when:
    - A field has never been set
    - A key doesn't exist in a mapping
    - A value is conceptually undefined (not just unset)

    Example:
        >>> d = {"a": 1}
        >>> d.get("b", Undefined) is Undefined
        True
    """

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __str__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __reduce__(self):
        """Ensure pickle preservation of singleton identity."""
        return "Undefined"


class UnsetType(SingletonType):
    """Sentinel for a key present but value not yet provided.

    Use this when:
    - A parameter exists but hasn't been given a value
    - Distinguishing between None and "not provided"
    - API parameters that are optional but need explicit handling

    Example:
        >>> def func(param=Unset):
        ...     if param is not Unset:
        ...         # param was explicitly provided
        ...         process(param)
    """

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Unset"]:
        return "Unset"

    def __str__(self) -> Literal["Unset"]:
        return "Unset"

    def __reduce__(self):
        """Ensure pickle preservation of singleton identity."""
        return "Unset"


Undefined: Final = UndefinedType()
"""A key or field entirely missing from a namespace"""
Unset: Final = UnsetType()
"""A key present but value not yet provided."""

MaybeUndefined = Union[T, UndefinedType]
MaybeUnset = Union[T, UnsetType]
MaybeSentinel = Union[T, UndefinedType, UnsetType]


def is_sentinel(value: Any) -> bool:
    """Check if a value is any sentinel (Undefined or Unset)."""
    return value is Undefined or value is Unset


def not_sentinel(value: Any) -> bool:
    """Check if a value is NOT a sentinel. Useful for filtering operations."""
    return value is not Undefined and value is not Unset


class Enum(_Enum):
    @classmethod
    def allowed(cls) -> tuple[str, ...]:
        return tuple(e.value for e in cls)


class KeysDict(TypedDict, total=False):
    """TypedDict for keys dictionary."""

    key: Any  # Represents any key-type pair


KeysLike = Sequence[str] | KeysDict


@dataclass(slots=True, frozen=True, init=False)
class Params:
    """Base class for parameters used in various functions."""

    _none_as_sentinel: ClassVar[bool] = False
    """If True, None is treated as a sentinel value."""

    _strict: ClassVar[bool] = False
    """No sentinels allowed if strict is True."""

    _prefill_unset: ClassVar[bool] = True
    """If True, unset fields are prefilled with Unset."""

    _allowed_keys: ClassVar[set[str]] = field(
        default=set(), init=False, repr=False
    )
    """Class variable cache to store allowed keys for parameters."""

    def __init__(self, **kwargs: Any):
        """Initialize the Params object with keyword arguments."""
        # Set all attributes from kwargs, allowing for sentinel values
        for k, v in kwargs.items():
            if k in self.allowed():
                object.__setattr__(self, k, v)
            else:
                raise ValueError(f"Invalid parameter: {k}")

        # Validate after setting all attributes
        self._validate()

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if a value is a sentinel (Undefined or Unset)."""
        if value is None and cls._none_as_sentinel:
            return True
        return is_sentinel(value)

    @classmethod
    def allowed(cls) -> set[str]:
        """Return the keys of the parameters."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = {
            i for i in cls.__dataclass_fields__.keys() if not i.startswith("_")
        }
        return cls._allowed_keys

    @override
    def _validate(self) -> None:
        def _validate_strict(k):
            if self._strict and self._is_sentinel(getattr(self, k, Unset)):
                raise ValueError(f"Missing required parameter: {k}")
            if (
                self._prefill_unset
                and getattr(self, k, Undefined) is Undefined
            ):
                object.__setattr__(self, k, Unset)

        for k in self.allowed():
            _validate_strict(k)

    def default_kw(self) -> Any:
        # create a partial function with the current parameters
        dict_ = self.to_dict()

        # handle kwargs if present, handle both 'kwargs' and 'kw'
        kw_ = {}
        kw_.update(dict_.pop("kwargs", {}))
        kw_.update(dict_.pop("kw", {}))
        dict_.update(kw_)
        return dict_

    def to_dict(self) -> dict[str, str]:
        data = {}
        for k in self.allowed():
            if not self._is_sentinel(v := getattr(self, k, Undefined)):
                data[k] = v
        return data

    def __hash__(self) -> int:
        from ._hash import hash_dict

        return hash_dict(self.to_dict())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Params):
            return False
        return hash(self) == hash(other)


@dataclass(slots=True)
class DataClass:
    """A base class for data classes with strict parameter handling."""

    _none_as_sentinel: ClassVar[bool] = False
    """If True, None is treated as a sentinel value."""

    _strict: ClassVar[bool] = False
    """No sentinels allowed if strict is True."""

    _prefill_unset: ClassVar[bool] = True
    """If True, unset fields are prefilled with Unset."""

    _allowed_keys: ClassVar[set[str]] = field(
        default=set(), init=False, repr=False
    )
    """Class variable cache to store allowed keys for parameters."""

    def __post_init__(self):
        """Post-initialization to ensure all fields are set."""
        self._validate()

    @classmethod
    def allowed(cls) -> set[str]:
        """Return the keys of the parameters."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = {
            i for i in cls.__dataclass_fields__.keys() if not i.startswith("_")
        }
        return cls._allowed_keys

    @override
    def _validate(self) -> None:
        def _validate_strict(k):
            if self._strict and self._is_sentinel(getattr(self, k, Unset)):
                raise ValueError(f"Missing required parameter: {k}")
            if (
                self._prefill_unset
                and getattr(self, k, Undefined) is Undefined
            ):
                self.__setattr__(k, Unset)

        for k in self.allowed():
            _validate_strict(k)

    def to_dict(self) -> dict[str, str]:
        data = {}
        print(self.allowed())
        for k in type(self).allowed():
            if not self._is_sentinel(v := getattr(self, k)):
                data[k] = v
        return data

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if a value is a sentinel (Undefined or Unset)."""
        if value is None and cls._none_as_sentinel:
            return True
        return is_sentinel(value)


@dataclass(slots=True, frozen=True)
class Meta:
    """Immutable metadata container for field templates."""

    key: str
    value: Any

    @override
    def __hash__(self) -> int:
        """Make metadata hashable for caching.

        Note: For callables, we hash by id to maintain identity semantics.
        """
        # For callables, use their id
        if callable(self.value):
            return hash((self.key, id(self.value)))
        # For other values, try to hash directly
        try:
            return hash((self.key, self.value))
        except TypeError:
            # Fallback for unhashable types
            return hash((self.key, str(self.value)))

    @override
    def __eq__(self, other: object) -> bool:
        """Compare metadata for equality.

        For callables, compare by id to increase cache hits when the same
        validator instance is reused. For other values, use standard equality.
        """
        if not isinstance(other, Meta):
            return NotImplemented

        if self.key != other.key:
            return False

        # For callables, compare by identity
        if callable(self.value) and callable(other.value):
            return id(self.value) == id(other.value)

        # For other values, use standard equality
        return bool(self.value == other.value)


# Global cache for annotated types with bounded size
_MAX_CACHE_SIZE = int(os.environ.get("LIONAGI_FIELD_CACHE_SIZE", "10000"))
_annotated_cache: OrderedDict[tuple[type, tuple[Meta, ...]], type] = (
    OrderedDict()
)
_cache_lock = threading.RLock()  # Thread-safe access to cache
_PYDANTIC_FIELD_PARAMS: set[str] | None = None


def _get_pydantic_field_params() -> set[str]:
    """Get valid Pydantic Field parameters (cached)."""
    global _PYDANTIC_FIELD_PARAMS
    if _PYDANTIC_FIELD_PARAMS is None:
        import inspect

        from pydantic import Field as PydanticField

        _PYDANTIC_FIELD_PARAMS = set(
            inspect.signature(PydanticField).parameters.keys()
        )
        _PYDANTIC_FIELD_PARAMS.discard("kwargs")
    return _PYDANTIC_FIELD_PARAMS


@dataclass(slots=True, frozen=True, init=False)
class FieldTemplate:
    base_type: type[Any]
    metadata: tuple[Meta, ...] = field(default_factory=tuple)

    def __init__(
        self,
        name: str,
        value: Any = Unset,
        base_type: type[Any] = Unset,
        *,
        metadata: tuple[Meta, ...] = Unset,
        nullable: Literal[True] = Unset,
        listable: Literal[True] = Unset,
        default: Any = Unset,
        default_factory: Callable[[], Any] = Unset,
        **kw: Any,
    ) -> None:

        if is_sentinel(base_type):
            base_type = Any
        if is_sentinel(metadata):
            metadata = tuple()

        meta_list = list(metadata) if metadata else []

        if not_sentinel(nullable) and nullable is True:
            meta_list.append(Meta("nullable", True))

        if not_sentinel(listable) and listable is True:
            meta_list.append(Meta("listable", True))

        if (
            not_sentinel(name)
            and isinstance(name, str)
            and bool(n := name.strip())
        ):
            meta_list.append(Meta("name", n))

        if sum(not_sentinel(x) for x in (default, default_factory)) > 1:
            raise ValueError("Cannot have both default and default_factory")

        if not_sentinel(default):
            meta_list.append(Meta("default", default))

        if not_sentinel(default_factory):
            if not callable(default_factory):
                raise ValueError("default_factory must be callable")
            meta_list.append(Meta("default", default_factory))

        # Convert remaining kwargs to Meta
        for key, value in kw.items():
            meta_list.append(Meta(key, value))

        # Use object.__setattr__ to set frozen dataclass fields
        object.__setattr__(self, "base_type", base_type)
        object.__setattr__(self, "metadata", tuple(meta_list))

    def __getattr__(self, name: str) -> Any:
        """Handle access to custom attributes stored in metadata."""
        for meta in self.metadata:
            if meta.key == name:
                return meta.value
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def as_nullable(self) -> Self:
        """Create a new field model that allows None values. Handled at annoatation time."""
        return self.with_meta("nullable", True)

    def as_listable(self) -> Self:
        """Create a new field model that wraps the type in a list."""
        new_base = list[self.base_type]  # type: ignore
        new_metadata = (*self.metadata, Meta("listable", True))
        return type(self)(new_base, new_metadata)

    def with_default(self, default: Any) -> Self:
        """Add a default value to this field model."""
        return self.with_meta("default", default)

    def with_meta(self, key: str, value: Any) -> Self:
        """Add custom metadata to this field."""
        filtered_metadata = tuple(m for m in self.metadata if m.key != key)
        new_metadata = (*filtered_metadata, Meta(key, value))
        return type(self)(self.base_type, new_metadata)

    @property
    def is_nullable(self) -> bool:
        """Check if this field allows None values."""
        return any(m.key == "nullable" and m.value for m in self.metadata)

    @property
    def is_listable(self) -> bool:
        """Check if this field is a list type."""
        return any(m.key == "listable" and m.value for m in self.metadata)

    def get_meta(self, key: str) -> Any:
        for meta in self.metadata:
            if meta.key == key:
                return meta.value
        return Undefined

    def annotated(self) -> type[Any]:
        """Materialize this template into an Annotated type."""
        cache_key = (self.base_type, self.metadata)

        with _cache_lock:
            if cache_key in _annotated_cache:
                # Move to end to mark as recently used
                _annotated_cache.move_to_end(cache_key)
                return _annotated_cache[cache_key]

            # Handle nullable case - wrap in Optional-like union
            actual_type = self.base_type
            if any(m.key == "nullable" and m.value for m in self.metadata):
                # Use union syntax for nullable
                actual_type = actual_type | None  # type: ignore

            if self.metadata:
                # Python 3.10 doesn't support unpacking in Annotated, so we need to build it differently
                # We'll use Annotated.__class_getitem__ to build the type dynamically
                args = [actual_type] + list(self.metadata)
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

    def create_pydantic_field(self) -> FieldInfo:
        """Create a Pydantic FieldInfo object from this template.

        Returns:
            A Pydantic FieldInfo object with all metadata applied
        """
        from pydantic import Field as PydanticField

        # Get valid Pydantic Field parameters (cached)
        pydantic_field_params = _get_pydantic_field_params()

        # Extract metadata for FieldInfo
        field_kwargs = {}

        for meta in self.metadata:
            if meta.key == "default":
                # Handle callable defaults as default_factory
                if callable(meta.value):
                    field_kwargs["default_factory"] = meta.value
                else:
                    field_kwargs["default"] = meta.value
            elif meta.key == "validator":
                # Validators are handled separately in create_model
                continue
            elif meta.key in pydantic_field_params:
                # Pass through standard Pydantic field attributes
                field_kwargs[meta.key] = meta.value
            elif meta.key in {"nullable", "listable"}:
                # These are FieldTemplate markers, don't pass to FieldInfo
                pass
            else:
                # Any other metadata goes in json_schema_extra
                if "json_schema_extra" not in field_kwargs:
                    field_kwargs["json_schema_extra"] = {}
                field_kwargs["json_schema_extra"][meta.key] = meta.value

        # Handle nullable case - ensure default is set if not already
        if (
            self.is_nullable
            and "default" not in field_kwargs
            and "default_factory" not in field_kwargs
        ):
            field_kwargs["default"] = None

        field_info = PydanticField(**field_kwargs)

        return field_info
