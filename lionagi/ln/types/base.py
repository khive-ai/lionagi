from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence, MutableSet, Sequence
from dataclasses import dataclass, field, fields
from enum import Enum as _Enum
from typing import Any, ClassVar, Literal

from typing_extensions import Self, TypedDict, override

from ._sentinel import Undefined, Unset, is_sentinel, is_undefined

__all__ = (
    "DataClass",
    "Enum",
    "KeysDict",
    "KeysLike",
    "Meta",
    "ModelConfig",
    "Params",
)


class Enum(_Enum):
    """Enhanced Enum with allowed() classmethod."""

    @classmethod
    def allowed(cls) -> tuple[str, ...]:
        return tuple(e.value for e in cls)


class KeysDict(TypedDict, total=False):
    """TypedDict for flexible key-type mappings."""

    key: Any


@dataclass(slots=True, frozen=True, init=False)
class ModelConfig:
    """Configuration for Params and DataClass behavior.

    Attributes:
        sentinel_additions: Additional sentinel categories beyond Undefined/Unset.
            Valid values: "none", "empty", "pydantic", "dataclass".
        strict: If True, no sentinels allowed (all fields must have values).
        prefill_unset: If True, unset fields are prefilled with Unset.
        use_enum_values: If True, use enum values instead of enum instances in to_dict().
    """

    sentinel_additions: frozenset[str] = field(default_factory=frozenset)
    strict: bool = False
    prefill_unset: bool = True
    use_enum_values: bool = False

    def __init__(
        self,
        *,
        sentinel_additions: frozenset[str] | set[str] | None = None,
        # backwards compat
        none_as_sentinel: bool = False,
        empty_as_sentinel: bool = False,
        strict: bool = False,
        prefill_unset: bool = True,
        use_enum_values: bool = False,
    ):
        additions = set(sentinel_additions) if sentinel_additions else set()
        if none_as_sentinel:
            additions.add("none")
        if empty_as_sentinel:
            additions.add("empty")
        object.__setattr__(self, "sentinel_additions", frozenset(additions))
        object.__setattr__(self, "strict", strict)
        object.__setattr__(self, "prefill_unset", prefill_unset)
        object.__setattr__(self, "use_enum_values", use_enum_values)

    @property
    def none_as_sentinel(self) -> bool:
        """Backwards compat: True if "none" in sentinel_additions."""
        return "none" in self.sentinel_additions

    @property
    def empty_as_sentinel(self) -> bool:
        """Backwards compat: True if "empty" in sentinel_additions."""
        return "empty" in self.sentinel_additions

    def is_sentinel(self, value: Any) -> bool:
        """Check if value is sentinel per this config."""
        return is_sentinel(value, self.sentinel_additions)

    def is_sentinel_field(self, obj: Any, field_name: str) -> bool:
        """Check if a field holds a sentinel value on the given object.

        Raises:
            ValueError: If field_name not in allowed().
        """
        if hasattr(obj, "allowed") and field_name not in obj.allowed():
            raise ValueError(f"Invalid field name: {field_name}")
        return self.is_sentinel(getattr(obj, field_name, Undefined))


class _SentinelMixin:
    """Shared sentinel-aware serialization logic for Params and DataClass.

    Provides: allowed(), _is_sentinel(), _normalize_value(), _validate(),
    to_dict(), with_updates(), is_sentinel_field(), __hash__().

    Subclasses must define:
        _config: ClassVar[ModelConfig]
        _allowed_keys: ClassVar[set[str]]
    """

    __slots__ = ()

    @classmethod
    def allowed(cls) -> set[str]:
        """Return set of valid field names (excludes private/ClassVar)."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = set(
            f.name for f in fields(cls) if not f.name.startswith("_")
        )
        return cls._allowed_keys

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if value is sentinel per _config settings."""
        return is_sentinel(value, cls._config.sentinel_additions)

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """Normalize value for serialization (enum to value if configured)."""
        if cls._config.use_enum_values and isinstance(value, _Enum):
            return value.value
        return value

    def _validate(self) -> None:
        """Validate fields per _config. Raises ValueError if strict violations."""
        missing: list[str] = []
        for k in self.allowed():
            if self._config.strict and self._is_sentinel(getattr(self, k, Unset)):
                missing.append(k)
            if self._config.prefill_unset and is_undefined(
                getattr(self, k, Undefined)
            ):
                object.__setattr__(self, k, Unset)
        if missing:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing)}"
            )

    def to_dict(
        self,
        mode: Literal["python", "json"] = "python",
        exclude: set[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize to dict, excluding sentinel values.

        Args:
            mode: "python" returns Python objects, "json" returns JSON-safe dict.
            exclude: Field names to skip.
        """
        data = {}
        exclude = exclude or set()
        for k in type(self).allowed():
            if k not in exclude:
                v = getattr(self, k, Undefined)
                if not self._is_sentinel(v):
                    data[k] = self._normalize_value(v)
        if mode == "json":
            from .._json_dump import json_dumps

            return json_dumps(data, as_loaded=True, **kwargs)
        return data

    def with_updates(
        self,
        copy_containers: Literal["shallow", "deep"] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Return new instance with updated fields.

        Args:
            copy_containers: "shallow", "deep", or None (share references).
            **kwargs: Field values to update.
        """
        dict_ = self.to_dict()

        def _out(d: dict):
            d.update(kwargs)
            return type(self)(**d)

        if copy_containers is None:
            return _out(dict_)

        if copy_containers == "shallow":
            for k, v in dict_.items():
                if k not in kwargs and isinstance(
                    v, (MutableSequence, MutableMapping, MutableSet)
                ):
                    dict_[k] = v.copy() if hasattr(v, "copy") else list(v)
            return _out(dict_)

        if copy_containers == "deep":
            import copy

            for k, v in dict_.items():
                if k not in kwargs and isinstance(
                    v, (MutableSequence, MutableMapping, MutableSet)
                ):
                    dict_[k] = copy.deepcopy(v)
            return _out(dict_)

        raise ValueError(
            f"Invalid copy_containers: {copy_containers!r}. "
            "Must be 'shallow', 'deep', or None."
        )

    def is_sentinel_field(self, field_name: str) -> bool:
        """Check if field holds a sentinel value.

        Raises:
            ValueError: If field_name not in allowed().
        """
        if field_name not in self.allowed():
            raise ValueError(f"Invalid field name: {field_name}")
        return self._is_sentinel(getattr(self, field_name, Undefined))

    def __hash__(self) -> int:
        """Hash based on serialized dict contents."""
        from .._hash import hash_obj

        return hash_obj(self.to_dict())


@dataclass(slots=True, frozen=True, init=False)
class Params(_SentinelMixin):
    """Immutable parameter container with sentinel-aware serialization.

    Frozen dataclass with custom __init__ for sentinel support.
    Subclass and override _config for custom behavior.

    Example:
        >>> @dataclass(slots=True, frozen=True, init=False)
        ... class MyParams(Params):
        ...     _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
        ...     param1: str
        ...     param2: int
    """

    _config: ClassVar[ModelConfig] = ModelConfig()
    _allowed_keys: ClassVar[set[str]] = set()

    def __init__(self, **kwargs: Any):
        """Initialize from kwargs with validation.

        Raises:
            ValueError: If kwargs contains invalid field names or strict mode and required fields missing.
        """
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            if f.name not in kwargs:
                from dataclasses import MISSING as _MISSING

                if f.default is not _MISSING:
                    object.__setattr__(self, f.name, f.default)
                elif f.default_factory is not _MISSING:
                    object.__setattr__(self, f.name, f.default_factory())

        for k, v in kwargs.items():
            if k in self.allowed():
                object.__setattr__(self, k, v)
            else:
                raise ValueError(f"Invalid parameter: {k}")

        self._validate()

    def default_kw(self) -> Any:
        """Return dict with kwargs/kw fields merged into top level."""
        dict_ = self.to_dict()
        kw_ = {}
        kw_.update(dict_.pop("kwargs", {}))
        kw_.update(dict_.pop("kw", {}))
        dict_.update(kw_)
        return dict_

    def __eq__(self, other: object) -> bool:
        """Equality via hash. Returns NotImplemented for incompatible types."""
        if not isinstance(other, Params):
            return NotImplemented
        return hash(self) == hash(other)


@dataclass(slots=True)
class DataClass(_SentinelMixin):
    """Mutable dataclass with sentinel-aware serialization.

    Like Params but mutable (not frozen). Validates on __post_init__.
    Subclass and override _config for custom behavior.

    Example:
        >>> @dataclass(slots=True)
        ... class MyDataClass(DataClass):
        ...     _config: ClassVar[ModelConfig] = ModelConfig(strict=True, prefill_unset=False)
        ...     field1: str
        ...     field2: int
    """

    _config: ClassVar[ModelConfig] = ModelConfig()
    _allowed_keys: ClassVar[set[str]] = set()

    def __post_init__(self):
        """Validate fields after initialization."""
        self._validate()

    def __eq__(self, other: object) -> bool:
        """Equality via hash. Returns NotImplemented for incompatible types."""
        if not isinstance(other, DataClass):
            return NotImplemented
        return hash(self) == hash(other)


KeysLike = Sequence[str] | KeysDict
"""Type alias for key specifications: sequence of names or KeysDict."""


@dataclass(slots=True, frozen=True)
class Meta:
    """Immutable key-value metadata container.

    Hashable for use in sets/dicts. Special handling for callables
    (hashed by id for identity semantics).
    """

    key: str
    value: Any

    @override
    def __hash__(self) -> int:
        """Hash by (key, value). Callables use id(), unhashables use str()."""
        if callable(self.value):
            return hash((self.key, id(self.value)))
        try:
            return hash((self.key, self.value))
        except TypeError:
            return hash((self.key, str(self.value)))

    @override
    def __eq__(self, other: object) -> bool:
        """Equality by key then value. Callables compared by id."""
        if not isinstance(other, Meta):
            return NotImplemented
        if self.key != other.key:
            return False
        if callable(self.value) and callable(other.value):
            return id(self.value) == id(other.value)
        return bool(self.value == other.value)
