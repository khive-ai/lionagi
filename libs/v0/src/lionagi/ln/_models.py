from dataclasses import dataclass, field
from typing import Any, ClassVar

from typing_extensions import override

from ._types import Undefined, Unset, is_sentinel

__all__ = ("Params", "DataClass")


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
