from dataclasses import dataclass, field
from functools import partial
from typing import Any, ClassVar

from typing_extensions import override

from ._types import Undefined, Unset, is_sentinel

__all__ = ("Params", "DataClass")


class _SentinelAware:
    """Metaclass to ensure sentinels are handled correctly in subclasses."""

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

    @classmethod
    def allowed(cls) -> set[str]:
        """Return the keys of the parameters."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = {
            i for i in cls.__dataclass_fields__.keys() if not i.startswith("_")
        }
        return cls._allowed_keys

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if a value is a sentinel (Undefined or Unset)."""
        if value is None and cls._none_as_sentinel:
            return True
        return is_sentinel(value)

    def __post_init__(self):
        """Post-initialization to ensure all fields are set."""
        self._validate()

    def _validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, str]:
        data = {}
        for k in self.allowed():
            if not self._is_sentinel(v := getattr(self, k)):
                data[k] = v
        return data


@dataclass(slots=True, frozen=True, init=False)
class Params(_SentinelAware):
    """Base class for parameters used in various functions."""

    _func: ClassVar[Any] = Unset
    _particial_func: ClassVar[Any] = Unset

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

    def as_partial(self) -> Any:
        # if partial function is already cached, return it
        if self._particial_func is not Unset:
            return self._particial_func

        # validate is there is a function to apply
        if self._func is Unset:
            raise ValueError("No function defined for partial application.")
        if not callable(self._func):
            raise TypeError(
                f"Expected a callable, got {type(self._func).__name__}."
            )

        # create a partial function with the current parameters
        dict_ = self.to_dict()
        if not dict_:
            self._particial_func = self._func
            return self._func

        # handle kwargs if present, handle both 'kwargs' and 'kw'
        kw_ = {}
        kw_.update(dict_.pop("kwargs", {}))
        kw_.update(dict_.pop("kw", {}))
        dict_.update(kw_)
        self._particial_func = partial(self._func, **dict_)
        return self._particial_func


@dataclass(slots=True)
class DataClass(_SentinelAware):
    """A base class for data classes with strict parameter handling."""

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
