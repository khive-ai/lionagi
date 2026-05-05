from ._sentinel import (
    MaybeSentinel,
    MaybeUndefined,
    MaybeUnset,
    AdditionalSentinels,
    SingletonType,
    T,
    Undefined,
    UndefinedType,
    Unset,
    UnsetType,
    is_sentinel,
    is_undefined,
    is_unset,
    not_sentinel,
)
from .base import (
    DataClass,
    Enum,
    KeysDict,
    KeysLike,
    Meta,
    ModelConfig,
    Params,
)
from ._compat import StrEnum
from .operable import Operable
from .spec import CommonMeta, Spec


def __getattr__(name: str):
    if name == "HashableModel":
        from lionagi.models.hashable_model import HashableModel
        return HashableModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    # Sentinel types
    "Undefined",
    "Unset",
    "MaybeUndefined",
    "MaybeUnset",
    "MaybeSentinel",
    "AdditionalSentinels",
    "SingletonType",
    "UndefinedType",
    "UnsetType",
    "is_sentinel",
    "is_undefined",
    "is_unset",
    "not_sentinel",
    # Base classes
    "ModelConfig",
    "HashableModel",
    "StrEnum",
    "Enum",
    "Params",
    "DataClass",
    "Meta",
    "KeysDict",
    "KeysLike",
    "T",
    # Spec system
    "Spec",
    "CommonMeta",
    "Operable",
)
