from dataclasses import dataclass
from typing import TypeVar, Any
from uuid import UUID

from lionagi.ln import import_module
from lionagi.utils import is_union_type, union_members

from ._concepts import  Observable
from .ids import to_list_type

E = TypeVar("E", bound=Observable)


_ELEMENT = None

@dataclass(slots=True, frozen=True)
class TypeLock:
    item_type: set[type[E]] | None = None
    strict_type: bool = False

    def __post_init__(self):
        validated = self._validate_item_type(self.item_type)
        validated = None if not validated else frozenset(validated)

        object.__setattr__(self, "item_type", validated)
        object.__setattr__(
            self, "strict_type", self._validate_strict_type(self.strict_type)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_type": (
                None
                if self.item_type is None
                else [c.class_name(full=True) for c in self.item_type]
            ),
            "strict_type": "true" if self.strict_type else "false",
        }

    @staticmethod
    def _validate_strict_type(value, /) -> set[type[E]] | None:
        if isinstance(value, str):
            return True if value == "true" else False
        if not value:
            return False
        if value is True:
            return True
        raise ValueError("Invalid strict_type value, must be bool or str")

    @staticmethod
    def _validate_item_type(value, /) -> set[type[E]] | None:
        if value is None:
            return None

        value = to_list_type(value)
        out = set()

        for i in value:
            subcls = i
            if isinstance(i, str):
                try:
                    mod, imp = i.rsplit(".", 1)
                    subcls = import_module(mod, import_name=imp)
                except Exception as e:
                    raise ValueError(
                        f"Invalid item_type {i}, expected subclass of Observable."
                    ) from e

            if isinstance(subcls, type):
                if is_union_type(subcls):
                    members = union_members(subcls)
                    for m in members:
                        if not issubclass(m, Observable):
                            raise ValueError(
                                f"Invalid item_type {m.__name__}, expected subclass of Observable."
                            )
                        out.add(m)
                elif not issubclass(subcls, Observable):
                    raise ValueError(
                        f"Invalid item_type {subcls.__name__}, expected subclass of Observable."
                    )
                else:
                    out.add(subcls)
            else:
                raise ValueError(
                    f"Invalid item_type {i}, expected subclass of Observable."
                )

        if len(value) != len(set(value)):
            raise ValueError("Detected duplicated item types in item_type.")

        if len(value) > 0:
            return out

    def validate_items(self, value) -> dict[UUID, E]:
        if not value:
            return {}
        value = to_list_type(value)
        result = {}
        for i in value:
            if isinstance(i, dict):
                i = get_element_class().from_dict(i)
            if self.item_type:
                if self.strict_type:
                    if type(i) not in self.item_type:
                        raise TypeError(
                            f"Input {type(i).__name__} is not of valid type, expected to be one of {self.item_type}, no subclass allowed"
                        )
                else:
                    if not any(issubclass(type(i), t) for t in self.item_type):
                        raise TypeError(
                            f"Input {type(i).__name__} is not of valid type, expected to be one or subclasses of {self.item_type}"
                        )
            else:
                if not isinstance(i, Observable):
                    raise ValueError(f"Invalid pile item {i}")

            result[i.id] = i
        return result


def get_element_class() -> type[Observable]:
    global _ELEMENT
    if _ELEMENT is None:
        from .element import Element
        _ELEMENT = Element
    return _ELEMENT