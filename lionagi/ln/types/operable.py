from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._sentinel import MaybeUnset, Unset

if TYPE_CHECKING:
    from .spec import Spec

__all__ = ("Operable",)


@dataclass(frozen=True, slots=True)
class Operable:
    __op_fields__: frozenset["Spec"] = frozenset()

    name: str | None = None
    sha256: str | None = None

    def allowed(self) -> set[str]:
        return {i.name for i in self.__op_fields__}

    def check_allowed(self, *args, as_boolean: bool = False):
        if not set(args).issubset(self.allowed()):
            if as_boolean:
                return False
            raise ValueError(
                "Some specified fields are not allowed: "
                f"{set(args).difference(self.allowed())}"
            )
        return True

    def get(self, key: str, /, default=Unset) -> MaybeUnset["Spec"]:
        if not self.check_allowed(key, as_boolean=True):
            return default
        for i in self.__op_fields__:
            if i.name == key:
                return i

    def get_specs(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> frozenset["Spec"]:

        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both include and exclude")

        if include:
            if self.check_allowed(*include, as_boolean=True) is False:
                raise ValueError(
                    "Some specified fields are not allowed: "
                    f"{set(include).difference(self.allowed())}"
                )
            return frozenset(
                {self.get(i) for i in set(include) if self.get(i) is not Unset}
            )

        if exclude:
            _discards = {
                self.get(i) for i in set(exclude) if self.get(i) is not Unset
            }
            return frozenset(set(self.__op_fields__).difference(_discards))

        return self.__op_fields__.copy()

    def create_model(
        self,
        adapter: Literal["pydantic"] = "pydantic",
        model_name: str | None = None,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        **kw,
    ):
        match adapter:
            case "pydantic":
                from lionagi.adapters.spec_adapters.pydantic_field import (
                    PydanticSpecAdapter,
                )

                kws = {
                    "model_name": model_name or self.name or "DynamicModel",
                    "include": include,
                    "exclude": exclude,
                    **kw,
                }
                return PydanticSpecAdapter.create_model(self, **kws)
            case _:
                raise ValueError(f"Unsupported adapter: {adapter}")
