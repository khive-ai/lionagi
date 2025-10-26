"""Operable - Container for Spec collections with model generation.

This module provides the Operable class for managing collections of Spec objects
and generating framework-specific models via adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._sentinel import MaybeUnset, Unset

if TYPE_CHECKING:
    from .spec import Spec

__all__ = ("Operable",)


@dataclass(frozen=True, slots=True, init=False)
class Operable:
    """Collection of Spec objects with model generation capabilities.

    Operable manages an ordered collection of Spec objects and provides
    methods to generate framework-specific models via adapters.

    Attributes:
        __op_fields__: Ordered tuple of Spec objects
        name: Optional name for this operable

    Example:
        >>> from lionagi.ln.types import Spec, Operable
        >>> specs = (
        ...     Spec(str, name="username"),
        ...     Spec(int, name="age"),
        ... )
        >>> operable = Operable(specs, name="User")
        >>> UserModel = operable.create_model(adapter="pydantic")
    """

    __op_fields__: tuple[Spec, ...]
    name: str | None

    def __init__(
        self,
        specs: tuple[Spec, ...] | list[Spec] = (),
        *,
        name: str | None = None,
    ):
        """Initialize Operable with specs.

        Args:
            specs: Tuple or list of Spec objects
            name: Optional name for this operable

        Raises:
            TypeError: If specs contains non-Spec objects
            ValueError: If specs contains duplicate field names
        """
        # Import here to avoid circular import
        from .spec import Spec

        # Convert to tuple if list
        if isinstance(specs, list):
            specs = tuple(specs)

        # Validate all items are Spec objects
        for i, item in enumerate(specs):
            if not isinstance(item, Spec):
                raise TypeError(
                    f"All specs must be Spec objects, got {type(item).__name__} "
                    f"at index {i}"
                )

        # Check for duplicate names
        names = [s.name for s in specs if s.name is not None]
        if len(names) != len(set(names)):
            from collections import Counter

            duplicates = [
                name for name, count in Counter(names).items() if count > 1
            ]
            raise ValueError(
                f"Duplicate field names found: {duplicates}. "
                "Each spec must have a unique name."
            )

        object.__setattr__(self, "__op_fields__", specs)
        object.__setattr__(self, "name", name)

    def allowed(self) -> set[str]:
        """Get set of allowed field names.

        Returns:
            Set of field names from specs
        """
        return {i.name for i in self.__op_fields__}

    def check_allowed(self, *args, as_boolean: bool = False):
        """Check if field names are allowed.

        Args:
            *args: Field names to check
            as_boolean: If True, return bool instead of raising

        Returns:
            True if all allowed, False if as_boolean=True and not all allowed

        Raises:
            ValueError: If as_boolean=False and not all allowed
        """
        if not set(args).issubset(self.allowed()):
            if as_boolean:
                return False
            raise ValueError(
                "Some specified fields are not allowed: "
                f"{set(args).difference(self.allowed())}"
            )
        return True

    def get(self, key: str, /, default=Unset) -> MaybeUnset[Spec]:
        """Get Spec by field name.

        Args:
            key: Field name
            default: Default value if not found

        Returns:
            Spec object or default
        """
        if not self.check_allowed(key, as_boolean=True):
            return default
        for i in self.__op_fields__:
            if i.name == key:
                return i
        return default

    def get_specs(
        self,
        *,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> tuple[Spec, ...]:
        """Get filtered tuple of Specs.

        Args:
            include: Only include these field names
            exclude: Exclude these field names

        Returns:
            Filtered tuple of Specs

        Raises:
            ValueError: If both include and exclude specified, or if invalid names
        """
        if include is not None and exclude is not None:
            raise ValueError("Cannot specify both include and exclude")

        if include:
            if self.check_allowed(*include, as_boolean=True) is False:
                raise ValueError(
                    "Some specified fields are not allowed: "
                    f"{set(include).difference(self.allowed())}"
                )
            return tuple(
                self.get(i) for i in include if self.get(i) is not Unset
            )

        if exclude:
            _discards = {
                self.get(i) for i in exclude if self.get(i) is not Unset
            }
            return tuple(s for s in self.__op_fields__ if s not in _discards)

        return self.__op_fields__

    def create_model(
        self,
        adapter: Literal["pydantic"] = "pydantic",
        model_name: str | None = None,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        **kw,
    ):
        """Create framework-specific model from specs.

        Args:
            adapter: Adapter type (currently only "pydantic")
            model_name: Name for generated model
            include: Only include these fields
            exclude: Exclude these fields
            **kw: Additional adapter-specific kwargs

        Returns:
            Generated model class

        Raises:
            ImportError: If adapter not installed
            ValueError: If adapter not supported
        """
        match adapter:
            case "pydantic":
                try:
                    from lionagi.adapters.spec_adapters import (
                        PydanticSpecAdapter,
                    )
                except ImportError as e:
                    raise ImportError(
                        "PydanticSpecAdapter requires Pydantic. "
                        "Install with: pip install pydantic"
                    ) from e

                kws = {
                    "model_name": model_name or self.name or "DynamicModel",
                    "include": include,
                    "exclude": exclude,
                    **kw,
                }
                return PydanticSpecAdapter.create_model(self, **kws)
            case _:
                raise ValueError(f"Unsupported adapter: {adapter}")
