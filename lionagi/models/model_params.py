# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""ModelParams implementation using Params base class with aggressive caching.

This module provides ModelParams, a configuration class for dynamically creating
Pydantic models with explicit behavior and aggressive caching for performance.
"""

from __future__ import annotations

import inspect
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, ClassVar

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

from lionagi.ln.types import Params, Unset
from lionagi.utils import copy

from .field_model import FieldModel

__all__ = ("ModelParams",)

# Global cache configuration
_MODEL_CACHE_SIZE = int(os.environ.get("LIONAGI_MODEL_CACHE_SIZE", "1000"))
_model_cache: OrderedDict[int, type[BaseModel]] = OrderedDict()
_model_cache_lock = threading.RLock()


@dataclass(slots=True, frozen=True, init=False)
class ModelParams(Params):
    """Configuration for dynamically creating Pydantic models.

    This class provides a way to configure and create Pydantic models dynamically
    with explicit behavior (no silent conversions) and aggressive caching for
    performance optimization.

    Key features:
    - All unspecified fields are explicitly Unset (not None or empty)
    - No silent type conversions - fails fast on incorrect types
    - Aggressive caching of created models with LRU eviction
    - Thread-safe model creation and caching
    - Not directly instantiable - requires keyword arguments

    Attributes:
        name: Name for the generated model class
        parameter_fields: Field definitions for the model
        base_type: Base model class to inherit from
        field_models: List of FieldModel definitions
        exclude_fields: Fields to exclude from the final model
        field_descriptions: Custom descriptions for fields
        inherit_base: Whether to inherit from base_type
        config_dict: Pydantic model configuration
        doc: Docstring for the generated model
        frozen: Whether the model should be immutable

    Environment Variables:
        LIONAGI_MODEL_CACHE_SIZE: Maximum number of cached models (default: 1000)

    Examples:
        >>> params = ModelParams(
        ...     name="UserModel",
        ...     frozen=True,
        ...     field_models=[
        ...         FieldModel(str, name="username"),
        ...         FieldModel(int, name="age", default=0)
        ...     ],
        ...     doc="A user model with basic attributes."
        ... )
        >>> UserModel = params.create_new_model()

        >>> # All unspecified fields are Unset
        >>> params2 = ModelParams(name="SimpleModel")
        >>> assert params2.doc is Unset
        >>> assert params2.frozen is Unset
    """

    # Class configuration
    _prefill_unset: ClassVar[bool] = (
        True  # All unspecified fields become Unset
    )
    _strict: ClassVar[bool] = False  # Allow Unset values
    _cache_models: ClassVar[bool] = True  # Enable model caching by default

    # Public fields (all start as Unset when not provided)
    name: str | None
    parameter_fields: dict[str, FieldInfo]
    base_type: type[BaseModel]
    field_models: list[FieldModel]
    exclude_fields: list[str]
    field_descriptions: dict[str, str]
    inherit_base: bool
    config_dict: dict[str, Any] | None
    doc: str | None
    frozen: bool

    # Private state (excluded from allowed() since they start with _)
    _validators: dict[str, Callable] = dc_field(default_factory=dict)
    _use_keys: set[str] = dc_field(default_factory=set)
    _cache_key: int | None = dc_field(default=None)

    def _validate(self) -> None:
        """Validate types and setup model configuration.

        This method performs explicit type validation without silent conversions,
        then sets up the model configuration including field processing and
        cache key computation.

        Raises:
            TypeError: If any field has an incorrect type
            ValueError: If field configurations are invalid
        """
        # Call parent validation (handles _prefill_unset)
        Params._validate(self)

        # Type validation without silent conversion
        self._validate_types()

        # Setup model configuration
        self._setup_model_params()

        # Compute cache key if caching is enabled
        if self._cache_models:
            cache_key = self._compute_cache_key()
            object.__setattr__(self, "_cache_key", cache_key)

    def _validate_types(self) -> None:
        """Validate field types explicitly without conversion."""
        # Validate parameter_fields
        if self.parameter_fields is not Unset:
            # Treat None, [], {} as "not provided" - convert to Unset
            if self.parameter_fields in (None, [], {}):
                object.__setattr__(self, "parameter_fields", Unset)
            elif not isinstance(self.parameter_fields, dict):
                # Raise ValueError for backward compatibility
                raise ValueError(
                    f"parameter_fields must be dict, got {type(self.parameter_fields).__name__}"
                )
            else:
                # Validate dict contents
                for k, v in self.parameter_fields.items():
                    if not isinstance(k, str):
                        # Raise ValueError for backward compatibility
                        raise ValueError(
                            f"Field name must be str, got {type(k).__name__}"
                        )
                    if not isinstance(v, FieldInfo):
                        # Raise ValueError for backward compatibility
                        raise ValueError(
                            f"Field {k} must be FieldInfo, got {type(v).__name__}"
                        )

        # Validate base_type
        if self.base_type is not Unset:
            if not (
                inspect.isclass(self.base_type)
                and issubclass(self.base_type, BaseModel)
            ):
                # Raise ValueError for backward compatibility with tests
                raise ValueError(
                    f"base_type must be BaseModel subclass, got {self.base_type}"
                )

        # Validate field_models
        if self.field_models is not Unset:
            # Treat None or [] as "not provided" - convert to Unset
            if self.field_models in (None, []):
                object.__setattr__(self, "field_models", Unset)
            elif not isinstance(self.field_models, list):
                # Raise ValueError for backward compatibility
                raise ValueError(
                    f"field_models must be list, got {type(self.field_models).__name__}"
                )
            else:
                # Validate list contents
                for i, fm in enumerate(self.field_models):
                    if not isinstance(fm, FieldModel):
                        # Raise ValueError for backward compatibility
                        raise ValueError(
                            f"field_models[{i}] must be FieldModel, got {type(fm).__name__}"
                        )

        # Validate exclude_fields
        if self.exclude_fields is not Unset:
            # Treat None or [] as "not provided" - convert to Unset
            if self.exclude_fields in (None, []):
                object.__setattr__(self, "exclude_fields", Unset)
            elif not isinstance(self.exclude_fields, list):
                raise TypeError(
                    f"exclude_fields must be list, got {type(self.exclude_fields).__name__}"
                )
            else:
                # Validate list contents
                for i, field in enumerate(self.exclude_fields):
                    if not isinstance(field, str):
                        raise TypeError(
                            f"exclude_fields[{i}] must be str, got {type(field).__name__}"
                        )

        # Validate field_descriptions
        if self.field_descriptions is not Unset:
            # Treat None or {} as "not provided" - convert to Unset
            if self.field_descriptions in (None, {}):
                object.__setattr__(self, "field_descriptions", Unset)
            elif not isinstance(self.field_descriptions, dict):
                raise TypeError(
                    f"field_descriptions must be dict, got {type(self.field_descriptions).__name__}"
                )
            else:
                # Validate dict contents
                for k, v in self.field_descriptions.items():
                    if not isinstance(k, str):
                        raise TypeError(
                            f"field_descriptions key must be str, got {type(k).__name__}"
                        )
                    if not isinstance(v, str):
                        raise TypeError(
                            f"field_descriptions[{k}] must be str, got {type(v).__name__}"
                        )

        # Validate booleans
        if self.inherit_base is not Unset and not isinstance(
            self.inherit_base, bool
        ):
            raise TypeError(
                f"inherit_base must be bool, got {type(self.inherit_base).__name__}"
            )

        if self.frozen is not Unset and not isinstance(self.frozen, bool):
            raise TypeError(
                f"frozen must be bool, got {type(self.frozen).__name__}"
            )

        # Validate config_dict
        if self.config_dict is not Unset:
            if self.config_dict is not None and not isinstance(
                self.config_dict, dict
            ):
                raise TypeError(
                    f"config_dict must be dict or None, got {type(self.config_dict).__name__}"
                )

        # Validate doc
        if self.doc is not Unset:
            if self.doc is not None and not isinstance(self.doc, str):
                raise TypeError(
                    f"doc must be str or None, got {type(self.doc).__name__}"
                )

        # Validate name
        if self.name is not Unset:
            if self.name is not None and not isinstance(self.name, str):
                raise TypeError(
                    f"name must be str or None, got {type(self.name).__name__}"
                )

    def _setup_model_params(self) -> None:
        """Setup model parameters with explicit Unset handling.

        This method processes and merges field definitions from various sources,
        handles field exclusions, and prepares validators.
        """
        # Initialize or copy parameter_fields
        if self.parameter_fields is Unset:
            fields = {}
        else:
            fields = copy(self.parameter_fields)

        # Merge fields from base_type if provided and not excluded
        if self.base_type is not Unset:
            base_fields = copy(self.base_type.model_fields)
            # Only include base fields not in exclude list
            if self.exclude_fields is not Unset:
                base_fields = {
                    k: v
                    for k, v in base_fields.items()
                    if k not in self.exclude_fields
                }
            fields.update(base_fields)

        # Apply field descriptions to field_models BEFORE processing them
        if (
            self.field_descriptions is not Unset
            and self.field_models is not Unset
        ):
            updated_models = []
            for model in self.field_models:
                if model.name in self.field_descriptions:
                    updated_models.append(
                        model.with_description(
                            self.field_descriptions[model.name]
                        )
                    )
                else:
                    updated_models.append(model)
            object.__setattr__(self, "field_models", updated_models)

        # Process field_models
        validators = {}
        if self.field_models is not Unset:
            for fm in self.field_models:
                # Add field info
                fields[fm.name] = fm.field_info
                # Set annotation from FieldModel's base_type
                fields[fm.name].annotation = fm.base_type

                # Collect validators
                if fm.field_validator is not None:
                    validators.update(fm.field_validator)

        # Store processed fields back only if we have fields
        # Keep Unset if originally Unset and no fields were added
        if fields or self.parameter_fields is not Unset:
            object.__setattr__(self, "parameter_fields", fields)

        # Store validators
        if validators:
            object.__setattr__(self, "_validators", validators)

        # Calculate use_keys (fields to include in model)
        use_keys = set(fields.keys())
        if self.exclude_fields is not Unset:
            use_keys -= set(self.exclude_fields)
        object.__setattr__(self, "_use_keys", use_keys)

        # Auto-resolve name from base_type if not provided
        if self.name is Unset and self.base_type is not Unset:
            name = None
            if hasattr(self.base_type, "class_name"):
                name = self.base_type.class_name
                if callable(name):
                    name = name()
            elif inspect.isclass(self.base_type):
                name = self.base_type.__name__

            if name:
                object.__setattr__(self, "name", name)

    def _compute_cache_key(self) -> int:
        """Compute stable cache key for this configuration.

        The cache key includes all parameters that affect model creation
        to ensure cache correctness.

        Returns:
            Hash value representing this configuration
        """
        # Build tuple of all relevant state for hashing
        key_data = []

        # Name
        key_data.append(self.name if self.name is not Unset else None)

        # Parameter fields (sorted for stability)
        if self.parameter_fields is not Unset:
            # Convert FieldInfo to stable representation
            field_items = []
            for k, v in sorted(self.parameter_fields.items()):
                # Include key aspects of FieldInfo
                field_repr = (
                    k,
                    str(v.annotation) if v.annotation else None,
                    v.default if not callable(v.default) else "callable",
                    v.alias,
                    v.title,
                    v.description,
                )
                field_items.append(field_repr)
            key_data.append(tuple(field_items))
        else:
            key_data.append(())

        # Base type
        if self.base_type is not Unset:
            key_data.append(
                f"{self.base_type.__module__}.{self.base_type.__qualname__}"
            )
        else:
            key_data.append(None)

        # Field models (use their hash)
        if self.field_models is not Unset:
            key_data.append(tuple(hash(fm) for fm in self.field_models))
        else:
            key_data.append(())

        # Simple fields
        key_data.append(
            tuple(self.exclude_fields)
            if self.exclude_fields is not Unset
            else ()
        )
        key_data.append(
            tuple(sorted(self.field_descriptions.items()))
            if self.field_descriptions is not Unset
            else ()
        )
        key_data.append(
            self.inherit_base if self.inherit_base is not Unset else None
        )
        key_data.append(self.doc if self.doc is not Unset else None)
        key_data.append(self.frozen if self.frozen is not Unset else None)

        # Config dict
        if self.config_dict is not Unset and self.config_dict is not None:
            key_data.append(tuple(sorted(self.config_dict.items())))
        else:
            key_data.append(())

        # Validators (by their keys)
        if hasattr(self, "_validators") and self._validators:
            key_data.append(tuple(sorted(self._validators.keys())))
        else:
            key_data.append(())

        return hash(tuple(key_data))

    @property
    def use_fields(self) -> dict[str, tuple[type, FieldInfo]]:
        """Get field definitions to use in new model.

        Filters and prepares fields based on _use_keys set for model creation.

        Returns:
            Dictionary mapping field names to (type, FieldInfo) tuples
        """
        fields = (
            self.parameter_fields if self.parameter_fields is not Unset else {}
        )
        use_keys = self._use_keys if hasattr(self, "_use_keys") else set()
        return {
            k: (v.annotation, v)
            for k, v in fields.items()
            if not use_keys or k in use_keys
        }

    def create_new_model(self) -> type[BaseModel]:
        """Create new Pydantic model with specified configuration.

        This method generates a new Pydantic model class based on the configured
        parameters. Results are cached for performance when the same configuration
        is used multiple times.

        Returns:
            Newly created or cached Pydantic model class
        """
        # Check cache first if enabled
        if self._cache_models and self._cache_key is not None:
            with _model_cache_lock:
                if self._cache_key in _model_cache:
                    # Move to end for LRU
                    _model_cache.move_to_end(self._cache_key)
                    return _model_cache[self._cache_key]

        # Determine model name
        name = self.name if self.name is not Unset else "GeneratedModel"

        # Determine base class
        base_type = None
        if self.inherit_base is not Unset and self.inherit_base:
            if self.base_type is not Unset:
                base_type = self.base_type
                # Don't inherit if excluding base fields
                if self.exclude_fields is not Unset:
                    if any(
                        f in self.exclude_fields
                        for f in base_type.model_fields
                    ):
                        base_type = None

        # Create the model
        validators = (
            self._validators
            if hasattr(self, "_validators") and self._validators
            else None
        )
        model = create_model(
            name,
            __base__=base_type,
            __config__=(
                self.config_dict if self.config_dict is not Unset else None
            ),
            __doc__=self.doc if self.doc is not Unset else None,
            __validators__=validators,
            **self.use_fields,
        )

        # Apply frozen configuration
        if self.frozen is not Unset and self.frozen:
            model.model_config["frozen"] = True

        # Cache the result if enabled
        if self._cache_models and self._cache_key is not None:
            with _model_cache_lock:
                _model_cache[self._cache_key] = model

                # LRU eviction
                while len(_model_cache) > _MODEL_CACHE_SIZE:
                    try:
                        _model_cache.popitem(last=False)  # Remove oldest
                    except KeyError:
                        # Handle race condition
                        break

        return model
