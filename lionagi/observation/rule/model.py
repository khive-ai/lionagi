# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import TYPE_CHECKING, Any, Type

from lionagi._errors import ValidationError
from lionagi.ln.fuzzy import fuzzy_validate_pydantic

from .base import Rule, register_rule

if TYPE_CHECKING:
    from pydantic import BaseModel


@register_rule("model")
@register_rule("pydantic")
@register_rule("basemodel")
class ModelRule(Rule):
    """Rule for validating Pydantic models using fuzzy validation."""

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: str = None,
        **kwargs,
    ) -> bool:
        """Apply to fields with model/pydantic annotation or BaseModel values."""
        model_types = {"model", "pydantic", "basemodel"}
        if annotation and annotation.lower() in model_types:
            return True

        # Check if value is already a BaseModel instance
        if hasattr(value, "model_validate") and hasattr(value, "model_fields"):
            return True

        # Check if model_type is provided in kwargs
        return "model_type" in kwargs

    async def validate(self, value: Any, **kwargs) -> "BaseModel":
        """Validate and convert value to Pydantic model.

        Args:
            value: Value to validate (can be dict, JSON string, BaseModel, etc.)
            **kwargs: Validation options:
                - model_type: Pydantic model class (required)
                - nullable: Allow None values
                - default: Default value if None

        Returns:
            Validated Pydantic model instance

        Raises:
            ValidationError: If not a valid model or constraints violated
        """
        if value is None:
            if kwargs.get("nullable", False):
                return None
            if kwargs.get("default") is not None:
                return kwargs["default"]
            raise ValidationError("Model value cannot be None")

        # Get model type from kwargs
        model_type = kwargs.get("model_type")
        if not model_type:
            raise ValidationError("model_type must be provided in kwargs")

        # Validate that model_type is a Pydantic model
        if not (
            inspect.isclass(model_type)
            and hasattr(model_type, "model_validate")
            and hasattr(model_type, "model_fields")
        ):
            raise ValidationError(
                f"model_type must be a Pydantic BaseModel class, got {type(model_type)}"
            )

        # If value is already the correct model type, return it
        if isinstance(value, model_type):
            return value

        # If value is a different BaseModel, convert to dict first
        if hasattr(value, "model_dump"):
            value = value.model_dump()

        try:
            # Just validate with Pydantic - no fuzzy matching in validation
            result = model_type.model_validate(value)
            return result

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid model: {e}")

    async def fix(self, value: Any, **kwargs) -> "BaseModel":
        """Attempt to fix model value using fuzzy validation.

        Args:
            value: Value to fix
            **kwargs: Fix options:
                - model_type: Pydantic model class (required)
                - field_defaults: Default values for missing fields
                - similarity_threshold: Similarity threshold for fixing (default 0.9)
                - default: Default model instance if conversion fails

        Returns:
            Fixed model instance
        """
        if value is None:
            # Return default model instance if provided
            default = kwargs.get("default")
            if default is not None:
                return default

            # Create empty model if model_type is available
            model_type = kwargs.get("model_type")
            if model_type:
                try:
                    return model_type()
                except:
                    pass

            raise ValidationError(
                "Cannot fix None value without default or model_type"
            )

        # Get model type from kwargs
        model_type = kwargs.get("model_type")
        if not model_type:
            raise ValidationError("model_type must be provided for fixing")

        # Get field defaults
        field_defaults = kwargs.get("field_defaults", {})

        try:
            # Try aggressive fuzzy matching for fixing
            if hasattr(value, "model_dump"):
                value = value.model_dump()

            # Merge with field defaults for missing keys
            if isinstance(value, dict):
                for field_name, default_value in field_defaults.items():
                    if field_name not in value:
                        value[field_name] = default_value

            # Handle different input types for fixing
            if isinstance(value, str):
                result = fuzzy_validate_pydantic(
                    value,
                    model_type=model_type,
                    fuzzy_parse=True,
                    fuzzy_match=True,  # Enable fuzzy matching for fixing
                    fuzzy_match_params={
                        "handle_unmatched": "remove",
                        "similarity_threshold": kwargs.get(
                            "similarity_threshold", 0.9
                        ),
                    },
                )
            else:
                # For dict and other inputs, handle fuzzy matching manually
                from lionagi.ln.fuzzy import fuzzy_match_keys

                model_data = fuzzy_match_keys(
                    value,
                    model_type.model_fields,
                    similarity_threshold=kwargs.get(
                        "similarity_threshold", 0.9
                    ),
                    handle_unmatched="remove",
                )
                result = model_type.model_validate(model_data)

            return result

        except:
            # Last resort - return default or create empty model
            default = kwargs.get("default")
            if default is not None:
                return default

            try:
                return model_type()
            except:
                raise ValidationError(
                    f"Cannot fix value for model type {model_type}"
                )
