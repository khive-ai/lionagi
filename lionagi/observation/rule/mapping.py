# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, Dict

from lionagi._errors import ValidationError
from lionagi.ln.fuzzy import fuzzy_validate_mapping

from .base import Rule, register_rule


@register_rule("mapping")
@register_rule("dict")
@register_rule("map")
@register_rule("object")
class MappingRule(Rule):
    """Rule for validating mapping/dictionary values using fuzzy validation."""

    async def _custom_applies(
        self,
        field: str,
        value: Any,
        form: Any,
        annotation: str = None,
        **kwargs,
    ) -> bool:
        """Apply to fields with dict/mapping annotation."""
        mapping_types = {"dict", "mapping", "map", "object"}
        return (
            annotation in mapping_types
            if annotation
            else isinstance(value, Mapping)
        )

    async def validate(self, value: Any, **kwargs) -> Dict[str, Any]:
        """Validate and convert value to dictionary using fuzzy validation.

        Args:
            value: Value to validate (can be dict, JSON string, object, etc.)
            **kwargs: Validation options:
                - required_keys: Keys that must be present
                - optional_keys: Additional allowed keys
                - allow_extra: Whether to allow extra keys (default True)
                - fuzzy_match: Enable fuzzy key matching (default True)
                - similarity_threshold: Minimum similarity for fuzzy matching (default 0.85)
                - nullable: Allow None values
                - default: Default value if None

        Returns:
            Validated dictionary

        Raises:
            ValidationError: If not a valid mapping or constraints violated
        """
        if value is None:
            if kwargs.get("nullable", False):
                return None
            if kwargs.get("default") is not None:
                return kwargs["default"]
            raise ValidationError("Mapping value cannot be None")

        # Prepare keys for fuzzy validation
        required_keys = kwargs.get("required_keys", [])
        optional_keys = kwargs.get("optional_keys", [])
        all_keys = list(required_keys) + list(optional_keys)

        # Determine handle_unmatched behavior based on allow_extra
        allow_extra = kwargs.get("allow_extra", True)
        handle_unmatched = "ignore" if allow_extra else "remove"

        # If we need specific keys to be required, use strict mode
        strict = bool(required_keys)

        try:
            # Use fuzzy_validate_mapping to handle conversion and validation
            result = fuzzy_validate_mapping(
                value,
                (
                    all_keys if all_keys else None
                ),  # If no keys specified, just convert to dict
                similarity_threshold=kwargs.get("similarity_threshold", 0.85),
                fuzzy_match=kwargs.get("fuzzy_match", True),
                handle_unmatched=handle_unmatched,
                strict=strict,
                suppress_conversion_errors=False,
            )

            # Additional check for required keys if fuzzy matching didn't enforce them
            if required_keys:
                missing_keys = set(required_keys) - set(result.keys())
                if missing_keys:
                    raise ValidationError(
                        f"Missing required keys: {missing_keys}"
                    )

            return result

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid mapping: {e}")

    async def fix(self, value: Any, **kwargs) -> Dict[str, Any]:
        """Attempt to fix mapping value using fuzzy validation.

        Args:
            value: Value to fix
            **kwargs: Fix options:
                - required_keys: Keys that must be present
                - optional_keys: Additional allowed keys
                - key_defaults: Default values for missing keys
                - allow_extra: Whether to allow extra keys
                - default: Default value if conversion fails

        Returns:
            Fixed dictionary value
        """
        if value is None:
            # Return default or empty dict
            return kwargs.get("default", {})

        # Prepare keys and defaults
        required_keys = kwargs.get("required_keys", [])
        optional_keys = kwargs.get("optional_keys", [])
        all_keys = list(required_keys) + list(optional_keys)
        key_defaults = kwargs.get("key_defaults", {})

        # Determine handling strategy
        allow_extra = kwargs.get("allow_extra", True)
        handle_unmatched = (
            "force" if not allow_extra else "fill"
        )  # Force = fill + remove

        try:
            # Use fuzzy_validate_mapping with aggressive fixing
            result = fuzzy_validate_mapping(
                value,
                all_keys if all_keys else None,
                similarity_threshold=0.6,  # Lower threshold for fixing
                fuzzy_match=True,
                handle_unmatched=handle_unmatched,
                fill_value=None,
                fill_mapping=key_defaults,
                strict=False,  # Don't fail on missing keys
                suppress_conversion_errors=True,  # Always try to return something
            )

            # Ensure all required keys exist with defaults
            for key in required_keys:
                if key not in result:
                    result[key] = key_defaults.get(key)

            return result

        except:
            # Last resort - return default or empty dict
            return kwargs.get("default", {})
