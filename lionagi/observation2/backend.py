# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Backend: Validation execution engines.

Simple backends that perform actual validation without knowledge of
capabilities or invariants.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Protocol

if TYPE_CHECKING:
    from ..observation.validator import Validator  # Current validator
    from .form import Form


@dataclass
class ValidationResult:
    """Result from backend validation."""

    success: bool
    fields: Dict[str, Any]
    errors: Dict[str, list] = field(default_factory=dict)
    warnings: Dict[str, list] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationBackend(Protocol):
    """
    Protocol for validation backends (FFI-optimized).

    Following opinion 2: Backends receive serialized data, not full forms.
    This enables efficient FFI and clean separation.
    """

    async def execute_batch(
        self, data: Dict[str, Any], rules: Dict[str, dict]
    ) -> ValidationResult:
        """Execute validation on serialized data."""
        ...


class PydanticBackend:
    """
    Backend using current Pydantic-based validator.

    This wraps the existing validation system for compatibility.
    """

    def __init__(self, validator: Validator = None):
        """Initialize with existing validator."""
        # Import here to avoid circular dependency
        if validator is None:
            from ..observation.validator import Validator

            validator = Validator()
        self.validator = validator

    async def execute_batch(
        self, data: Dict[str, Any], rules: Dict[str, dict]
    ) -> ValidationResult:
        """Execute validation using Pydantic validator."""
        result = ValidationResult(success=True, fields=data.copy())

        # Validate each field
        for field_name, value in data.items():
            val_result = await self.validator.validate(
                field=field_name, value=value, fix=False
            )

            if val_result.success:
                result.fields[field_name] = val_result.value
            else:
                result.success = False
                result.errors[field_name] = val_result.errors

        return result


class TypeBackend:
    """
    Pure Python type checking backend.

    Simple type validation without external dependencies.
    """

    async def execute_batch(
        self, data: Dict[str, Any], rules: Dict[str, dict]
    ) -> ValidationResult:
        """Validate types match specifications."""
        result = ValidationResult(success=True, fields=data.copy())

        for field_name, rule in rules.items():
            if field_name not in data:
                continue

            value = data[field_name]
            if value is None:
                continue

            expected_type = rule.get("type")
            if not expected_type:
                continue

            # Type check
            if not isinstance(value, expected_type):
                # Try coercion
                try:
                    result.fields[field_name] = expected_type(value)
                    result.warnings[field_name] = [
                        f"Coerced to {expected_type.__name__}"
                    ]
                except (ValueError, TypeError):
                    result.success = False
                    result.errors[field_name] = [
                        f"Expected {expected_type.__name__}, got {type(value).__name__}"
                    ]

        return result


class SimpleBackend:
    """
    Minimal backend for testing.

    Does basic validation without any external dependencies.
    """

    async def execute_batch(
        self, data: Dict[str, Any], rules: Dict[str, dict]
    ) -> ValidationResult:
        """Minimal validation - just checks required fields exist."""
        result = ValidationResult(success=True, fields=data.copy())

        for field_name, rule in rules.items():
            if field_name not in data:
                default = rule.get("default")
                if default is not None:
                    result.fields[field_name] = default
                    result.warnings[field_name] = ["Used default value"]

        return result


class RustBackend:
    """
    Future Rust backend for high performance.

    This is a placeholder showing the interface.
    """

    def __init__(self, license_key: str):
        """Initialize with commercial license."""
        self.license_key = license_key
        # Future: self._ffi = load_rust_validator(license_key)

    async def execute_batch(
        self, data: Dict[str, Any], rules: Dict[str, dict]
    ) -> ValidationResult:
        """
        Validate using Rust for 10-100x performance.

        Future implementation will:
        1. Serialize data+rules with msgspec (not JSON)
        2. Pass to Rust via PyO3 FFI
        3. Execute in Rust with released GIL
        4. Return binary result
        """
        # Future: Use msgspec for zero-copy serialization
        # import msgspec
        # payload = msgspec.encode((data, rules))
        # result_bytes = await self._ffi.validate_msgpack(payload)
        # return msgspec.decode(result_bytes, type=ValidationResult)

        raise NotImplementedError(
            "Rust backend coming soon as commercial offering. "
            "Use PydanticBackend or TypeBackend for now."
        )
