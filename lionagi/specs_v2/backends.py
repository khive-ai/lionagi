"""Backend implementations: Pydantic (free), Rust (paid), Cloud (enterprise).

This module provides concrete implementations of validation backends.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .spec import FieldSpec


class PydanticBackend:
    """Free tier: Basic Pydantic validation.

    This backend uses Pydantic for runtime validation. It provides
    good performance and covers most common use cases, but doesn't
    provide formal verification guarantees.

    Examples:
        >>> from lionagi.specs_v2 import BackendRegistry, PydanticBackend
        >>> BackendRegistry.register("pydantic", PydanticBackend())
        >>> spec = FieldSpec(int, {"min": 0, "max": 100})
        >>> value = BackendRegistry.validate(spec, 42, backend="pydantic")
    """

    def validate(self, spec: FieldSpec, value: Any) -> Any:
        """Validate value using Pydantic.

        Args:
            spec: Field specification with constraints
            value: Value to validate

        Returns:
            Validated value (may be coerced)

        Raises:
            ValidationError: If validation fails
        """
        from pydantic import Field, ValidationError, create_model

        # Create Pydantic field from spec
        field_info = self.create_field(spec)

        # Handle nullable
        annotation = spec.type
        if spec.constraints.get("nullable"):
            annotation = Optional[annotation]

        # Create a temporary model for validation
        TempModel = create_model("TempModel", value=(annotation, field_info))

        try:
            # Validate
            validated = TempModel(value=value)
            return validated.value
        except ValidationError as e:
            # Re-raise with clearer context
            raise ValidationError(
                f"Validation failed for {spec}: {e}", model=TempModel
            ) from e

    def create_field(self, spec: FieldSpec) -> Any:
        """Create Pydantic FieldInfo from FieldSpec.

        Args:
            spec: Field specification

        Returns:
            Pydantic Field object
        """
        from pydantic import Field

        constraints = spec.constraints.copy()

        # Extract common Pydantic field parameters
        field_kwargs = {}

        # Numeric constraints
        if "min" in constraints:
            field_kwargs["ge"] = constraints["min"]
        if "max" in constraints:
            field_kwargs["le"] = constraints["max"]

        # String constraints
        if "pattern" in constraints:
            field_kwargs["pattern"] = constraints["pattern"]
        if "min_length" in constraints:
            field_kwargs["min_length"] = constraints["min_length"]
        if "max_length" in constraints:
            field_kwargs["max_length"] = constraints["max_length"]

        # Description
        if "description" in constraints:
            field_kwargs["description"] = constraints["description"]

        # Default value
        if "default" in constraints:
            field_kwargs["default"] = constraints["default"]
        else:
            field_kwargs["default"] = ...  # Required field

        return Field(**field_kwargs)


class RustBackend:
    """Paid tier: Local Rust formal verification via PyO3.

    This backend uses Rust for formal verification, providing
    mathematical guarantees about validation correctness.

    Requires: `pip install lionagi[rust]` to install the Rust extension.

    Examples:
        >>> try:
        ...     from lionagi.specs_v2 import BackendRegistry, RustBackend
        ...     BackendRegistry.register("rust", RustBackend())
        ...     spec = FieldSpec(int, {"min": 0, "max": 100})
        ...     value = BackendRegistry.validate(spec, 42, backend="rust")
        ... except ImportError:
        ...     print("Rust backend not installed")
    """

    def __init__(self):
        """Initialize Rust backend.

        Raises:
            ImportError: If Rust extension not installed
        """
        try:
            import lionbridge  # Rust PyO3 extension

            self.lionbridge = lionbridge
        except ImportError as e:
            raise ImportError(
                "Rust backend requires lionagi[rust] extra. "
                "Install with: pip install 'lionagi[rust]'"
            ) from e

    def validate(self, spec: FieldSpec, value: Any) -> Any:
        """Validate value using Rust formal verification.

        Args:
            spec: Field specification with constraints
            value: Value to validate

        Returns:
            Validated value with formal guarantee

        Raises:
            ValidationError: If validation fails
        """
        # Serialize spec to JSON for Rust
        spec_json = spec.to_dict()

        # Call Rust validation via PyO3
        try:
            result = self.lionbridge.validate(spec_json, value)
            return result
        except Exception as e:
            raise ValidationError(
                f"Rust validation failed for {spec}: {e}"
            ) from e

    def create_field(self, spec: FieldSpec) -> Any:
        """Create Rust field representation.

        Args:
            spec: Field specification

        Returns:
            Rust field handle (opaque)
        """
        spec_json = spec.to_dict()
        return self.lionbridge.create_field(spec_json)


class CloudBackend:
    """Enterprise tier: Cloud validation with full guarantees.

    This backend sends validation requests to LionAGI's cloud service,
    which provides:
    - Formal verification guarantees
    - Audit trails
    - Compliance certifications
    - Insurance-backed guarantees

    Requires: API key from https://lionagi.ai

    Examples:
        >>> from lionagi.specs_v2 import BackendRegistry, CloudBackend
        >>> cloud = CloudBackend(
        ...     api_key=os.getenv("LIONAGI_API_KEY"),
        ...     endpoint="https://api.lionagi.ai"
        ... )
        >>> BackendRegistry.register("cloud", cloud)
        >>> spec = FieldSpec(int, {"min": 0, "max": 100})
        >>> value = BackendRegistry.validate(spec, 42, backend="cloud")
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = "https://api.lionagi.ai",
    ):
        """Initialize cloud backend.

        Args:
            api_key: LionAGI API key (or set LIONAGI_API_KEY env var)
            endpoint: API endpoint URL

        Raises:
            ValueError: If API key not provided
        """
        self.api_key = api_key or os.getenv("LIONAGI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cloud backend requires API key. "
                "Set LIONAGI_API_KEY environment variable or pass api_key parameter."
            )

        self.endpoint = endpoint
        self._session = None

    @property
    def session(self):
        """Lazy-load requests session."""
        if self._session is None:
            try:
                import requests

                self._session = requests.Session()
                self._session.headers.update(
                    {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    }
                )
            except ImportError as e:
                raise ImportError(
                    "Cloud backend requires 'requests' library. "
                    "Install with: pip install requests"
                ) from e
        return self._session

    def validate(self, spec: FieldSpec, value: Any) -> Any:
        """Validate value using cloud service.

        Args:
            spec: Field specification with constraints
            value: Value to validate

        Returns:
            Validated value with enterprise guarantees

        Raises:
            ValidationError: If validation fails
            RuntimeError: If API request fails
        """
        # Serialize spec and value
        payload = {
            "spec": spec.to_dict(),
            "value": value,
        }

        # Send to cloud API
        try:
            response = self.session.post(
                f"{self.endpoint}/v1/validate",
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            if result.get("valid"):
                return result["validated_value"]
            else:
                raise ValidationError(
                    f"Cloud validation failed: {result.get('error')}"
                )

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise RuntimeError(f"Cloud API request failed: {e}") from e

    def create_field(self, spec: FieldSpec) -> Any:
        """Create field via cloud API.

        Args:
            spec: Field specification

        Returns:
            Field handle from cloud service
        """
        payload = {"spec": spec.to_dict()}

        response = self.session.post(
            f"{self.endpoint}/v1/fields",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        return response.json()


class ValidationError(Exception):
    """Validation error raised by backends."""

    pass
