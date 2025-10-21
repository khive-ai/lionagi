from typing import TYPE_CHECKING, Any, Literal

from pydapter import adapters

from lionagi.ln.types import Operable

if TYPE_CHECKING:
    from pydantic import BaseModel


class Operative:
    """Framework-agnostic operation handler with spec-based design.

    Manages request/response field specifications using Spec/Operable,
    delegating all framework-specific operations to adapters. Currently
    supports Pydantic; msgspec and Rust adapters planned.

    Architecture:
        Spec Definition → Operable Collection → Adapter → Framework Model
    """

    def __init__(
        self,
        name: str | None = None,
        adapter: Literal["pydantic"] = "pydantic",
        strict: bool = False,
        auto_retry_parse: bool = True,
        max_retries: int = 3,
        base_type: type["BaseModel"] | None = None,
        operable: Operable | None = None,
        request_exclude: set[str] | None = None,
    ):
        """Initialize Operative with a single immutable Operable.

        Args:
            name: Operation name
            adapter: Validation framework ("pydantic" only for now)
            strict: If True, raise on validation errors; if False, use fuzzy matching
            auto_retry_parse: Auto-retry validation on failure with fuzzy matching
            max_retries: Maximum validation retry attempts
            base_type: Base Pydantic model to extend
            operable: Single Operable with all fields
            request_exclude: Fields to exclude from request model (e.g., {"action_responses"})
        """
        self.name = name or (base_type.__name__ if base_type else "Operative")
        self.adapter = adapter
        self.strict = strict
        self.auto_retry_parse = auto_retry_parse
        self.max_retries = max_retries
        self.base_type = base_type

        # Single source of truth - one Operable with all fields
        self.operable = operable or Operable(name=self.name)
        self.request_exclude = request_exclude or set()

        # Materialized models (cached, adapter-specific)
        self._request_model_cls = None
        self._response_model_cls = None

        # Response state
        self.response_model = None  # Validated instance
        self.response_str_dict = None  # Raw/failed response
        self._should_retry = None

    # ---- Model Materialization (Adapter-Delegated) ----

    def create_request_model(self) -> type:
        """Materialize request specs into Pydantic model (excluding certain fields)."""
        if self._request_model_cls:
            return self._request_model_cls

        self._request_model_cls = self.operable.create_model(
            adapter=self.adapter,
            model_name=f"{self.name}Request",
            base_type=self.base_type,
            exclude=self.request_exclude,  # Exclude response-only fields
        )
        return self._request_model_cls

    def create_response_model(self) -> type:
        """Materialize all specs into Pydantic response model."""
        if self._response_model_cls:
            return self._response_model_cls

        # Ensure request model exists first (response extends request)
        if not self._request_model_cls:
            self.create_request_model()

        # Response model uses ALL fields and inherits from request
        self._response_model_cls = self.operable.create_model(
            model_name=f"{self.name}Response",
            base_type=self._request_model_cls,
        )

        return self._response_model_cls

    # ---- Validation (Adapter-Delegated) ----

    def validate_response(self, text: str, strict: bool | None = None) -> Any:
        """Validate response text using adapter.

        Args:
            text: Raw response text
            strict: If True, raise on validation errors; if False, use fuzzy matching.
                   If None, uses self.strict (default: False)
        """
        # Use instance-level strict setting if not overridden
        strict = self.strict if strict is None else strict

        # Ensure response model exists
        if not self._response_model_cls:
            self.create_response_model()

        if self.adapter == "pydantic":
            from lionagi.adapters.spec_adapters import PydanticSpecAdapter

            try:
                self.response_model = PydanticSpecAdapter.validate_response(
                    text,
                    self._response_model_cls,
                    strict=strict,
                    fuzzy_parse=True,
                )
                self._should_retry = False
                return self.response_model

            except Exception as e:
                # Store raw response
                self.response_str_dict = text
                self._should_retry = strict

                if strict:
                    raise e

                # Try fuzzy validation if auto-retry enabled
                if self.auto_retry_parse and not strict:
                    try:
                        self.response_model = (
                            PydanticSpecAdapter.validate_response(
                                text,
                                self._response_model_cls,
                                strict=False,
                                fuzzy_parse=True,
                            )
                        )
                        self._should_retry = False
                        return self.response_model
                    except Exception:
                        pass

                return None
        else:
            raise ValueError(f"Unsupported adapter: {self.adapter}")

    def update_response_model(
        self, text: str | None = None, data: dict | None = None
    ) -> Any:
        """Update response model from text or dict.

        Args:
            text: Raw response text to validate
            data: Dictionary updates to merge
        """
        if text is None and data is None:
            raise ValueError("Either text or data must be provided")

        if text:
            self.response_str_dict = text
            # Try strict, fall back to fuzzy
            self.validate_response(text, strict=False)

        if data and self._response_model_cls and self.response_model:
            if self.adapter == "pydantic":
                from lionagi.adapters.spec_adapters import PydanticSpecAdapter

                self.response_model = PydanticSpecAdapter.update_model(
                    self.response_model, data, self._response_model_cls
                )
            else:
                raise ValueError(f"Unsupported adapter: {self.adapter}")

        return self.response_model or self.response_str_dict

    @property
    def request_type(self) -> type | None:
        """Get request model type (backwards compatibility).

        Returns:
            Request model class or None if not materialized
        """
        return self._request_model_cls

    @property
    def response_type(self) -> type | None:
        """Get response model type (backwards compatibility).

        Returns:
            Response model class or None if not materialized
        """
        return self._response_model_cls


__all__ = ("Operative",)
