# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from lionagi.libs.validate.fuzzy_match_keys import fuzzy_match_keys
from lionagi.models import FieldModel, ModelParams, OperableModel
from lionagi.utils import UNDEFINED, to_json


class Operative:
    """Class representing an operative that handles request and response models for operations.

    This implementation uses OperableModel internally for better performance while
    maintaining backward compatibility with the existing API.
    """

    def __init__(
        self,
        name: str | None = None,
        request_type: type[BaseModel] | None = None,
        response_type: type[BaseModel] | None = None,
        response_model: BaseModel | None = None,
        response_str_dict: dict | str | None = None,
        auto_retry_parse: bool = True,
        max_retries: int = 3,
        parse_kwargs: dict | None = None,
        request_params: (
            ModelParams | None
        ) = None,  # Deprecated, for backward compatibility
        **_kwargs,  # Ignored for backward compatibility
    ):
        """Initialize the Operative.

        Args:
            name: Name of the operative
            request_type: Pydantic model type for requests
            response_type: Pydantic model type for responses
            response_model: Current response model instance
            response_str_dict: Raw response string/dict
            auto_retry_parse: Whether to auto-retry parsing
            max_retries: Maximum parse retries
            parse_kwargs: Additional parse arguments
            request_params: Deprecated - use direct field addition
            response_params: Deprecated - use direct field addition
        """
        self.name = name
        self.request_type = request_type
        self.response_type = response_type
        self.response_model = response_model
        self.response_str_dict = response_str_dict
        self.auto_retry_parse = auto_retry_parse
        self.max_retries = max_retries
        self.parse_kwargs = parse_kwargs or {}
        self._should_retry = None

        # Internal OperableModel instances
        self._request_operable = OperableModel()
        self._response_operable = OperableModel()

        # Handle deprecated ModelParams for backward compatibility
        if request_params:
            self._init_from_model_params(request_params)

        # Set default name if not provided
        if not self.name:
            self.name = (
                self.request_type.__name__
                if self.request_type
                else "Operative"
            )

    def _init_from_model_params(self, params: ModelParams):
        """Initialize from ModelParams for backward compatibility."""
        # Add field models to the request operable
        if params.field_models:
            for field_model in params.field_models:
                self._request_operable.add_field(
                    field_model.name,
                    field_model=field_model,
                    annotation=field_model.base_type,
                )

        # Add parameter fields (skip if already added from field_models)
        if params.parameter_fields:
            for name, field_info in params.parameter_fields.items():
                if (
                    name not in (params.exclude_fields or [])
                    and name not in self._request_operable.all_fields
                ):
                    self._request_operable.add_field(
                        name, field_obj=field_info
                    )

        # Generate request_type if not provided
        if not self.request_type:
            exclude_fields = params.exclude_fields or []
            use_fields = set(self._request_operable.all_fields.keys()) - set(
                exclude_fields
            )
            self.request_type = self._request_operable.new_model(
                name=params.name or "RequestModel",
                use_fields=use_fields,
                base_type=params.base_type,
                frozen=params.frozen,
                config_dict=params.config_dict,
                doc=params.doc,
            )

        # Update name if not set
        if not self.name and params.name:
            self.name = params.name

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for backward compatibility.

        Note: This returns a Python dict, not JSON-serializable data.
        For JSON serialization, convert types appropriately.
        """
        return {
            "name": self.name,
            "request_type": self.request_type,  # Python class object
            "response_type": self.response_type,  # Python class object
            "response_model": self.response_model,
            "response_str_dict": self.response_str_dict,
            "auto_retry_parse": self.auto_retry_parse,
            "max_retries": self.max_retries,
            "parse_kwargs": self.parse_kwargs,
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for model_dump() - more appropriate name for non-Pydantic class."""
        return self.model_dump()

    def raise_validate_pydantic(self, text: str) -> None:
        """Validates and updates the response model using strict matching.

        Args:
            text (str): The text to validate and parse into the response model.

        Raises:
            Exception: If the validation fails.
        """
        d_ = to_json(text, fuzzy_parse=True)
        if isinstance(d_, list | tuple) and len(d_) == 1:
            d_ = d_[0]
        try:
            d_ = fuzzy_match_keys(
                d_, self.request_type.model_fields, handle_unmatched="raise"
            )
            d_ = {k: v for k, v in d_.items() if v != UNDEFINED}
            self.response_model = self.request_type.model_validate(d_)
            self._should_retry = False
        except Exception:
            self.response_str_dict = d_
            self._should_retry = True

    def force_validate_pydantic(self, text: str):
        """Forcibly validates and updates the response model, allowing unmatched fields.

        Args:
            text (str): The text to validate and parse into the response model.
        """
        d_ = text
        try:
            d_ = to_json(text, fuzzy_parse=True)
            if isinstance(d_, list | tuple) and len(d_) == 1:
                d_ = d_[0]
            d_ = fuzzy_match_keys(
                d_, self.request_type.model_fields, handle_unmatched="force"
            )
            d_ = {k: v for k, v in d_.items() if v != UNDEFINED}
            self.response_model = self.request_type.model_validate(d_)
            self._should_retry = False
        except Exception:
            self.response_str_dict = d_
            self.response_model = None
            self._should_retry = True

    def update_response_model(
        self, text: str | None = None, data: dict | None = None
    ) -> BaseModel | dict | str | None:
        """Updates the response model based on the provided text or data.

        Args:
            text (str, optional): The text to parse and validate.
            data (dict, optional): The data to update the response model with.

        Returns:
            BaseModel | dict | str | None: The updated response model or raw data.

        Raises:
            ValueError: If neither text nor data is provided.
        """
        if text is None and data is None:
            raise ValueError("Either text or data must be provided.")

        if text:
            self.response_str_dict = text
            try:
                self.raise_validate_pydantic(text)
            except Exception:
                self.force_validate_pydantic(text)

        if data and self.response_type:
            d_ = self.response_model.model_dump()
            d_.update(data)
            self.response_model = self.response_type.model_validate(d_)

        if not self.response_model and isinstance(
            self.response_str_dict, list
        ):
            try:
                self.response_model = [
                    self.request_type.model_validate(d_)
                    for d_ in self.response_str_dict
                ]
            except Exception:
                pass

        return self.response_model or self.response_str_dict

    def create_response_type(
        self,
        response_params: ModelParams | None = None,
        field_models: list[FieldModel] | None = None,
        parameter_fields: dict[str, FieldInfo] | None = None,
        exclude_fields: list[str] | None = None,
        field_descriptions: dict[str, str] | None = None,
        inherit_base: bool = True,
        config_dict: dict | None = None,
        doc: str | None = None,
        frozen: bool = False,
        validators: dict | None = None,
    ) -> None:
        """Creates a new response type based on the provided parameters.

        Args:
            response_params (ModelParams, optional): Parameters for the new response model.
            field_models (list[FieldModel], optional): List of field models.
            parameter_fields (dict[str, FieldInfo], optional): Dictionary of parameter fields.
            exclude_fields (list, optional): List of fields to exclude.
            field_descriptions (dict, optional): Dictionary of field descriptions.
            inherit_base (bool, optional): Whether to inherit the base model.
            config_dict (dict | None, optional): Configuration dictionary.
            doc (str | None, optional): Documentation string.
            frozen (bool, optional): Whether the model is frozen.
            validators (dict, optional): Dictionary of validators.
        """
        # Process response_params if provided (for backward compatibility)
        if response_params:
            # Extract values from ModelParams
            field_models = field_models or response_params.field_models
            parameter_fields = (
                parameter_fields or response_params.parameter_fields
            )
            exclude_fields = exclude_fields or response_params.exclude_fields
            field_descriptions = (
                field_descriptions or response_params.field_descriptions
            )
            inherit_base = (
                response_params.inherit_base if inherit_base else False
            )
            config_dict = config_dict or response_params.config_dict
            doc = doc or response_params.doc
            frozen = frozen or response_params.frozen

        # Clear response operable and rebuild
        self._response_operable = OperableModel()

        # Copy fields from request operable if inherit_base
        if inherit_base and self._request_operable:
            for (
                field_name,
                field_model,
            ) in self._request_operable.extra_field_models.items():
                self._response_operable.add_field(
                    field_name, field_model=field_model
                )

        # Add field models (skip if already exists from inheritance)
        if field_models:
            for field_model in field_models:
                if field_model.name not in self._response_operable.all_fields:
                    self._response_operable.add_field(
                        field_model.name,
                        field_model=field_model,
                        annotation=field_model.base_type,
                    )

        # Add parameter fields (skip if already added)
        if parameter_fields:
            for name, field_info in parameter_fields.items():
                if (
                    name not in (exclude_fields or [])
                    and name not in self._response_operable.all_fields
                ):
                    self._response_operable.add_field(
                        name, field_obj=field_info
                    )

        # Add validators if provided
        if validators:
            for field_name, validator in validators.items():
                if field_name in self._response_operable.all_fields:
                    field_model = (
                        self._response_operable.extra_field_models.get(
                            field_name
                        )
                    )
                    if field_model:
                        field_model.validator = validator

        # Generate response type
        exclude_fields = exclude_fields or []
        use_fields = set(self._response_operable.all_fields.keys()) - set(
            exclude_fields
        )

        # Determine base type - use request_type if inheriting and no specific base provided
        base_type = None
        if response_params and response_params.base_type:
            base_type = response_params.base_type
        elif inherit_base and self.request_type:
            base_type = self.request_type

        self.response_type = self._response_operable.new_model(
            name=(response_params.name if response_params else None)
            or "ResponseModel",
            use_fields=use_fields,
            base_type=base_type,
            frozen=frozen,
            config_dict=config_dict,
            doc=doc,
        )
