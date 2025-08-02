# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict

from lionagi._errors import ValidationError
from lionagi.libs.validate.fuzzy_match_keys import FuzzyMatchKeysParams
from lionagi.libs.validate.fuzzy_validate_pydantic import (
    fuzzy_validate_pydantic,
)
from lionagi.models import FieldModel, OperableModel

__all__ = (
    "Operative",
    "OperativeConfig",
)


class OperativeConfig(BaseModel):
    """
    Available validation_backend is only "pydantic" for now.
    """

    model_congig = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    fuzzy_json: bool = True
    fuzzy_keys: bool = True
    fuzzy_keys_params: FuzzyMatchKeysParams | dict = None

    llm_reparse: bool = True
    llm_reparse_with_context: bool = True
    llm_reparse_params: dict[str, Any] | None = None

    max_reparse_attempts: int = 2
    validation_backend: Literal["pydantic"] = "pydantic"

    response_model_name: str | None = None
    field_models: list[FieldModel] | None = None
    exclude_fields: list[str] | None = None
    inherit_base: bool = True
    doc: str | None = None
    frozen: bool = False


class Operative:
    """
    Class representing an operative that handles request and response models for operations.
    Should exist only within the context of an operation.
    """

    def __init__(
        self,
        config: OperativeConfig | None = None,
        request_type: type[BaseModel] | None = None,
        response_type: type[BaseModel] | None = None,
        field_models: list[FieldModel] | None = None,
    ):
        self.request_type = request_type
        self.response_type = response_type
        if isinstance(config, BaseModel):
            config = config.model_dump()

        self.config = config or OperativeConfig()
        self.config.field_models = self.config.field_models or []
        if field_models:
            self.config.field_models.extend(field_models)

        self.response_model: BaseModel = None
        self.response_string: str | None = None

        self._validation_methods: dict[str, Callable] = {}

        self._should_retry = None
        self._request_operable = OperableModel()
        self._response_operable = OperableModel()

        # Set default name if not provided
        if not self.name:
            self.name = (
                self.request_type.__name__
                if self.request_type
                else "Operative"
            )

    def _get_validation_meth(self, input_type: type) -> Callable:
        """Get the validation method based on the configured backend."""
        backend = self.config.validation_backend
        match backend:
            case "pydantic" if input_type is str:
                if "pydantic_string" not in self._validation_methods:
                    self._validation_methods["pydantic_string"] = partial(
                        fuzzy_validate_pydantic,
                        model_type=self.request_type,
                        fuzzy_json=self.config.fuzzy_json,
                        fuzzy_keys=self.config.fuzzy_keys,
                        fuzzy_keys_params=self.config.fuzzy_keys_params,
                    )
                    return self._validation_methods["pydantic_string"]
            case "pydantic" if input_type is dict:
                if "pydantic_dict" not in self._validation_methods:
                    self._validation_methods["pydantic_dict"] = (
                        self.response_model.model_validate
                    )
                    return self._validation_methods["pydantic_dict"]
            case _:
                raise ValueError(f"Unsupported validation backend: {backend}")

    def validate_response(self, input: str | dict[str, Any]) -> None:
        """Interface to validate the response based on the input type."""
        t_ = None
        if isinstance(input, str):
            t_ = str
            self.response_string = input
        elif isinstance(input, dict):
            t_ = dict
        meth = self._get_validation_meth(t_)
        try:
            self.response_model = meth(input)
            self._should_retry = False

        # only catch ValidationError, other exceptions should propagate
        except ValidationError as e:
            self.response_model = None
            if self.config.llm_reparse:
                self._should_retry = True
            else:
                raise e

    def update_response_model(
        self,
        text: str | None = None,
        data: dict | None = None,
    ) -> BaseModel | dict | str | None:

        if text is None and data is None:
            raise ValueError("Either text or data must be provided.")

        if text:
            self.validate_response(text)

        if data:
            if self.response_model is None:
                raise RuntimeError("Response model hasn't been not validated.")
            d_ = {**self.response_model.model_dump(), **data}
            self.validate_response(d_)

        return self.response_model

    def create_response_type(self) -> None:
        """Creates a new response type based on the provided parameters."""

        # Clear response operable and rebuild
        self._response_operable = OperableModel()

        # Add field models (skip if already exists from inheritance)
        if fms := self.config.field_models:
            for fm in fms:
                if fm.name not in self._response_operable.all_fields:
                    self._response_operable.add_field(fm.name, field_model=fm)

        self.response_type = self._response_operable.new_model(
            name=self.config.response_model_name or "OperativeResponseModel",
            base_type=self.request_type,
            exclude_fields=self.config.exclude_fields,
            inerit_base=self.config.inherit_base,
            frozen=self.config.frozen,
            doc=self.config.doc,
        )
