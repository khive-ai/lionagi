# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from functools import partial

from pydantic import BaseModel

from lionagi.ln import extract_json, fuzzy_validate_pydantic, lcall
from lionagi.ln.fuzzy._fuzzy_match import FuzzyMatchKeysParams
from lionagi.ln.types import DataClass, Unset
from lionagi.models import FieldModel, OperableModel


@dataclass(slots=True)
class PydOperative(DataClass):

    request_type: type[BaseModel]
    response_type: type[BaseModel]
    fuzzy_match: FuzzyMatchKeysParams

    name: str
    res_models: list[BaseModel]
    res_model: BaseModel
    res_str: str
    should_retry: bool
    error: Exception
    listable: bool = False

    auto_retry_parse: bool = True
    max_retries: int = 3
    parse_kwargs: dict = field(default_factory=dict)

    _request_operable: OperableModel = field(default_factory=OperableModel)
    _response_operable: OperableModel = field(default_factory=OperableModel)

    def validate_response(self, text: str, fuzzy_match: bool = True) -> None:

        d_ = extract_json(text, fuzzy_parse=True)
        self.res_str = text

        if not d_:
            self.should_retry = True
            return

        try:
            f = partial(
                fuzzy_validate_pydantic,
                model_type=self.response_type,
                fuzzy_parse=False,
                fuzzy_match=fuzzy_match,
                fuzzy_match_params=(
                    None if not fuzzy_match else self.fuzzy_match
                ),
            )
            if self.listable is True:
                self.res_models = lcall(d_, f)
            else:
                self.res_model = f(d_[0])

            self.should_retry = False

        except Exception as e:
            self.should_retry = True
            self.error = e

    def update_response(
        self,
        text: str | None = None,
        data: dict | None = None,
        fuzzy_match=True,
    ):
        if text is None and data is None:
            raise ValueError("Either text or data must be provided.")

        if text:
            self.validate_response(text, fuzzy_match=fuzzy_match)
            if self.error is not Unset:
                return False

        if data and self.response_type:
            d_ = self.response_model.model_dump()
            d_.update(data)
            self.response_model = self.response_type.model_validate(d_)
            if self.error is not Unset:
                return False
        return True

    def create_response_type(
        self,
        field_models: list[FieldModel] | None = None,
        exclude_fields: list[str] | None = None,
        config_dict: dict | None = None,
        doc: str | None = None,
        frozen: bool = False,
    ) -> None:
        """Creates a new response type based on the provided parameters.

        Args:
            field_models (list[FieldModel], optional): List of field models.
            exclude_fields (list, optional): List of fields to exclude.
            inherit_base (bool, optional): Whether to inherit the base model.
            config_dict (dict | None, optional): Configuration dictionary.
            doc (str | None, optional): Documentation string.
            frozen (bool, optional): Whether the model is frozen.
        """

        # Clear response operable and rebuild
        self._response_operable = OperableModel()

        # Copy fields from request operable if inherit_base
        for (
            field_name,
            field_model,
        ) in self._request_operable.extra_field_models.items():
            self._response_operable.add_field(
                field_name, field_model=field_model
            )

        # Add field models (skip if already exists from inheritance)
        if field_models:
            # Coerce to list if single FieldModel instance
            field_models_list = (
                [field_models]
                if isinstance(field_models, FieldModel)
                else field_models
            )
            for field_model in field_models_list:
                if field_model.name not in self._response_operable.all_fields:
                    self._response_operable.add_field(
                        field_model.name,
                        field_model=field_model,
                    )

        # Generate response type
        exclude_fields = exclude_fields or []
        use_fields = set(self._response_operable.all_fields.keys()) - set(
            exclude_fields
        )

        # Determine base type - use request_type if inheriting and no specific base provided
        self.response_type = self._response_operable.new_model(
            name=self.name or "ResponseModel",
            use_fields=use_fields,
            base_type=self.request_type or None,
            frozen=frozen,
            config_dict=config_dict,
            doc=doc,
        )


Operative = PydOperative  # Alias for backward compatibility

__all__ = ("Operative", "PydOperative")