import json

from pydantic import BaseModel

from lionagi._errors import ValidationError
from lionagi.utils import to_json

from .fuzzy_match_keys import FuzzyMatchKeysParams, fuzzy_match_keys

__all__ = ("fuzzy_validate_pydantic",)


def fuzzy_validate_pydantic(
    text,
    /,
    model_type: type[BaseModel],
    fuzzy_json: bool = True,
    fuzzy_keys: bool = False,
    fuzzy_keys_params: FuzzyMatchKeysParams | dict = None,
):
    """
    extract all json block like object from the text and validate the first one
    against the target pydantic model.
    - if fuzzy_json is True, it will try to parse the text as json with au json
        repair for parsing.
    - if fuzzy_keys is True, it will use fuzzy matching to correct the keys.
        - if fuzzy_keys_params is not provided, by defualt, will remove all un-fuzzy-matchable
            keys via default algorithm (jaro_winkler, 0.85 threshold). Which might result in error
            if the sepecific fields do not have default values.
        - if fuzzy_keys_params is provided, will use those parameters to correct the keys.

    See also:
        - "lionagi.libs.validate.fuzzy_match_keys": for parameters and usage of fuzzy_match_keys.
    """
    try:
        model_data = (
            (a := to_json(text, fuzzy_parse=fuzzy_json)) if a else None
        )
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Failed to parse JSON from model response: {e}"
        ) from e
    except ValueError as e:
        raise ValidationError(
            f"Failed to fuzzy parse JSON from model response: {e}"
        ) from e

    d = model_data
    if fuzzy_keys:
        if fuzzy_keys_params is None:
            model_data = fuzzy_match_keys(
                d, model_type.model_fields, handle_unmatched="remove"
            )
        elif isinstance(fuzzy_keys_params, dict):
            model_data = fuzzy_match_keys(
                d, model_type.model_fields, **fuzzy_keys_params
            )
        elif isinstance(fuzzy_keys_params, FuzzyMatchKeysParams):
            model_data = fuzzy_keys_params(d, model_type.model_fields)
        else:
            raise TypeError(
                "fuzzy_keys_params must be a dict or FuzzyMatchKeysParams instance"
            )

    try:
        return model_type.model_validate(model_data)
    except Exception as e:
        raise ValidationError(f"Validation failed: {e}") from e
