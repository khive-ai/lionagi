from pydantic import BaseModel

from lionagi._errors import ValidationError

from ._extract_json import extract_json
from ._fuzzy_match import FuzzyMatchKeysParams, fuzzy_match_keys

__all__ = ("fuzzy_validate_pydantic",)


def fuzzy_validate_pydantic(
    text,
    /,
    model_type: type[BaseModel],
    fuzzy_parse: bool = True,
    fuzzy_match: bool = False,
    fuzzy_match_params: FuzzyMatchKeysParams | dict = None,
):
    try:
        model_data = extract_json(text, fuzzy_parse=fuzzy_parse)
    except Exception as e:
        raise ValidationError(
            f"Failed to extract valid JSON from model response: {e}"
        ) from e

    d = model_data
    if fuzzy_match:
        if fuzzy_match_params is None:
            model_data = fuzzy_match_keys(
                d, model_type.model_fields, handle_unmatched="remove"
            )
        elif isinstance(fuzzy_match_params, dict):
            model_data = fuzzy_match_keys(
                d, model_type.model_fields, **fuzzy_match_params
            )
        elif isinstance(fuzzy_match_params, FuzzyMatchKeysParams):
            model_data = fuzzy_match_params(d, model_type.model_fields)
        else:
            raise TypeError(
                "fuzzy_keys_params must be a dict or FuzzyMatchKeysParams instance"
            )

    try:
        return model_type.model_validate(model_data)
    except Exception as e:
        raise ValidationError(f"Validation failed: {e}") from e
