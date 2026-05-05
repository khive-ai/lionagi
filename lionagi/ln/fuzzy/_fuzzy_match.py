from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from ..types import KeysLike, ModelConfig, Params, Unset
from ._string_similarity import (
    SIMILARITY_ALGO_MAP,
    SIMILARITY_TYPE,
    SimilarityAlgo,
    SimilarityFunc,
    string_similarity,
)

__all__ = (
    "HandleUnmatched",
    "fuzzy_match_keys",
    "FuzzyMatchKeysParams",
)


class HandleUnmatched(str, Enum):
    IGNORE = "ignore"
    RAISE = "raise"
    REMOVE = "remove"
    FILL = "fill"
    FORCE = "force"


def fuzzy_match_keys(
    d_: dict[str, Any],
    keys: KeysLike,
    /,
    *,
    similarity_algo: SIMILARITY_TYPE | SimilarityAlgo | SimilarityFunc = "jaro_winkler",
    similarity_threshold: float = 0.85,
    fuzzy_match: bool = True,
    handle_unmatched: HandleUnmatched | str = HandleUnmatched.IGNORE,
    fill_value: Any = None,
    fill_mapping: dict[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Remap dictionary keys to expected keys using fuzzy string similarity."""
    if not isinstance(d_, dict):
        raise TypeError("First argument must be a dictionary")
    if keys is None:
        raise TypeError("Keys argument cannot be None")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0.0 and 1.0")

    if isinstance(keys, (list, tuple)):
        fields_set = set(keys)
    elif hasattr(keys, "keys"):
        fields_set = set(keys.keys())
    else:
        fields_set = set(keys)
    if not fields_set:
        return d_.copy()

    corrected_out = {}
    matched_expected = set()
    matched_input = set()

    if isinstance(similarity_algo, SimilarityAlgo):
        similarity_func = SIMILARITY_ALGO_MAP[similarity_algo.value]
    elif isinstance(similarity_algo, str):
        if similarity_algo not in SIMILARITY_ALGO_MAP:
            raise ValueError(f"Unknown similarity algorithm: {similarity_algo}")
        similarity_func = SIMILARITY_ALGO_MAP[similarity_algo]
    else:
        similarity_func = similarity_algo

    handle_unmatched = HandleUnmatched(handle_unmatched)

    for key in d_:
        if key in fields_set:
            corrected_out[key] = d_[key]
            matched_expected.add(key)
            matched_input.add(key)

    if fuzzy_match:
        remaining_input = set(d_.keys()) - matched_input
        remaining_expected = fields_set - matched_expected

        for key in remaining_input:
            if not remaining_expected:
                break

            matches = string_similarity(
                key,
                list(remaining_expected),
                algorithm=similarity_func,
                threshold=similarity_threshold,
                return_most_similar=True,
            )

            if matches:
                match = matches
                corrected_out[match] = d_[key]
                matched_expected.add(match)
                matched_input.add(key)
                remaining_expected.remove(match)
            elif handle_unmatched == HandleUnmatched.IGNORE:
                corrected_out[key] = d_[key]

    unmatched_input = set(d_.keys()) - matched_input
    unmatched_expected = fields_set - matched_expected

    if handle_unmatched == HandleUnmatched.RAISE and unmatched_input:
        raise ValueError(f"Unmatched keys found: {unmatched_input}")

    elif handle_unmatched == HandleUnmatched.IGNORE:
        for key in unmatched_input:
            corrected_out[key] = d_[key]

    elif handle_unmatched in (HandleUnmatched.FILL, HandleUnmatched.FORCE):
        for key in unmatched_expected:
            if fill_mapping and key in fill_mapping:
                corrected_out[key] = fill_mapping[key]
            else:
                corrected_out[key] = fill_value

        if handle_unmatched == HandleUnmatched.FILL:
            for key in unmatched_input:
                corrected_out[key] = d_[key]

    if strict and unmatched_expected:
        raise ValueError(f"Missing required keys: {unmatched_expected}")

    return corrected_out


@dataclass(slots=True, init=False, frozen=True)
class FuzzyMatchKeysParams(Params):
    _config: ClassVar[ModelConfig] = ModelConfig(none_as_sentinel=False)
    _func: ClassVar[Any] = fuzzy_match_keys

    similarity_algo: SIMILARITY_TYPE | SimilarityAlgo | SimilarityFunc = "jaro_winkler"
    similarity_threshold: float = 0.85

    fuzzy_match: bool = True
    handle_unmatched: HandleUnmatched | str = HandleUnmatched.IGNORE

    fill_value: Any = Unset
    fill_mapping: dict[str, Any] | Any = Unset
    strict: bool = False

    def __call__(self, d_: dict[str, Any], keys: KeysLike) -> dict[str, Any]:
        return fuzzy_match_keys(d_, keys, **self.default_kw())
