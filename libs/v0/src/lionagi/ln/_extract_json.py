import re
from typing import Any

import orjson

from ._fuzzy_json import fuzzy_json

# Precompile the regex for extracting JSON code blocks
_JSON_BLOCK_PATTERN = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def extract_json(
    input_data: str | list[str],
    /,
    *,
    fuzzy_parse: bool = False,
    return_one_if_single: bool = True,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Extract and parse JSON content from a string or markdown code blocks.
    Attempts direct JSON parsing first. If that fails, looks for JSON content
    within markdown code blocks denoted by ```json.

    Args:
        input_data (str | list[str]): The input string or list of strings to parse.
        fuzzy_parse (bool): If True, attempts fuzzy JSON parsing on failed attempts.
        return_one_if_single (bool): If True and only one JSON object is found,
            returns a dict instead of a list with one dict.
    """

    # If input_data is a list, join into a single string
    if isinstance(input_data, list):
        input_str = "\n".join(input_data)
    else:
        input_str = input_data

    # 1. Try direct parsing
    try:
        if fuzzy_parse:
            return fuzzy_json(input_str)
        return orjson.loads(input_str)
    except Exception:
        pass

    # 2. Attempt extracting JSON blocks from markdown
    matches = _JSON_BLOCK_PATTERN.findall(input_str)
    if not matches:
        return []

    # If only one match, return single dict; if multiple, return list of dicts
    if return_one_if_single and len(matches) == 1:
        data_str = matches[0]
        if fuzzy_parse:
            return fuzzy_json(data_str)
        return orjson.loads(data_str)

    # Multiple matches
    if fuzzy_parse:
        return [fuzzy_json(m) for m in matches]
    else:
        return [orjson.loads(m) for m in matches]
