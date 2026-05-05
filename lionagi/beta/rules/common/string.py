# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import re
from re import Pattern
from typing import Any

from ..rule import Rule, RuleParams, RuleQualifier

__all__ = ("StringRule",)

# ReDoS protection: max input length for regex matching
DEFAULT_REGEX_MAX_INPUT_LENGTH = 10_000

# Heuristic ReDoS detection (nested quantifiers). Not exhaustive - for untrusted
# patterns use google-re2 or regex timeout. Input length limit provides additional mitigation.
_REDOS_PATTERNS = [
    r"\(\.\*\)\*",
    r"\(\.\+\)\*",
    r"\(\.\*\)\+",
    r"\(\.\+\)\+",
    r"\(\[.*?\]\+\)\+",
    r"\(\[.*?\]\*\)\*",
]
_REDOS_DETECTOR = re.compile("|".join(_REDOS_PATTERNS))


def _get_string_params() -> RuleParams:
    """Default params: applies to str via ANNOTATION qualifier, auto_fix enabled."""
    return RuleParams(
        apply_types={str},
        apply_fields=set(),
        default_qualifier=RuleQualifier.ANNOTATION,
        auto_fix=True,
        kw={},
    )


class StringRule(Rule):
    """Rule for validating and converting string values.

    Features:
    - Type checking (must be str)
    - Length constraints (min_length, max_length)
    - Pattern matching (regex) with ReDoS protection
    - Auto-conversion from any type to string

    Usage:
        rule = StringRule(min_length=1, max_length=100, pattern=r'^[A-Za-z]+$')
        result = await rule.invoke("name", "Ocean", str)
    """

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
        regex_max_input_length: int = DEFAULT_REGEX_MAX_INPUT_LENGTH,
        params: RuleParams | None = None,
        **kw,
    ):
        """Initialize string rule.

        Args:
            min_length: Minimum string length (inclusive)
            max_length: Maximum string length (inclusive)
            pattern: Regex pattern to match. Patterns with nested quantifiers
                are rejected to prevent ReDoS attacks.
            regex_max_input_length: Maximum input length for regex matching
                (default 10,000 chars). Inputs exceeding this are rejected.
            params: Custom RuleParams (uses default if None)
            **kw: Additional validation kwargs

        Raises:
            ValueError: If pattern contains potential ReDoS vulnerabilities
        """
        if params is None:
            params = _get_string_params()
        super().__init__(params, **kw)
        self.min_length = min_length
        self.max_length = max_length
        self.regex_max_input_length = regex_max_input_length

        self._compiled_pattern: Pattern[str] | None
        if pattern is not None:
            if _REDOS_DETECTOR.search(pattern):
                raise ValueError(
                    f"Pattern '{pattern}' contains nested quantifiers that could cause "
                    "ReDoS (Regular Expression Denial of Service). Use simpler patterns."
                )
            try:
                self._compiled_pattern = re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
        else:
            self._compiled_pattern = None
        self.pattern = pattern

    async def validate(self, v: Any, t: type, **kw) -> None:
        """Validate that value is a string with correct length and pattern.

        Raises:
            ValueError: If not a string or constraints violated
        """
        if not isinstance(v, str):
            raise ValueError(f"Invalid string value: expected str, got {type(v).__name__}")

        if self.min_length is not None and len(v) < self.min_length:
            raise ValueError(
                f"String too short: got {len(v)} characters, minimum {self.min_length}"
            )
        if self.max_length is not None and len(v) > self.max_length:
            raise ValueError(f"String too long: got {len(v)} characters, maximum {self.max_length}")

        if self._compiled_pattern is not None:
            if len(v) > self.regex_max_input_length:
                raise ValueError(
                    f"String too long for regex matching: got {len(v)} characters, "
                    f"maximum {self.regex_max_input_length}"
                )
            if not self._compiled_pattern.match(v):
                raise ValueError(f"String does not match required pattern: {self.pattern}")

    async def perform_fix(self, v: Any, t: type) -> Any:
        """Attempt to convert value to string and re-validate.

        Returns:
            String representation of value (validated)

        Raises:
            ValueError: If conversion or re-validation fails
        """
        try:
            fixed = str(v)
        except Exception as e:
            raise ValueError(f"Failed to convert {v} to string") from e

        # Re-validate the fixed value
        await self.validate(fixed, t)
        return fixed
