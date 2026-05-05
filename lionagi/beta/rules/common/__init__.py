# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Common validation rules for basic types.

Provides built-in rules for:
- StringRule: String validation with patterns, length constraints
- NumberRule: Numeric validation with range constraints
- BooleanRule: Boolean validation with auto-conversion
- ChoiceRule: Enumerated choice validation
- MappingRule: Dict/mapping validation
- BaseModelRule: Pydantic model validation
- RuleRegistry: Type-to-rule mapping with inheritance
"""

from ..registry import RuleRegistry, get_default_registry, reset_default_registry
from .boolean import BooleanRule
from .choice import ChoiceRule
from .mapping import MappingRule
from .model import BaseModelRule
from .number import NumberRule
from .string import StringRule

__all__ = (
    "BaseModelRule",
    "BooleanRule",
    "ChoiceRule",
    "MappingRule",
    "NumberRule",
    "RuleRegistry",
    "StringRule",
    "get_default_registry",
    "reset_default_registry",
)
