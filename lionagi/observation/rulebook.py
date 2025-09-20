# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
RuleBook: A registry of validation rules with configuration.

Simple, clean registry pattern for validation rules.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .rule.base import Rule


class RuleBook:
    """Registry of validation rules with ordered application."""

    def __init__(
        self,
        rules: Optional[Dict[str, type[Rule]]] = None,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Initialize RuleBook with rules and configuration.

        Args:
            rules: Mapping of rule names to Rule classes
            config: Configuration for each rule
        """
        self._rules = OrderedDict(rules or {})
        self._config = config or {}
        self._instances: Dict[str, Rule] = {}

    @property
    def rules(self) -> Dict[str, type[Rule]]:
        """Get rule classes."""
        return dict(self._rules)

    @property
    def rule_names(self) -> List[str]:
        """Get ordered list of rule names."""
        return list(self._rules.keys())

    def get_rule(self, name: str) -> Optional[Rule]:
        """Get or create a rule instance.

        Args:
            name: Rule name

        Returns:
            Rule instance or None if not found
        """
        if name not in self._rules:
            return None

        if name not in self._instances:
            rule_class = self._rules[name]
            rule_config = self._config.get(name, {})
            self._instances[name] = rule_class(config=rule_config)

        return self._instances[name]

    def add_rule(
        self,
        name: str,
        rule_class: type[Rule],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a rule to the book.

        Args:
            name: Rule name
            rule_class: Rule class to add
            config: Optional configuration
        """
        self._rules[name] = rule_class
        if config:
            self._config[name] = config
        # Clear cached instance if exists
        self._instances.pop(name, None)

    def remove_rule(self, name: str) -> None:
        """Remove a rule from the book.

        Args:
            name: Rule name to remove
        """
        self._rules.pop(name, None)
        self._config.pop(name, None)
        self._instances.pop(name, None)

    def update_config(self, name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a rule.

        Args:
            name: Rule name
            config: New configuration
        """
        if name in self._rules:
            self._config[name] = config
            # Clear cached instance to force recreation with new config
            self._instances.pop(name, None)

    def find_applicable_rule(
        self,
        field: str,
        value: Any,
        annotation: Optional[str] = None,
        **kwargs,
    ) -> Optional[Rule]:
        """Find the first applicable rule for a field.

        Args:
            field: Field name
            value: Field value
            annotation: Type annotation hint
            **kwargs: Additional context

        Returns:
            First applicable rule or None
        """
        for name in self._rules:
            rule = self.get_rule(name)
            if rule and rule.applies(
                field, value, annotation=annotation, **kwargs
            ):
                return rule
        return None

    def __len__(self) -> int:
        """Number of rules in the book."""
        return len(self._rules)

    def __contains__(self, name: str) -> bool:
        """Check if a rule exists."""
        return name in self._rules

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RuleBook({len(self._rules)} rules: {list(self._rules.keys())})"
        )
