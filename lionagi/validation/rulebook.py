"""
Simple rule collection for validation.

This is kept minimal - most users should just pass rules directly to Validator.
"""

from typing import Dict, List, Optional

from ..rule import Rule


class RuleBook:
    """
    Simple container for rules if you need to share them across validators.

    For most use cases, just pass rules directly to Validator.
    """

    def __init__(self, rules: Optional[Dict[str, Rule]] = None):
        """
        Initialize with optional rule dictionary.

        Args:
            rules: Dictionary mapping names to Rule instances
        """
        self.rules = rules or {}

    def add(self, name: str, rule: Rule):
        """Add a rule to the book."""
        self.rules[name] = rule

    def get(self, name: str) -> Optional[Rule]:
        """Get a rule by name."""
        return self.rules.get(name)

    def get_rules(self) -> List[Rule]:
        """Get all rules as a list."""
        return list(self.rules.values())

    def __getitem__(self, key: str) -> Rule:
        """Get rule by name using dict syntax."""
        return self.rules[key]

    def __len__(self) -> int:
        """Number of rules in the book."""
        return len(self.rules)
