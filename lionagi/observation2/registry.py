# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Registry: Field specification registry with inverted indexing.

Following opinion 2: Use inverted indexes for efficient capability->field lookup
rather than iterating over thousands of fields.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Set

if TYPE_CHECKING:
    from .capability import Capability
    from .capability_expr import CapabilityRequirement
    from .field import FieldSpec


class FieldRegistry:
    """
    Registry for field specifications with inverted indexing.

    Opinion 2: "The CapabilityOperable should maintain an inverted index
    mapping capability patterns to fields they protect."
    """

    def __init__(self):
        """Initialize with empty registry and indexes."""
        # Primary storage
        self.fields: Dict[str, FieldSpec] = {}

        # Inverted indexes for O(1) lookup
        self.cap_to_fields: Dict[str, Set[str]] = defaultdict(set)
        self.type_to_fields: Dict[type, Set[str]] = defaultdict(set)
        self.tag_to_fields: Dict[str, Set[str]] = defaultdict(set)

        # Pattern cache for wildcard matching
        self._pattern_cache: Dict[str, re.Pattern] = {}

    def register(self, spec: FieldSpec, tags: List[str] = None) -> None:
        """
        Register a field specification with indexing.

        Args:
            spec: Field specification
            tags: Optional tags for categorization
        """
        self.fields[spec.name] = spec

        # Index by capability requirement
        self.cap_to_fields[spec.required_capability].add(spec.name)

        # Index by type
        self.type_to_fields[spec.type].add(spec.name)

        # Index by tags
        if tags:
            for tag in tags:
                self.tag_to_fields[tag].add(spec.name)

        # Handle wildcard patterns
        if "*" in spec.required_capability:
            pattern = self._compile_pattern(spec.required_capability)
            self._pattern_cache[spec.required_capability] = pattern

    def find_accessible_fields(
        self, capabilities: Set[Capability]
    ) -> Set[str]:
        """
        Find all fields accessible with given capabilities.

        Opinion 2: "Iterate over the actor's capabilities (usually smaller)
        and use the index to look up accessible fields."

        Args:
            capabilities: Set of capabilities

        Returns:
            Set of field names that can be accessed
        """
        accessible = set()

        # Iterate over capabilities (smaller set)
        for cap in capabilities:
            resource_pattern = cap.resource

            # Direct lookup
            if resource_pattern in self.cap_to_fields:
                accessible.update(self.cap_to_fields[resource_pattern])

            # Pattern matching for wildcards
            if "*" in resource_pattern:
                pattern = self._compile_pattern(resource_pattern)
                for field_cap, field_names in self.cap_to_fields.items():
                    if pattern.match(field_cap):
                        accessible.update(field_names)
            else:
                # Check if this capability matches any registered patterns
                for pattern_str, pattern in self._pattern_cache.items():
                    if pattern.match(resource_pattern):
                        accessible.update(self.cap_to_fields[pattern_str])

        return accessible

    def find_by_type(self, field_type: type) -> Set[str]:
        """Find all fields of a given type."""
        return self.type_to_fields.get(field_type, set())

    def find_by_tag(self, tag: str) -> Set[str]:
        """Find all fields with a given tag."""
        return self.tag_to_fields.get(tag, set())

    def compose_form_fields(
        self,
        capabilities: Set[Capability],
        required_types: List[type] = None,
        tags: List[str] = None,
    ) -> Dict[str, FieldSpec]:
        """
        Compose a set of fields for a form based on capabilities.

        Args:
            capabilities: Available capabilities
            required_types: Optional type filter
            tags: Optional tag filter

        Returns:
            Dictionary of field specs the actor can access
        """
        # Start with capability-accessible fields
        accessible = self.find_accessible_fields(capabilities)

        # Apply type filter
        if required_types:
            type_fields = set()
            for req_type in required_types:
                type_fields.update(self.find_by_type(req_type))
            accessible &= type_fields

        # Apply tag filter
        if tags:
            tag_fields = set()
            for tag in tags:
                tag_fields.update(self.find_by_tag(tag))
            accessible &= tag_fields

        # Return field specs
        return {
            name: self.fields[name]
            for name in accessible
            if name in self.fields
        }

    def _compile_pattern(self, pattern_str: str) -> re.Pattern:
        """Compile capability pattern to regex."""
        # Convert capability wildcard to regex
        # field:* -> field:.*
        # *:read -> .*:read
        regex_str = pattern_str.replace("*", ".*")
        regex_str = f"^{regex_str}$"
        return re.compile(regex_str)

    def get_field(self, name: str) -> FieldSpec | None:
        """Get field specification by name."""
        return self.fields.get(name)

    def __len__(self) -> int:
        """Number of registered fields."""
        return len(self.fields)

    def clear_indexes(self) -> None:
        """Clear all indexes (useful for testing)."""
        self.cap_to_fields.clear()
        self.type_to_fields.clear()
        self.tag_to_fields.clear()
        self._pattern_cache.clear()

    def rebuild_indexes(self) -> None:
        """Rebuild all indexes from current fields."""
        self.clear_indexes()
        for spec in self.fields.values():
            # Re-register without modifying primary storage
            self.cap_to_fields[spec.required_capability].add(spec.name)
            self.type_to_fields[spec.type].add(spec.name)


# Global registry instance
_global_registry = FieldRegistry()


def get_registry() -> FieldRegistry:
    """Get the global field registry."""
    return _global_registry


def register_field(spec: FieldSpec, tags: List[str] = None) -> None:
    """Register a field in the global registry."""
    _global_registry.register(spec, tags)
