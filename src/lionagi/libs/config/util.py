# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

__all__ = ("merge_configs",)

def merge_configs(
    base: Dict[str, Any], override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge configuration dictionaries.

    Args:
        base: The base configuration dictionary.
        override: The override configuration dictionary. Values from this dict
                  will overwrite values in the base dictionary. If a key is
                  present in both and both values are dictionaries, they will
                  be merged recursively.

    Returns:
        A new dictionary representing the merged configuration.
    """
    if override is None:
        return base.copy()

    result = base.copy() # Start with a copy of the base
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # If both base and override have a dict for the same key, recurse
            result[key] = merge_configs(result[key], value)
        else:
            # Otherwise, the override value takes precedence
            result[key] = value
    return result