"""
Configuration utilities for the Pynector library.
"""

import os
from typing import Any, Optional


def get_env_config(key: str, default: Any = None) -> Any:
    """Get a configuration value from environment variables.

    Args:
        key: The configuration key (will be prefixed with PYNECTOR_)
        default: The default value if not found

    Returns:
        The configuration value
    """
    env_key = f"PYNECTOR_{key.upper()}"
    if env_key in os.environ:
        return os.environ[env_key]
    return default


def merge_configs(
    base: dict[str, Any], override: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Merge configuration dictionaries.

    Args:
        base: The base configuration
        override: The override configuration

    Returns:
        The merged configuration
    """
    if override is None:
        return base.copy()

    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result
