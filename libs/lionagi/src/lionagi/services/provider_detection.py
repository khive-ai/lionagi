# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

"""Provider prefix helper (explicit; no regex on hot paths).

Use either:
  - provider="openai", model="gpt-4o"
  - or model="openai/gpt-4o"
This module is intentionally tiny. Resolution is handled by the ProviderRegistry.
"""


def parse_provider_prefix(model: str | None) -> tuple[str | None, str | None]:
    """Parse provider prefix from model string.

    Args:
        model: Model name, optionally prefixed with provider (e.g., "openai/gpt-4")

    Returns:
        Tuple of (provider, model_without_prefix)
    """
    if not model or "/" not in model:
        return None, model
    p, _, rest = model.partition("/")
    return (p or None), (rest or None)
