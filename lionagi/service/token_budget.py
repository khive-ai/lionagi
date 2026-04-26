# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Token budget tracking — know how much context you've used and how much remains.

Model context windows loaded from model_registry.yaml (per-provider).
Resolution: endpoint config.context_window > YAML registry > default.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


_REGISTRY_PATH = Path(__file__).parent / "model_registry.yaml"
_registry: dict | None = None


def _load_registry() -> dict:
    global _registry
    if _registry is None:
        with open(_REGISTRY_PATH) as f:
            _registry = yaml.safe_load(f) or {}
    return _registry


def lookup_context_window(model_name: str, provider: str | None = None) -> int | None:
    """Look up context window from model_registry.yaml.

    Tries provider-specific match first, then scans all providers.
    Uses longest prefix match within each provider.
    """
    reg = _load_registry()
    model_lower = model_name.lower()

    def _search_provider(prov_dict: dict) -> int | None:
        best_match = None
        best_len = 0
        for prefix, window in prov_dict.items():
            if prefix in model_lower and len(prefix) > best_len:
                best_match = window
                best_len = len(prefix)
        return best_match

    if provider:
        prov_dict = reg.get(provider.lower(), {})
        if isinstance(prov_dict, dict):
            result = _search_provider(prov_dict)
            if result is not None:
                return result

    for key, prov_dict in reg.items():
        if key == "default" or not isinstance(prov_dict, dict):
            continue
        result = _search_provider(prov_dict)
        if result is not None:
            return result

    return reg.get("default", 128_000)


@dataclass(frozen=True)
class TokenBudget:
    used: int
    limit: int
    model: str | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def usage_pct(self) -> float:
        return self.used / self.limit if self.limit > 0 else 0.0

    @property
    def is_warning(self) -> bool:
        return self.usage_pct >= 0.7

    @property
    def is_critical(self) -> bool:
        return self.usage_pct >= 0.9


def get_context_window(branch: Branch) -> int:
    """Get context window size for branch's chat model.

    Resolution: endpoint config.context_window > YAML registry > default.
    """
    try:
        endpoint = branch.chat_model.endpoint
        if getattr(endpoint.config, "context_window", None):
            return endpoint.config.context_window

        model_name = ""
        provider = getattr(endpoint.config, "provider", None)
        if hasattr(endpoint.config, "kwargs"):
            model_name = endpoint.config.kwargs.get("model", "")
        if not model_name and hasattr(endpoint.config, "params"):
            model_name = endpoint.config.params.get("model", "")

        if model_name:
            result = lookup_context_window(model_name, provider)
            if result is not None:
                return result
    except (AttributeError, KeyError):
        pass

    return _load_registry().get("default", 128_000)


def get_token_budget(branch: Branch) -> TokenBudget:
    """Calculate current token budget for a branch."""
    from lionagi.service.token_calculator import TokenCalculator

    limit = get_context_window(branch)
    progression = branch.progression
    pile = branch.msgs.messages

    used = 0
    for uid in progression:
        if uid in pile:
            msg = pile[uid]
            c = msg.content if hasattr(msg, "content") else ""
            if c:
                used += TokenCalculator.tokenize(
                    str(c) if not isinstance(c, str) else c
                )

    model_name = None
    try:
        if hasattr(branch.chat_model.endpoint.config, "kwargs"):
            model_name = branch.chat_model.endpoint.config.kwargs.get("model")
    except AttributeError:
        pass

    return TokenBudget(used=used, limit=limit, model=model_name)
