# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Claude Code adapter for provider registry integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core import Service
from ..providers.claude_code import ClaudeCodeRequestModel, create_claude_code_service
from ..providers.provider_registry import ProviderAdapter


class ClaudeCodeAdapter(ProviderAdapter):
    """Adapter for Claude Code CLI provider.

    Supports:
      - provider="claude_code"
      - model="claude_code/<model>"
      - base_url="claude_code://path/to/repo"
    """

    name = "claude_code"
    default_base_url = "claude_code://."
    request_model = ClaudeCodeRequestModel
    requires = frozenset({"exec:claude", "fs.read", "fs.write"})

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        """Check if this adapter supports the given configuration."""
        if (provider or "").lower() == "claude_code":
            return True
        if (model or "").lower().startswith("claude_code/"):
            return True
        return (base_url or "").lower().startswith("claude_code://")

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service:
        """Create Claude Code CLI service instance."""
        # Extract base repo from base_url if provided
        base_repo = kwargs.pop("base_repo", None)

        if base_url and base_url.startswith("claude_code://"):
            # Handle claude_code://path format
            repo_path = base_url[len("claude_code://") :]
            if repo_path and repo_path != ".":
                base_repo = repo_path

        return create_claude_code_service(base_repo=base_repo)

    def required_rights(self, *, base_url: str | None, **kwargs: Any) -> set[str]:
        """Calculate required capabilities for this service."""
        rights = {"exec:claude"}

        # Extract base repo from base_url or kwargs
        base_repo = kwargs.get("base_repo")
        if base_url and base_url.startswith("claude_code://"):
            # Handle claude_code://path format
            repo_path = base_url[len("claude_code://") :]
            if repo_path and repo_path != ".":
                base_repo = repo_path

        if base_repo:
            repo_path = Path(base_repo).resolve()
            rights.add(f"fs.read:{repo_path}")
            rights.add(f"fs.write:{repo_path}")
        else:
            rights.add("fs.read")
            rights.add("fs.write")

        return rights
