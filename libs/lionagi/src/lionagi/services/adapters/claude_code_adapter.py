# Copyright (c) 2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Claude Code CLI adapter for v1 provider registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..core import Service
from ..providers.claude_code import ClaudeCodeRequestModel, create_claude_code_service
from ..providers.provider_registry import ProviderAdapter


class ClaudeCodeAdapter(ProviderAdapter):
    """Adapter for Claude Code CLI integration.

    Supports:
      - provider="claude_code"
      - model="claude_code/<model>"
      - Any base_url="claude_code://<path>" for custom repos
    """

    name = "claude_code"
    default_base_url = "claude_code://."  # Current directory
    request_model = ClaudeCodeRequestModel
    requires = {"exec:claude", "fs.read", "fs.write"}

    # Configuration validator
    class ConfigModel(BaseModel):
        base_repo: str | None = None
        permission_mode: str | None = None
        allowed_tools: list[str] | None = None
        disallowed_tools: list[str] | None = None
        mcp_config: str | None = None
        auto_finish: bool = False
        verbose_output: bool = False

    def supports(self, *, provider: str | None, model: str | None, base_url: str | None) -> bool:
        """Check if this adapter supports the given configuration."""
        if (provider or "").lower() == "claude_code":
            return True
        if (model or "").lower().startswith("claude_code/"):
            return True
        if (base_url or "").startswith("claude_code://"):
            return True
        return False

    def create_service(self, *, base_url: str | None, **kwargs: Any) -> Service:
        """Create Claude Code CLI service instance."""
        # Extract repo path from base_url if provided
        base_repo = kwargs.pop("base_repo", None)

        if not base_repo and base_url:
            if base_url.startswith("claude_code://"):
                base_repo = base_url[14:]  # Remove "claude_code://" prefix
            elif base_url != "claude_code://.":
                base_repo = base_url

        # Default to current directory if no repo specified
        if not base_repo:
            base_repo = "."

        return create_claude_code_service(base_repo=Path(base_repo))

    def required_rights(self, *, base_url: str | None, **kwargs: Any) -> set[str]:
        """Return required capabilities for Claude Code service."""
        rights = {"exec:claude", "fs.read", "fs.write"}

        # Add workspace-specific rights if needed
        base_repo = kwargs.get("base_repo")
        if not base_repo and base_url and base_url.startswith("claude_code://"):
            base_repo = base_url[14:]

        if base_repo and base_repo != ".":
            # Add specific directory access rights
            repo_path = Path(base_repo).resolve()
            rights.add(f"fs.read:{repo_path}")
            rights.add(f"fs.write:{repo_path}")

        return rights
