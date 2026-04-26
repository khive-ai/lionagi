# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Permission system for agent tool access control.

Three modes:
    - allow_all: everything permitted (default for orchestrators)
    - deny_all: nothing permitted (safe mode)
    - rules: per-tool allow/deny/escalate rules

Rules are checked in order: deny first, then allow, then default.
Escalation lets a worker agent request permission from its orchestrator.

Usage::

    policy = PermissionPolicy(
        mode="rules",
        allow={"reader": ["*"], "search": ["*"], "bash": ["git *", "cargo *"]},
        deny={"bash": ["rm *", "sudo *"], "editor": [".env", "*.key"]},
        escalate={"bash": ["*"]},  # anything not explicitly allowed → escalate
    )

    # As AgentConfig
    config = AgentConfig.coding()
    config.permissions = {
        "mode": "rules",
        "allow": {"reader": ["*"], "search": ["*"]},
        "deny": {"bash": ["rm *"]},
        "escalate": {"bash": ["*"]},
    }

    # In settings.yaml
    permissions:
      mode: rules
      allow:
        reader: ["*"]
        search: ["*"]
        bash: ["git *", "cargo *", "uv *"]
      deny:
        bash: ["rm -rf *", "sudo *"]
        editor: [".env", "credentials*"]
"""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PermissionDecision:
    behavior: str  # "allow" | "deny" | "escalate"
    tool_name: str
    action: str
    reason: str
    matched_rule: str | None = None


@dataclass
class PermissionPolicy:
    mode: str = "allow_all"  # "allow_all" | "deny_all" | "rules"
    allow: dict[str, list[str]] = field(default_factory=dict)
    deny: dict[str, list[str]] = field(default_factory=dict)
    escalate: dict[str, list[str]] = field(default_factory=dict)
    on_escalate: Callable | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PermissionPolicy:
        return cls(
            mode=data.get("mode", "allow_all"),
            allow=data.get("allow", {}),
            deny=data.get("deny", {}),
            escalate=data.get("escalate", {}),
        )

    @classmethod
    def allow_all(cls) -> PermissionPolicy:
        return cls(mode="allow_all")

    @classmethod
    def deny_all(cls) -> PermissionPolicy:
        return cls(mode="deny_all")

    @classmethod
    def read_only(cls) -> PermissionPolicy:
        return cls(
            mode="rules",
            allow={"reader": ["*"], "search": ["*"], "context": ["*"]},
            deny={"editor": ["*"], "bash": ["*"]},
        )

    @classmethod
    def safe(cls) -> PermissionPolicy:
        return cls(
            mode="rules",
            allow={
                "reader": ["*"],
                "editor": ["*"],
                "search": ["*"],
                "context": ["*"],
            },
            deny={"bash": ["rm *", "sudo *", "chmod *", "kill *", "mkfs *"]},
            escalate={"bash": ["*"]},
        )

    def check(self, tool_name: str, action: str, args: dict) -> PermissionDecision:
        if self.mode == "allow_all":
            return PermissionDecision("allow", tool_name, action, "mode=allow_all")
        if self.mode == "deny_all":
            return PermissionDecision("deny", tool_name, action, "mode=deny_all")

        match_str = _build_match_string(tool_name, action, args)

        for pattern in self.deny.get(tool_name, []) + self.deny.get("*", []):
            if _matches(match_str, pattern):
                return PermissionDecision(
                    "deny",
                    tool_name,
                    action,
                    f"denied by rule: {pattern}",
                    pattern,
                )

        for pattern in self.allow.get(tool_name, []) + self.allow.get("*", []):
            if _matches(match_str, pattern):
                return PermissionDecision(
                    "allow",
                    tool_name,
                    action,
                    f"allowed by rule: {pattern}",
                    pattern,
                )

        for pattern in self.escalate.get(tool_name, []) + self.escalate.get("*", []):
            if _matches(match_str, pattern):
                return PermissionDecision(
                    "escalate",
                    tool_name,
                    action,
                    f"escalate by rule: {pattern}",
                    pattern,
                )

        return PermissionDecision(
            "allow", tool_name, action, "no matching rule, default allow"
        )

    def to_pre_hook(self) -> Callable:
        """Convert this policy into a Tool preprocessor hook."""
        policy = self

        async def permission_check(
            tool_name: str, action: str, args: dict
        ) -> dict | None:
            decision = policy.check(tool_name, action, args)

            if decision.behavior == "allow":
                return None

            if decision.behavior == "escalate":
                if policy.on_escalate:
                    result = await policy.on_escalate(decision, args)
                    if result is True:
                        return None
                    if isinstance(result, dict):
                        return result
                raise PermissionError(
                    f"Permission escalation required for {tool_name}.{action}: "
                    f"{decision.reason}. No escalation handler configured."
                )

            raise PermissionError(
                f"Permission denied for {tool_name}.{action}: {decision.reason}"
            )

        return permission_check


def _build_match_string(tool_name: str, action: str, args: dict) -> str:
    if tool_name == "bash":
        return args.get("command", "")
    if tool_name == "editor":
        return args.get("file_path", "")
    if tool_name == "reader":
        return args.get("path", "")
    if tool_name == "search":
        return args.get("pattern", "") + " " + (args.get("path") or "")
    return f"{action} {' '.join(str(v) for v in args.values())}"


def _matches(text: str, pattern: str) -> bool:
    if pattern == "*":
        return True
    return fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(
        text.lower(), pattern.lower()
    )
