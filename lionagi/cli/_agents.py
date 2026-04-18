# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""Load agent profiles from .lionagi/agents/{name}.md

Profile format (YAML frontmatter + markdown body):

    ---
    model: claude_code/opus
    effort: high
    yolo: true
    ---

    You are an implementer. Write production code, not stubs...

Frontmatter fields (all optional, CLI flags override):
  model:  provider/model spec
  effort: reasoning effort level
  yolo:   auto-approve tool calls
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentProfile:
    name: str
    system_prompt: str = ""
    model: str | None = None
    effort: str | None = None
    yolo: bool = False
    extra: dict = field(default_factory=dict)


def _find_lionagi_dir() -> Path | None:
    """Find .lionagi/ directory — walk up from cwd to git root."""
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if root.returncode == 0:
            candidate = Path(root.stdout.strip()) / ".lionagi"
            if candidate.is_dir():
                return candidate
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / ".lionagi"
        if candidate.is_dir():
            return candidate
    return None


def list_agents() -> list[str]:
    """List available agent profile names."""
    d = _find_lionagi_dir()
    if not d:
        return []
    agents_dir = d / "agents"
    if not agents_dir.is_dir():
        return []
    return sorted(p.stem for p in agents_dir.glob("*.md"))


def load_agent_profile(name: str) -> AgentProfile:
    """Load an agent profile by name.

    Raises FileNotFoundError if .lionagi/agents/{name}.md doesn't exist.
    """
    d = _find_lionagi_dir()
    if not d:
        raise FileNotFoundError(
            "No .lionagi/ directory found. Create .lionagi/agents/ in your repo."
        )

    path = d / "agents" / f"{name}.md"
    if not path.exists():
        available = list_agents()
        msg = f"Agent profile not found: {path}"
        if available:
            msg += f"\nAvailable: {', '.join(available)}"
        raise FileNotFoundError(msg)

    text = path.read_text()
    return _parse_profile(name, text)


def _parse_profile(name: str, text: str) -> AgentProfile:
    """Parse YAML frontmatter + markdown body."""
    frontmatter = {}
    body = text

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()
            for line in fm_text.splitlines():
                line = line.strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    val = val.strip()
                    if val.lower() in ("true", "false"):
                        frontmatter[key.strip()] = val.lower() == "true"
                    else:
                        frontmatter[key.strip()] = val

    return AgentProfile(
        name=name,
        system_prompt=body,
        model=frontmatter.get("model"),
        effort=frontmatter.get("effort"),
        yolo=bool(frontmatter.get("yolo", False)),
        extra={k: v for k, v in frontmatter.items()
               if k not in ("model", "effort", "yolo")},
    )
