# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""Branch persistence — save/load/resume for `li` sessions."""

from __future__ import annotations

import json
from pathlib import Path

LIONAGI_HOME = Path.home() / ".lionagi"
LAST_BRANCH_POINTER = LIONAGI_HOME / "last_branch.json"


def find_branch_json(branch_id: str) -> tuple[str, Path]:
    """Scan ~/.lionagi/logs/agents/*/ for a file matching branch_id.

    Returns (provider, path). Raises FileNotFoundError if not found.
    """
    agents_dir = LIONAGI_HOME / "logs" / "agents"
    if not agents_dir.exists():
        raise FileNotFoundError(f"No agent logs directory at {agents_dir}")

    for provider_dir in sorted(agents_dir.iterdir()):
        if not provider_dir.is_dir():
            continue
        candidate = provider_dir / branch_id
        if candidate.exists():
            return provider_dir.name, candidate
        for match in provider_dir.glob(f"{branch_id}*"):
            return provider_dir.name, match

    raise FileNotFoundError(f"No branch log found for id {branch_id!r}")


def load_last_branch() -> tuple[str, str]:
    """Read the last-branch pointer. Returns (provider, branch_id)."""
    if not LAST_BRANCH_POINTER.exists():
        raise FileNotFoundError(
            f"No last-branch pointer at {LAST_BRANCH_POINTER}. "
            "Run `li agent <model> <prompt>` at least once before using -c."
        )
    data = json.loads(LAST_BRANCH_POINTER.read_text())
    return data["provider"], data["branch_id"]


def save_last_branch_pointer(provider: str, branch_id: str) -> None:
    LIONAGI_HOME.mkdir(parents=True, exist_ok=True)
    LAST_BRANCH_POINTER.write_text(
        json.dumps({"provider": provider, "branch_id": branch_id})
    )
