#!/usr/bin/env python3
"""
Real artifact registration - auto-register when agents create files
No more manual coordination, this happens automatically.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def register_artifact(file_path: str, timestamp: str):
    """Auto-register artifacts when they're created"""

    file_path = Path(file_path)

    # Only register files in claude_workspace
    if "claude_workspace" not in str(file_path):
        return

    # Find the session directory
    parts = file_path.parts
    try:
        workspace_idx = parts.index("claude_workspace")
        session_dir = Path("/".join(parts[: workspace_idx + 2]))
    except (ValueError, IndexError):
        return

    registry_file = session_dir / "artifact_registry.json"

    # Load existing registry
    if registry_file.exists():
        with open(registry_file, "r") as f:
            registry = json.load(f)
    else:
        registry = {
            "session_id": session_dir.name,
            "created_at": timestamp,
            "artifacts": [],
        }

    # Extract agent info from filename
    filename = file_path.name
    if "_" in filename:
        parts = filename.replace(".md", "").split("_")
        if len(parts) >= 3:
            phase = parts[0]
            agent_role = parts[1]
            domain = parts[2] if len(parts) > 2 else "general"
        else:
            phase = "unknown"
            agent_role = "agent"
            domain = "general"
    else:
        phase = "unknown"
        agent_role = "agent"
        domain = "general"

    # Check if already registered
    for artifact in registry["artifacts"]:
        if artifact["artifact_path"] == str(file_path):
            return  # Already registered

    # Add new artifact
    artifact = {
        "agent": f"{agent_role}_{domain}",
        "artifact_path": str(file_path),
        "created_at": timestamp,
        "status": "complete",
        "file_size": file_path.stat().st_size if file_path.exists() else 0,
        "phase": phase,
    }

    registry["artifacts"].append(artifact)

    # Save registry
    with open(registry_file, "w") as f:
        json.dump(registry, f, indent=2)

    # Log registration
    log_file = session_dir / "orchestration.log"
    with open(log_file, "a") as f:
        f.write(f"{timestamp}: ARTIFACT_REGISTERED: {file_path.name}\n")

    print(f"📄 Auto-registered artifact: {file_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-register artifact")
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--timestamp", required=True)

    args = parser.parse_args()
    register_artifact(args.file_path, args.timestamp)
