#!/usr/bin/env python3
"""
Auto-consolidation trigger
When phases complete, automatically suggest consolidation.
"""

import argparse
import re
from pathlib import Path


def auto_consolidate(message: str):
    """Trigger consolidation when phase completes"""

    # Extract phase from message
    phase_match = re.search(r"phase\s*(\w+)", message.lower())
    if not phase_match:
        return

    phase = phase_match.group(1)

    # Find active session
    workspace_dir = Path("/Users/lion/fannrs/claude_workspace")
    if not workspace_dir.exists():
        return

    # Get most recent session
    sessions = [
        d
        for d in workspace_dir.iterdir()
        if d.is_dir() and d.name != "scattered_cleanup"
    ]
    if not sessions:
        return

    latest_session = max(sessions, key=lambda x: x.stat().st_mtime)

    print(
        f"""
🔄 Phase {phase} Complete - Consolidation Available

Run this command to consolidate scattered artifacts:
uv run khive consolidate {latest_session.name} --phase {phase}

Or use this Task agent:
Task(description='consolidate_{latest_session.name}_{phase}', 
     prompt='You are the Consolidation Specialist. Read all {phase} artifacts in {latest_session} and create a unified summary.')
"""
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-consolidation trigger")
    parser.add_argument("--message", required=True)

    args = parser.parse_args()
    auto_consolidate(args.message)
