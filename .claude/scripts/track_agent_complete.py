#!/usr/bin/env python3
"""
Real orchestration tracking - Task agent completion
Simple, practical, actually works.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def track_agent_complete(description: str, timestamp: str, session_dir: str):
    """Track when a Task agent completes - real implementation"""

    workspace_dir = Path(session_dir)
    log_file = workspace_dir / "orchestration.log"
    active_agents_dir = workspace_dir / "active_agents"

    # Generate agent ID from description
    agent_id = description.replace(" ", "_").replace("/", "_")[:50]
    agent_file = active_agents_dir / f"{agent_id}.json"

    # Update agent status
    if agent_file.exists():
        with open(agent_file, "r") as f:
            agent_info = json.load(f)

        agent_info.update({"end_time": timestamp, "status": "completed"})

        with open(agent_file, "w") as f:
            json.dump(agent_info, f, indent=2)

    # Log completion
    with open(log_file, "a") as f:
        f.write(f"{timestamp}: AGENT_COMPLETE: {description}\n")

    # Check if this completes a phase
    check_phase_completion(workspace_dir, timestamp)

    print(f"✅ Tracked agent completion: {description}")


def check_phase_completion(workspace_dir: Path, timestamp: str):
    """Check if all agents in a phase are complete"""
    active_agents_dir = workspace_dir / "active_agents"

    if not active_agents_dir.exists():
        return

    active_count = len(list(active_agents_dir.glob("*.json")))

    if active_count == 0:
        log_file = workspace_dir / "orchestration.log"
        with open(log_file, "a") as f:
            f.write(f"{timestamp}: PHASE_COMPLETE: All agents finished\n")

        print("🎉 Phase complete - all agents finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track Task agent completion")
    parser.add_argument("--description", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--session_dir", required=True)

    args = parser.parse_args()
    track_agent_complete(args.description, args.timestamp, args.session_dir)
