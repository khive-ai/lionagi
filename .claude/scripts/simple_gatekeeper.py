#!/usr/bin/env python3
"""
Simple KB Gatekeeper - Prevents premature task completion

This implements the critical requirement from CLAUDE.md:
"ALWAYS check for pending events before completion"

Usage:
    python simple_gatekeeper.py         # Run completion check

Returns:
    Exit code 0: Safe to complete (no pending tasks)
    Exit code 1: Cannot complete (tasks still pending)
"""

import os
import subprocess
import sys


def orchestrator_completion_check():
    """
    CRITICAL FUNCTION: Must be run before declaring any task complete
    This is the implementation of the requirement from CLAUDE.md
    """

    print("🚨 Running mandatory orchestrator completion check...")

    # Use task_master to check for pending tasks
    try:
        # Run task_master.py --check
        result = subprocess.run(
            [sys.executable, "task_master.py", "--check"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
        )

        if result.returncode == 0:
            # No pending tasks
            print("✅ GATEKEEPER APPROVAL: Task completion allowed")
            print("   • No pending parallelizable events")
            print("   • No pending sequential events")
            print("   • System ready for task completion")
            return True
        # Tasks still pending
        print("❌ GATEKEEPER BLOCK: Task completion NOT allowed")
        print("   • Events still pending in the queue")
        print("   • Run 'python task_master.py --list' to see pending tasks")
        print("   • Process all events before attempting completion")
        return False

    except Exception as e:
        print(f"❌ GATEKEEPER ERROR: {e}")
        print("   • Cannot verify task status")
        print("   • Task completion blocked for safety")
        return False


def main():
    """Main entry point"""
    # Check if we can complete
    can_complete = orchestrator_completion_check()

    # Exit with appropriate code
    sys.exit(0 if can_complete else 1)


if __name__ == "__main__":
    main()
