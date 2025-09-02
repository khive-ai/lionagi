#!/usr/bin/env python3
"""
Test agent composition - demonstrates how agents acquire domain expertise
"""

import subprocess


def test_composition(role: str, domain: str = None):
    """Test agent composition and show the result"""
    cmd = [
        "python",
        "/Users/lion/liongate/.claude/scripts/compose_agent.py",
        role,
    ]
    if domain:
        cmd.extend(["--domain", domain])

    print(f"\n{'=' * 60}")
    print(f"Testing: {role} + {domain if domain else 'no domain'}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")


def main():
    # Test 1: Base research agent
    test_composition("research-agent")

    # Test 2: Research agent with security domain
    test_composition("research-agent", "security")

    # Test 3: Quality agent with rust domain
    test_composition("quality-agent", "rust")

    # Test 4: Innovation agent with python domain
    test_composition("innovation-agent", "python")


if __name__ == "__main__":
    main()
