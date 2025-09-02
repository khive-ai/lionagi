#!/usr/bin/env python3
"""
KB Validation Script

Supports validation of all KB template types using Pydantic models.
Now includes orchestrator gatekeeper functionality.

Usage:
    # Template validation
    python validate.py <template_type> <yaml_file>
    python validate.py research_proposal proposal.yaml
    python validate.py decision_document decision.yaml

    # Orchestrator gatekeeper
    python validate.py --gatekeeper-check
    python validate.py --scan-events
    python validate.py --emergency-override "reason"
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    from template_models import TEMPLATE_VALIDATORS, validate_template_data
except ImportError:
    print(
        "Error: Could not import template_models. Make sure you're in the correct directory."
    )
    sys.exit(1)

# Import gatekeeper functionality
try:
    from gatekeeper import (
        GatekeeperException,
        GitHubClient,
        KBEventScanner,
        OrchestratorGatekeeper,
    )

    GATEKEEPER_AVAILABLE = True
except ImportError:
    GATEKEEPER_AVAILABLE = False


def load_yaml_file(file_path: str) -> dict[str, Any]:
    """Load YAML file and return data"""
    try:
        with open(file_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        sys.exit(1)


async def run_gatekeeper_check():
    """Run orchestrator completion check"""
    if not GATEKEEPER_AVAILABLE:
        print("❌ Gatekeeper functionality not available")
        print("Install required dependencies: aiohttp aiofiles")
        return 1

    print("🚪 Running orchestrator completion check...")

    gatekeeper = OrchestratorGatekeeper()

    try:
        result = await gatekeeper.orchestrator_completion_check()
        print("✅ Orchestrator completion check passed!")
        print("🎉 All systems ready for completion")
        return 0

    except GatekeeperException as e:
        print("❌ Orchestrator completion check failed!")
        print(f"🚫 Reason: {e.reason}")

        if e.events:
            print("\n📊 Event Summary:")
            print(f"  • Parallelizable: {len(e.events.parallelizable)}")
            print(f"  • Sequential: {len(e.events.sequential)}")
            print(f"  • Blocked: {len(e.events.blocked)}")
            print(f"  • Active Swarms: {len(e.events.active_swarms)}")

            if e.events.parallelizable:
                print("\n🔄 Process these events in parallel:")
                for event in e.events.parallelizable[:5]:  # Show first 5
                    print(
                        f"    • {event.research_id}: {event.event_type.value}"
                    )

            if e.events.sequential:
                print("\n⏭️ Process these events sequentially:")
                for event in e.events.sequential[:5]:  # Show first 5
                    print(
                        f"    • {event.research_id}: {event.event_type.value}"
                    )

        return 1


async def run_event_scan():
    """Scan for KB lifecycle events"""
    if not GATEKEEPER_AVAILABLE:
        print("❌ Gatekeeper functionality not available")
        return 1

    print("📡 Scanning for KB lifecycle events...")

    async with GitHubClient() as github:
        scanner = KBEventScanner(github)
        events = await scanner.scan_kb_events()

        print(f"✅ Event scan completed at {events.scan_timestamp}")
        print(f"📊 Total events: {events.total_events}")
        print(f"🔄 Parallelizable: {len(events.parallelizable)}")
        print(f"⏭️ Sequential: {len(events.sequential)}")
        print(f"🚫 Blocked: {len(events.blocked)}")
        print(f"🏃 Active Swarms: {len(events.active_swarms)}")

        # Show event details
        if events.parallelizable:
            print("\n🔄 Parallelizable Events:")
            for event in events.parallelizable:
                print(f"  • {event.research_id}: {event.event_type.value}")
                print(
                    f"    Issue #{event.issue_number} | {event.priority} priority"
                )
                print(f"    Swarm: {event.swarm_pattern}")

        if events.sequential:
            print("\n⏭️ Sequential Events:")
            for event in events.sequential:
                print(f"  • {event.research_id}: {event.event_type.value}")
                print(f"    Dependencies: {', '.join(event.dependencies)}")

        if events.blocked:
            print("\n🚫 Blocked Events:")
            for event in events.blocked:
                print(f"  • {event.research_id}: {event.event_type.value}")
                print(
                    f"    Blocked by: {', '.join(event.blocking_conditions)}"
                )

        if events.active_swarms:
            print("\n🏃 Active Swarms:")
            for swarm in events.active_swarms:
                print(f"  • {swarm.swarm_id}")
                print(
                    f"    Status: {swarm.status} | Agents: {', '.join(swarm.agents)}"
                )

        return 0


async def run_emergency_override(reason: str):
    """Run emergency override"""
    if not GATEKEEPER_AVAILABLE:
        print("❌ Gatekeeper functionality not available")
        return 1

    print("🚨 Activating emergency override...")
    print(f"Reason: {reason}")

    gatekeeper = OrchestratorGatekeeper()
    result = await gatekeeper.emergency_override(reason)

    if result:
        print("✅ Emergency override activated")
        print("⚠️ Check override-records/ for audit trail")
        return 0
    print("❌ Emergency override failed")
    return 1


def validate_template(template_type: str, yaml_file: str):
    """Validate a template file"""
    if template_type not in TEMPLATE_VALIDATORS:
        print(f"Error: Unknown template type '{template_type}'")
        print(f"Available types: {', '.join(TEMPLATE_VALIDATORS.keys())}")
        sys.exit(1)

    if not Path(yaml_file).exists():
        print(f"Error: File '{yaml_file}' not found")
        sys.exit(1)

    # Load and validate
    data = load_yaml_file(yaml_file)

    try:
        validated_data = validate_template_data(template_type, data)
        print(f"✅ Validation successful for {template_type}")
        print(f"📄 File: {yaml_file}")

        # Output key validation results
        if template_type == "research_proposal":
            print(f"🔍 Research ID: {validated_data['research_id']}")
            print(f"📊 Category: {validated_data['category']}")
            print(f"⚡ Priority: {validated_data['priority']}")

        elif template_type == "decision_document":
            print(f"🎯 Confidence: {validated_data['confidence_level']}")
            print(f"📋 Status: {validated_data['status']}")

        elif template_type == "implementation_tracking":
            print(f"📈 Progress: {validated_data['completion_percentage']}%")
            print(f"🔄 Status: {validated_data['implementation_status']}")

        elif template_type == "roi_analysis":
            roi = validated_data["roi_calculation"]
            print(f"💰 ROI: {roi.get('roi_percentage', 'N/A')}%")
            print(f"⏱️  Payback: {roi.get('payback_period', 'N/A')}")

    except Exception as e:
        print(f"❌ Validation failed for {template_type}")
        print(f"📄 File: {yaml_file}")
        print(f"🐛 Error: {e!s}")
        sys.exit(1)


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  # Template validation")
        print("  python validate.py <template_type> <yaml_file>")
        print("  # Gatekeeper functions")
        print("  python validate.py --gatekeeper-check")
        print("  python validate.py --scan-events")
        print("  python validate.py --emergency-override 'reason'")
        print(
            f"\nAvailable template types: {', '.join(TEMPLATE_VALIDATORS.keys())}"
        )
        sys.exit(1)

    # Handle gatekeeper commands
    if sys.argv[1] == "--gatekeeper-check":
        return asyncio.run(run_gatekeeper_check())

    if sys.argv[1] == "--scan-events":
        return asyncio.run(run_event_scan())

    if sys.argv[1] == "--emergency-override":
        if len(sys.argv) != 3:
            print("Usage: python validate.py --emergency-override 'reason'")
            sys.exit(1)
        reason = sys.argv[2]
        return asyncio.run(run_emergency_override(reason))

    # Handle template validation
    if len(sys.argv) == 3:
        template_type = sys.argv[1]
        yaml_file = sys.argv[2]
        validate_template(template_type, yaml_file)
        return 0

    print("Invalid arguments. Use --help for usage information.")
    sys.exit(1)


if __name__ == "__main__":
    main()
