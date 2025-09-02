#!/usr/bin/env python3
"""
KB Deliverable Poster - Posts formatted deliverables to GitHub issues

This script ensures deliverables follow the correct template format before
posting them as GitHub issue comments.

Usage:
    python post_deliverable.py --issue <number> --event <event_num> --file <deliverable_file>
    python post_deliverable.py --issue <number> --event <event_num> --template  # Show template

Example:
    python post_deliverable.py --issue 42 --event 001 --file /tmp/research_proposal.md
"""

import argparse
import os
import re
import subprocess
import sys


class DeliverablePoster:
    """Posts validated deliverables to GitHub issues"""

    def __init__(self, owner: str = "ohdearquant", repo: str = "kb"):
        self.owner = owner
        self.repo = repo

        # Event definitions
        self.events = {
            "001": {
                "name": "Research Requested → Research Proposed",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "Problem Analysis",
                    "Proposed Approach",
                    "Success Criteria",
                    "Status",
                ],
                "next_stage": "stage:research.proposed",
            },
            "002": {
                "name": "Research Proposed → Research Active",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "Research Plan",
                    "Status",
                ],
                "next_stage": "stage:research.active",
            },
            "003": {
                "name": "Research Active → Decision Ready",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "Executive Summary",
                    "Status",
                ],
                "next_stage": "stage:decision.ready",
            },
            "004": {
                "name": "Decision Ready → Decision Review",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "Decision Summary",
                    "Status",
                ],
                "next_stage": "stage:decision.review",
            },
            "005": {
                "name": "Decision Review → Implementation Approved",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "Review Summary",
                    "Status",
                ],
                "next_stage": "stage:implementation.approved",
            },
            "006": {
                "name": "Implementation Approved → Implementation Started",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "Implementation Status",
                    "Status",
                ],
                "next_stage": "stage:metrics.collection",
            },
            "007": {
                "name": "Implementation Started → Metrics Collection",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "Metrics Summary",
                    "Status",
                ],
                "next_stage": "stage:metrics.review",
            },
            "008": {
                "name": "Metrics Collection → Metrics Review",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "ROI Summary",
                    "Status",
                ],
                "next_stage": "stage:knowledge.captured",
            },
            "009": {
                "name": "Metrics Review → Knowledge Captured",
                "required_fields": [
                    "Research ID",
                    "Event",
                    "Agent",
                    "Knowledge Summary",
                    "Status",
                ],
                "next_stage": "TERMINAL",
            },
        }

    def validate_deliverable(
        self, content: str, event_num: str
    ) -> dict[str, any]:
        """Validate deliverable content against template requirements"""
        event = self.events.get(event_num)
        if not event:
            return {
                "valid": False,
                "errors": [f"Unknown event number: {event_num}"],
            }

        errors = []

        # Check required fields (flexible format)
        for field in event["required_fields"]:
            # Check for various formats: **Field**:, ### Field, ## Field
            field_patterns = [
                f"**{field}**:",
                f"### {field}",
                f"## {field}",
                f"**{field}**\n",
            ]

            if not any(pattern in content for pattern in field_patterns):
                errors.append(f"Missing required field: {field}")

        # Check event marker
        expected_event = f"**Event**: {event_num} - {event['name']}"
        if expected_event not in content:
            errors.append(
                f"Missing or incorrect event marker. Expected: {expected_event}"
            )

        # Check for research ID pattern
        research_id_match = re.search(r"[A-Z]{3}_\d{3}", content)
        if not research_id_match:
            errors.append("Missing valid Research ID (format: XXX_###)")

        # Check for agent signature
        agent_match = re.search(
            r"\[.*?_AGENT-\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\]", content
        )
        if not agent_match:
            errors.append("Missing agent signature with timestamp")

        # Check for status marker
        if "**Status**: ✅" not in content:
            errors.append("Missing status completion marker")

        # Check for next stage
        if f"**Next Stage**: {event['next_stage']}" not in content:
            errors.append(
                f"Missing or incorrect next stage. Expected: {event['next_stage']}"
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "research_id": (
                research_id_match.group() if research_id_match else None
            ),
        }

    def post_comment(self, issue_number: int, content: str):
        """Post comment to GitHub issue"""
        # Write content to temp file (handles special characters better)
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as f:
            f.write(content)
            temp_file = f.name

        try:
            # Post comment using gh CLI
            cmd = [
                "gh",
                "issue",
                "comment",
                str(issue_number),
                "--repo",
                f"{self.owner}/{self.repo}",
                "--body-file",
                temp_file,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Deliverable posted to issue #{issue_number}")
                return True
            print(f"❌ Failed to post deliverable: {result.stderr}")
            return False
        finally:
            # Clean up temp file
            os.unlink(temp_file)

    def show_template(self, event_num: str):
        """Display the template for a given event"""
        template_file = "/Users/lion/liongate/.claude/resources/templates/deliverable-templates.md"

        try:
            with open(template_file) as f:
                content = f.read()

            # Extract the specific event template
            event_name = self.events.get(event_num, {}).get("name", "")
            if event_name:
                # Find the section for this event
                pattern = f"## Event {event_num}:.*?(?=## Event \\d{{3}}:|## Validation Rules|$)"
                match = re.search(pattern, content, re.DOTALL)

                if match:
                    print(f"📋 Template for Event {event_num}: {event_name}")
                    print("=" * 60)
                    print(match.group().strip())
                else:
                    print(f"Template section not found for event {event_num}")
            else:
                print(f"Unknown event number: {event_num}")

        except Exception as e:
            print(f"Error reading template: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Post deliverables to KB GitHub issues"
    )
    parser.add_argument("--issue", type=int, help="GitHub issue number")
    parser.add_argument(
        "--event",
        choices=[
            "001",
            "002",
            "003",
            "004",
            "005",
            "006",
            "007",
            "008",
            "009",
        ],
        help="Event number (001-009)",
    )
    parser.add_argument("--file", help="Path to deliverable file")
    parser.add_argument(
        "--template",
        action="store_true",
        help="Show template for the specified event",
    )
    parser.add_argument(
        "--force", action="store_true", help="Post even if validation fails"
    )

    args = parser.parse_args()

    if not args.event:
        parser.error("--event is required")

    poster = DeliverablePoster()

    # Show template mode
    if args.template:
        poster.show_template(args.event)
        return

    # Posting mode
    if not args.issue or not args.file:
        parser.error("--issue and --file are required for posting")

    # Read deliverable content
    try:
        with open(args.file) as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Validate content
    validation = poster.validate_deliverable(content, args.event)

    if not validation["valid"]:
        print("❌ Deliverable validation failed:")
        for error in validation["errors"]:
            print(f"   • {error}")

        if not args.force:
            print("\nUse --force to post anyway (not recommended)")
            sys.exit(1)
        else:
            print("\n⚠️  Forcing post despite validation errors...")
    else:
        print("✅ Deliverable validation passed")
        if validation["research_id"]:
            print(f"   Research ID: {validation['research_id']}")

    # Post to GitHub
    success = poster.post_comment(args.issue, content)

    if success:
        event_info = poster.events[args.event]
        print("\n📌 Next steps for orchestrator:")
        print(f"   1. Update issue label to: {event_info['next_stage']}")
        print(
            f"   2. Command: gh issue edit {args.issue} --add-label '{event_info['next_stage']}'"
        )
        print("   3. Remove old stage label if needed")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
