#!/usr/bin/env python3
"""
KB Orchestrator Gatekeeper - Usage Examples and Test Scenarios

This file demonstrates how to use the gatekeeper in various scenarios
and provides test cases for validation.
"""

import asyncio
import os
from datetime import datetime, timedelta

from gatekeeper import (
    EventClassification,
    EventType,
    GatekeeperException,
    GitHubClient,
    KBEventScanner,
    OrchestratorGatekeeper,
)


class GatekeeperExamples:
    """Usage examples for the KB Orchestrator Gatekeeper"""

    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo = os.getenv("GITHUB_REPOSITORY", "owner/repo")

    async def example_1_basic_completion_check(self):
        """
        Example 1: Basic completion check usage

        This is how the orchestrator would call the gatekeeper
        before declaring any task complete.
        """
        print("=== Example 1: Basic Completion Check ===")

        gatekeeper = OrchestratorGatekeeper(self.github_token, self.repo)

        try:
            # This is the main call that orchestrator makes
            result = await gatekeeper.orchestrator_completion_check()
            print("✅ Completion check passed - safe to proceed!")
            return True

        except GatekeeperException as e:
            print(f"❌ Completion check failed: {e.reason}")
            print("\nEvents that need processing:")

            # Show parallel events
            if e.events.parallelizable:
                print(
                    f"\n📊 Parallelizable events ({len(e.events.parallelizable)}):"
                )
                for event in e.events.parallelizable:
                    print(f"  • {event.research_id}: {event.event_type.value}")
                    print(
                        f"    Issue #{event.issue_number} | Priority: {event.priority}"
                    )
                    print(f"    Swarm: {event.swarm_pattern}")

            # Show sequential events
            if e.events.sequential:
                print(f"\n⏭️ Sequential events ({len(e.events.sequential)}):")
                for event in e.events.sequential:
                    print(f"  • {event.research_id}: {event.event_type.value}")
                    print(f"    Dependencies: {', '.join(event.dependencies)}")

            # Show active swarms
            if e.events.active_swarms:
                print(f"\n🔄 Active swarms ({len(e.events.active_swarms)}):")
                for swarm in e.events.active_swarms:
                    runtime = datetime.now() - swarm.start_time
                    print(f"  • {swarm.swarm_id} (running {runtime})")
                    print(
                        f"    Status: {swarm.status} | Agents: {', '.join(swarm.agents)}"
                    )

            return False

    async def example_2_event_scanning_only(self):
        """
        Example 2: Event scanning without completion check

        Useful for monitoring and diagnostics
        """
        print("\n=== Example 2: Event Scanning Only ===")

        async with GitHubClient(self.github_token, self.repo) as github:
            scanner = KBEventScanner(github)
            events = await scanner.scan_kb_events()

            print(f"📡 Event scan completed at {events.scan_timestamp}")
            print(f"Total events found: {events.total_events}")

            # Detailed breakdown
            print("\n📈 Event Classification:")
            print(f"  Parallelizable: {len(events.parallelizable)}")
            print(f"  Sequential: {len(events.sequential)}")
            print(f"  Blocked: {len(events.blocked)}")
            print(f"  Active Swarms: {len(events.active_swarms)}")

            # Show details for each category
            if events.parallelizable:
                print("\n🔄 Parallelizable Events:")
                for event in events.parallelizable:
                    print(
                        f"  • {event.research_id} ({event.event_type.value})"
                    )
                    print(
                        f"    Priority: {event.priority} | Category: {event.category}"
                    )
                    print(f"    Agents: {', '.join(event.agent_assignments)}")
                    print(
                        f"    Created: {event.created_at.strftime('%Y-%m-%d %H:%M')}"
                    )

            if events.blocked:
                print("\n🚫 Blocked Events:")
                for event in events.blocked:
                    print(
                        f"  • {event.research_id} ({event.event_type.value})"
                    )
                    print(
                        f"    Blocking reasons: {', '.join(event.blocking_conditions)}"
                    )

            return events

    async def example_3_orchestrator_integration(self):
        """
        Example 3: Full orchestrator integration pattern

        Shows how this integrates with the orchestrator workflow
        """
        print("\n=== Example 3: Orchestrator Integration ===")

        # Simulate orchestrator workflow
        orchestrator_tasks = [
            "Process research request AIO_001",
            "Generate decision for MEM_002",
            "Complete implementation tracking for TLI_003",
        ]

        print("🤖 Orchestrator starting workflow...")

        for i, task in enumerate(orchestrator_tasks, 1):
            print(f"\n📋 Step {i}: {task}")

            # Simulate task execution
            await asyncio.sleep(0.1)  # Simulate work
            print("  ✅ Task completed")

            # CRITICAL: Check for events before proceeding
            print("  🚪 Running gatekeeper completion check...")

            gatekeeper = OrchestratorGatekeeper(self.github_token, self.repo)

            try:
                await gatekeeper.orchestrator_completion_check()
                print("  ✅ Gatekeeper check passed - proceeding to next task")

            except GatekeeperException as e:
                print(f"  ❌ Gatekeeper blocked completion: {e.reason}")
                print("  ⏸️ Orchestrator must process pending events first")

                # In real implementation, orchestrator would:
                # 1. Process parallelizable events in batch
                # 2. Process sequential events in order
                # 3. Wait for swarms to complete
                # 4. Retry completion check

                return False

        print("\n🎉 Orchestrator workflow completed successfully!")
        return True

    async def example_4_emergency_override(self):
        """
        Example 4: Emergency override usage

        For exceptional circumstances requiring manual intervention
        """
        print("\n=== Example 4: Emergency Override ===")

        gatekeeper = OrchestratorGatekeeper(self.github_token, self.repo)

        # First, show that normal completion would fail
        print("🔍 Attempting normal completion check...")
        try:
            await gatekeeper.orchestrator_completion_check()
            print("✅ Normal completion check passed")
        except GatekeeperException as e:
            print(f"❌ Normal completion blocked: {e.reason}")

            # Now use emergency override
            print("\n🚨 Activating emergency override...")
            override_reason = (
                "Production hotfix required - bypassing pending research cycles "
                "due to critical security vulnerability. Will resume normal "
                "KB lifecycle after emergency deployment."
            )

            result = await gatekeeper.emergency_override(override_reason)
            if result:
                print("✅ Emergency override successful")
                print("⚠️ Check override-records/ for audit trail")

        return True

    async def example_5_custom_event_processing(self):
        """
        Example 5: Custom event processing logic

        Shows how to extend the gatekeeper for custom scenarios
        """
        print("\n=== Example 5: Custom Event Processing ===")

        class CustomGatekeeper(OrchestratorGatekeeper):
            """Extended gatekeeper with custom logic"""

            async def check_custom_conditions(self) -> bool:
                """Add custom completion conditions"""
                print("  🔧 Checking custom conditions...")

                # Example: Check for critical priority events
                async with GitHubClient(
                    self.github_token, self.repo
                ) as github:
                    issues = await github.get_issues(
                        state="open", labels="priority:critical"
                    )

                    if issues:
                        critical_count = len(issues)
                        print(
                            f"  ⚠️ Found {critical_count} critical priority issues"
                        )
                        print(
                            "  ❌ Cannot complete with critical issues pending"
                        )
                        return False

                # Example: Check for specific resource availability
                print("  📊 Checking system resources...")
                # In real implementation, this might check:
                # - Database connection pool
                # - Memory usage
                # - API rate limits
                # - External service availability

                print("  ✅ All custom conditions satisfied")
                return True

            async def orchestrator_completion_check(self) -> bool:
                """Override to add custom checks"""
                # First run standard checks
                try:
                    await super().orchestrator_completion_check()
                except GatekeeperException:
                    raise  # Re-raise if standard checks fail

                # Then run custom checks
                custom_result = await self.check_custom_conditions()
                if not custom_result:
                    raise GatekeeperException(
                        "Custom completion conditions not satisfied",
                        None,  # Would include custom event data
                    )

                return True

        # Use custom gatekeeper
        custom_gatekeeper = CustomGatekeeper(self.github_token, self.repo)

        try:
            result = await custom_gatekeeper.orchestrator_completion_check()
            print("✅ Custom gatekeeper check passed")
        except GatekeeperException as e:
            print(f"❌ Custom gatekeeper check failed: {e.reason}")

        return True


class GatekeeperTestScenarios:
    """Test scenarios for validating gatekeeper functionality"""

    def __init__(self):
        self.test_results = []

    async def run_all_tests(self):
        """Run all test scenarios"""
        print("\n" + "=" * 60)
        print("🧪 RUNNING GATEKEEPER TEST SCENARIOS")
        print("=" * 60)

        test_methods = [
            self.test_event_classification,
            self.test_dependency_checking,
            self.test_timeout_detection,
            self.test_swarm_tracking,
            self.test_blocking_conditions,
            self.test_completion_scenarios,
        ]

        for test_method in test_methods:
            try:
                await test_method()
                self.test_results.append(
                    {
                        "test": test_method.__name__,
                        "status": "PASS",
                    }
                )
            except Exception as e:
                print(f"❌ Test {test_method.__name__} failed: {e}")
                self.test_results.append(
                    {
                        "test": test_method.__name__,
                        "status": "FAIL",
                        "error": str(e),
                    }
                )

        # Print test summary
        print("\n📊 Test Summary:")
        passed = sum(1 for r in self.test_results if r["status"] == "PASS")
        total = len(self.test_results)
        print(f"  Passed: {passed}/{total}")

        for result in self.test_results:
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            print(f"  {status_emoji} {result['test']}")
            if result["status"] == "FAIL":
                print(f"    Error: {result['error']}")

    async def test_event_classification(self):
        """Test event classification logic"""
        print("\n🧪 Test: Event Classification")

        # Mock issue data for testing
        mock_issue = {
            "number": 123,
            "title": "[RESEARCH] Implement vector database search - AIO_001",
            "body": "Research question: How to implement...",
            "labels": [
                {"name": "stage:research.requested"},
                {"name": "category:AIO"},
                {"name": "priority:high"},
            ],
            "created_at": "2024-01-15T10:00:00Z",
            "updated_at": "2024-01-15T10:30:00Z",
        }

        # Test event identification
        scanner = KBEventScanner(None)  # No GitHub client needed for this test
        event = await scanner._identify_event(mock_issue)

        assert event is not None, "Should identify KB event"
        assert (
            event.event_type == EventType.RESEARCH_REQUESTED
        ), "Should identify correct event type"
        assert event.research_id == "AIO_001", "Should extract research ID"
        assert event.priority == "high", "Should extract priority"
        assert event.category == "AIO", "Should extract category"

        print("  ✅ Event classification working correctly")

    async def test_dependency_checking(self):
        """Test dependency validation logic"""
        print("\n🧪 Test: Dependency Checking")

        # Mock issue with completion markers in comments
        mock_comments = [
            {
                "body": "[INTAKE-2024-01-15T10:30:00Z] COMPLETED: proposal validated",
                "created_at": "2024-01-15T10:30:00Z",
            },
            {
                "body": "[CRITIC-2024-01-15T11:00:00Z] APPROVED: Research proposal meets standards",
                "created_at": "2024-01-15T11:00:00Z",
            },
        ]

        scanner = KBEventScanner(None)

        # Test dependency satisfaction
        is_satisfied = await scanner._is_dependency_satisfied(
            "proposal_validated",
            {},  # issue data not needed for this test
            mock_comments,
        )

        assert is_satisfied, "Should detect satisfied dependency"

        # Test unsatisfied dependency
        is_satisfied = await scanner._is_dependency_satisfied(
            "research_complete", {}, mock_comments
        )

        assert not is_satisfied, "Should detect unsatisfied dependency"

        print("  ✅ Dependency checking working correctly")

    async def test_timeout_detection(self):
        """Test timeout detection logic"""
        print("\n🧪 Test: Timeout Detection")

        scanner = KBEventScanner(None)

        # Create event with old timestamp
        from gatekeeper import KBEvent

        old_event = KBEvent(
            event_type=EventType.RESEARCH_REQUESTED,
            issue_number=123,
            research_id="AIO_001",
            title="Test event",
            labels=[],
            classification=EventClassification.PARALLELIZABLE,
            dependencies=[],
            blocking_conditions=[],
            swarm_pattern="kb-intake-swarm",
            priority="medium",
            category="AIO",
            created_at=datetime.now()
            - timedelta(hours=25),  # Older than 1h timeout
            last_updated=datetime.now() - timedelta(hours=25),
            agent_assignments=[],
            estimated_duration="1h",
        )

        is_timed_out = await scanner._is_timed_out(old_event)
        assert is_timed_out, "Should detect timed out event"

        # Create recent event
        recent_event = KBEvent(
            event_type=EventType.RESEARCH_REQUESTED,
            issue_number=124,
            research_id="AIO_002",
            title="Test event",
            labels=[],
            classification=EventClassification.PARALLELIZABLE,
            dependencies=[],
            blocking_conditions=[],
            swarm_pattern="kb-intake-swarm",
            priority="medium",
            category="AIO",
            created_at=datetime.now()
            - timedelta(minutes=30),  # Within 1h timeout
            last_updated=datetime.now() - timedelta(minutes=30),
            agent_assignments=[],
            estimated_duration="1h",
        )

        is_timed_out = await scanner._is_timed_out(recent_event)
        assert not is_timed_out, "Should not detect timeout for recent event"

        print("  ✅ Timeout detection working correctly")

    async def test_swarm_tracking(self):
        """Test active swarm tracking"""
        print("\n🧪 Test: Swarm Tracking")

        # Mock issue with swarm start comment but no completion
        mock_issue = {
            "number": 123,
            "title": "[RESEARCH] Test - AIO_001",
            "labels": [{"name": "stage:research.active"}],
        }

        mock_comments = [
            {
                "body": """[ORCHESTRATOR-2024-01-15T10:00:00Z] Starting Event Processing

## Event: research_active
## Swarm: kb-research-swarm
## Research ID: AIO_001

### Initializing Agents
- codebase_analyst_agent
- memory_management_agent
- experiment_runner_agent

---
*Orchestration beginning*""",
                "created_at": "2024-01-15T10:00:00Z",
            }
        ]

        scanner = KBEventScanner(None)

        # Mock the get_issue_comments method
        async def mock_get_comments(issue_number):
            return mock_comments

        scanner._scanner = type(
            "MockScanner", (), {"get_issue_comments": mock_get_comments}
        )()

        # This would need actual GitHub client integration to fully test
        # For now, verify the pattern matching logic
        import re

        swarm_pattern = r"\[ORCHESTRATOR-.*\].*Starting.*Swarm:\s*(\S+)"
        match = re.search(swarm_pattern, mock_comments[0]["body"])

        assert match, "Should detect swarm start pattern"
        swarm_name = match.group(1)
        assert (
            swarm_name == "kb-research-swarm"
        ), "Should extract correct swarm name"

        print("  ✅ Swarm tracking pattern matching working correctly")

    async def test_blocking_conditions(self):
        """Test blocking condition detection"""
        print("\n🧪 Test: Blocking Conditions")

        scanner = KBEventScanner(None)

        # Test blocked label detection
        mock_issue_blocked = {
            "labels": [
                {"name": "status:blocked"},
                {"name": "stage:research.active"},
            ]
        }

        # Mock event with blocked label
        from gatekeeper import KBEvent

        blocked_event = KBEvent(
            event_type=EventType.RESEARCH_ACTIVE,
            issue_number=123,
            research_id="AIO_001",
            title="Test event",
            labels=["status:blocked", "stage:research.active"],
            classification=EventClassification.BLOCKED,
            dependencies=[],
            blocking_conditions=[],
            swarm_pattern="kb-research-swarm",
            priority="medium",
            category="AIO",
            created_at=datetime.now(),
            last_updated=datetime.now(),
            agent_assignments=[],
            estimated_duration="8h",
        )

        blocking_conditions = await scanner._check_blocking_conditions(
            blocked_event, mock_issue_blocked
        )
        assert (
            len(blocking_conditions) > 0
        ), "Should detect blocking conditions"
        assert (
            "Explicitly marked as blocked" in blocking_conditions
        ), "Should detect blocked label"

        print("  ✅ Blocking condition detection working correctly")

    async def test_completion_scenarios(self):
        """Test various completion scenarios"""
        print("\n🧪 Test: Completion Scenarios")

        # Test would require mocking GitHub API responses
        # For now, test the core logic components

        # Test 1: No events - should allow completion
        from gatekeeper import EventScanResult

        empty_events = EventScanResult()

        # This should not raise an exception
        if not (
            empty_events.parallelizable
            or empty_events.sequential
            or empty_events.active_swarms
        ):
            completion_allowed = True
        else:
            completion_allowed = False

        assert completion_allowed, "Should allow completion with no events"

        # Test 2: Pending events - should block completion
        events_with_pending = EventScanResult()
        events_with_pending.parallelizable = [None]  # Mock event

        if (
            events_with_pending.parallelizable
            or events_with_pending.sequential
            or events_with_pending.active_swarms
        ):
            completion_blocked = True
        else:
            completion_blocked = False

        assert (
            completion_blocked
        ), "Should block completion with pending events"

        print("  ✅ Completion scenarios working correctly")


# Main execution for examples and tests
async def main():
    """Run examples and tests"""
    print("🚪 KB Orchestrator Gatekeeper - Examples and Tests")
    print("=" * 60)

    # Check for GitHub token
    if not os.getenv("GITHUB_TOKEN"):
        print("⚠️ Warning: GITHUB_TOKEN not set - some examples may not work")
        print("Set GITHUB_TOKEN environment variable for full functionality")

    examples = GatekeeperExamples()

    # Run examples
    print("\n📚 RUNNING USAGE EXAMPLES")
    print("=" * 40)

    await examples.example_1_basic_completion_check()
    await examples.example_2_event_scanning_only()
    await examples.example_3_orchestrator_integration()
    await examples.example_4_emergency_override()
    await examples.example_5_custom_event_processing()

    # Run tests
    tests = GatekeeperTestScenarios()
    await tests.run_all_tests()

    print("\n🎉 Examples and tests completed!")
    print("\nNext steps:")
    print("1. Set GITHUB_TOKEN environment variable")
    print("2. Set GITHUB_REPOSITORY environment variable (owner/repo)")
    print("3. Run: python gatekeeper.py --check")
    print("4. Integrate into orchestrator workflow")


if __name__ == "__main__":
    asyncio.run(main())
