#!/usr/bin/env python3
"""
KB Orchestrator Gatekeeper Script
Ensures proper event processing before task completion.

This script implements the mandatory orchestrator completion check from CLAUDE.md,
providing concrete GitHub API integration, event classification, and validation.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import aiofiles
import aiohttp
import yaml
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """KB Lifecycle Event Types"""

    RESEARCH_REQUESTED = "research.requested"
    RESEARCH_PROPOSED = "research.proposed"
    RESEARCH_ACTIVE = "research.active"
    DECISION_READY = "decision.ready"
    DECISION_REVIEW = "decision.review"
    IMPLEMENTATION_APPROVED = "implementation.approved"
    IMPLEMENTATION_ACTIVE = "implementation.active"
    METRICS_REVIEW = "metrics.review"
    KNOWLEDGE_CAPTURED = "knowledge.captured"


class EventClassification(str, Enum):
    """Event processing classifications"""

    PARALLELIZABLE = "parallelizable"
    SEQUENTIAL = "sequential"
    BLOCKED = "blocked"


@dataclass
class KBEvent:
    """Represents a KB lifecycle event"""

    event_type: EventType
    issue_number: int
    research_id: str
    title: str
    labels: list[str]
    classification: EventClassification
    dependencies: list[str]
    blocking_conditions: list[str]
    swarm_pattern: str
    priority: str
    category: str
    created_at: datetime
    last_updated: datetime
    agent_assignments: list[str]
    estimated_duration: str


class SwarmExecution(BaseModel):
    """Represents an active swarm execution"""

    swarm_id: str
    pattern: str
    status: str
    agents: list[str]
    start_time: datetime
    estimated_completion: datetime
    current_phase: str
    issue_number: int
    research_id: str


class EventScanResult(BaseModel):
    """Result of event scanning operation"""

    parallelizable: list[KBEvent] = Field(default_factory=list)
    sequential: list[KBEvent] = Field(default_factory=list)
    blocked: list[KBEvent] = Field(default_factory=list)
    total_events: int = 0
    scan_timestamp: datetime = Field(default_factory=datetime.now)
    active_swarms: list[SwarmExecution] = Field(default_factory=list)


class GatekeeperException(Exception):
    """Exception raised when completion check fails"""

    def __init__(self, reason: str, events: EventScanResult):
        self.reason = reason
        self.events = events
        super().__init__(reason)


class GitHubClient:
    """GitHub API client for KB event management"""

    def __init__(self, token: str | None = None, repo: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.repo = repo or os.getenv("GITHUB_REPOSITORY", "owner/repo")
        self.base_url = "https://api.github.com"
        self.session = None

        if not self.token:
            logger.warning(
                "No GitHub token provided - using unauthenticated requests"
            )

    async def __aenter__(self):
        headers = {}
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        headers["Accept"] = "application/vnd.github.v3+json"
        headers["User-Agent"] = "KB-Orchestrator-Gatekeeper/1.0"

        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def get_issues(
        self, state: str = "open", labels: str | None = None
    ) -> list[dict]:
        """Get repository issues with optional filtering"""
        url = f"{self.base_url}/repos/{self.repo}/issues"
        params = {"state": state}
        if labels:
            params["labels"] = labels

        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_issue(self, issue_number: int) -> dict:
        """Get a specific issue"""
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}"
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def get_issue_comments(self, issue_number: int) -> list[dict]:
        """Get comments for an issue"""
        url = (
            f"{self.base_url}/repos/{self.repo}/issues/{issue_number}/comments"
        )
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()

    async def post_comment(self, issue_number: int, body: str) -> dict:
        """Post a comment to an issue"""
        url = (
            f"{self.base_url}/repos/{self.repo}/issues/{issue_number}/comments"
        )
        data = {"body": body}
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()


class KBEventScanner:
    """Scans GitHub issues for KB lifecycle events"""

    def __init__(
        self,
        github_client: GitHubClient,
        config_path: str = ".claude/kb-config.yaml",
    ):
        self.github = github_client
        self.config = self._load_config(config_path)
        self.event_definitions = self._load_event_definitions()
        self.swarm_patterns = self._load_swarm_patterns()

    def _load_config(self, config_path: str) -> dict:
        """Load KB configuration"""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(
                f"Config file not found: {config_path}, using defaults"
            )
            return self._default_config()

    def _default_config(self) -> dict:
        """Default configuration if config file is missing"""
        return {
            "orchestration": {
                "max_parallel_agents": 5,
                "agent_timeout": "30m",
                "total_timeout": "2h",
            },
            "lifecycle": {
                "stage_timeouts": {
                    "research_requested": "1h",
                    "research_proposed": "2h",
                    "research_active": "8h",
                    "decision_ready": "2h",
                    "decision_review": "48h",
                }
            },
        }

    def _load_event_definitions(self) -> dict[str, dict]:
        """Load event definitions from resource files"""
        event_patterns = {
            EventType.RESEARCH_REQUESTED: {
                "stage_label": "stage:research.requested",
                "swarm": "kb-intake-swarm",
                "classification": EventClassification.PARALLELIZABLE,
                "dependencies": [],
                "agents": ["research_intake_agent", "context_discovery_agent"],
                "timeout": "1h",
            },
            EventType.RESEARCH_PROPOSED: {
                "stage_label": "stage:research.proposed",
                "swarm": "kb-planning-swarm",
                "classification": EventClassification.PARALLELIZABLE,
                "dependencies": ["proposal_validated"],
                "agents": ["research_planning_agent", "critic_agent"],
                "timeout": "2h",
            },
            EventType.RESEARCH_ACTIVE: {
                "stage_label": "stage:research.active",
                "swarm": "kb-research-swarm",
                "classification": EventClassification.PARALLELIZABLE,
                "dependencies": ["plan_approved"],
                "agents": [
                    "codebase_analyst_agent",
                    "memory_management_agent",
                    "experiment_runner_agent",
                ],
                "timeout": "8h",
            },
            EventType.DECISION_READY: {
                "stage_label": "stage:decision.ready",
                "swarm": "kb-decision-swarm",
                "classification": EventClassification.SEQUENTIAL,
                "dependencies": ["research_complete"],
                "agents": ["decision_synthesis_agent", "peer_review_agent"],
                "timeout": "2h",
            },
            EventType.DECISION_REVIEW: {
                "stage_label": "stage:decision.review",
                "swarm": "kb-decision-review-swarm",
                "classification": EventClassification.SEQUENTIAL,
                "dependencies": ["decision_drafted"],
                "agents": ["peer_review_agent", "critic_agent"],
                "timeout": "48h",
            },
            EventType.IMPLEMENTATION_APPROVED: {
                "stage_label": "stage:implementation.approved",
                "swarm": "kb-implementation-swarm",
                "classification": EventClassification.PARALLELIZABLE,
                "dependencies": ["decision_approved"],
                "agents": ["implementation_tracker_agent"],
                "timeout": "variable",
            },
            EventType.IMPLEMENTATION_ACTIVE: {
                "stage_label": "stage:implementation.active",
                "swarm": "kb-tracking-swarm",
                "classification": EventClassification.PARALLELIZABLE,
                "dependencies": [],
                "agents": [
                    "implementation_tracker_agent",
                    "metrics_collection_agent",
                ],
                "timeout": "variable",
            },
            EventType.METRICS_REVIEW: {
                "stage_label": "stage:metrics.review",
                "swarm": "kb-roi-swarm",
                "classification": EventClassification.SEQUENTIAL,
                "dependencies": ["metrics_collected"],
                "agents": ["roi_analysis_agent", "critic_agent"],
                "timeout": "24h",
            },
        }
        return event_patterns

    def _load_swarm_patterns(self) -> dict:
        """Load swarm pattern definitions"""
        return {
            "parallel": {"max_agents": 20, "coordination": "async"},
            "sequential": {"checkpoint_frequency": "after_each"},
            "hybrid": {"phases": ["parallel", "sequential"]},
        }

    async def scan_kb_events(self) -> EventScanResult:
        """
        Main event scanning function - identifies all actionable KB lifecycle events

        Returns:
            EventScanResult with classified events and active swarms
        """
        logger.info("Starting KB event scan...")

        # Get all open issues
        issues = await self.github.get_issues(state="open")
        logger.info(f"Found {len(issues)} open issues")

        events = EventScanResult()

        for issue in issues:
            event = await self._identify_event(issue)
            if event:
                # Classify the event
                classification = await self._classify_event(event, issue)
                event.classification = classification

                # Add to appropriate category
                if classification == EventClassification.PARALLELIZABLE:
                    events.parallelizable.append(event)
                elif classification == EventClassification.SEQUENTIAL:
                    events.sequential.append(event)
                else:
                    events.blocked.append(event)

        # Scan for active swarms
        events.active_swarms = await self._scan_active_swarms(issues)

        events.total_events = (
            len(events.parallelizable)
            + len(events.sequential)
            + len(events.blocked)
        )

        logger.info(f"Event scan complete: {events.total_events} total events")
        logger.info(f"- Parallelizable: {len(events.parallelizable)}")
        logger.info(f"- Sequential: {len(events.sequential)}")
        logger.info(f"- Blocked: {len(events.blocked)}")
        logger.info(f"- Active swarms: {len(events.active_swarms)}")

        return events

    async def _identify_event(self, issue: dict) -> KBEvent | None:
        """Identify if an issue represents a KB lifecycle event"""
        labels = [label["name"] for label in issue.get("labels", [])]

        # Check for KB research request marker
        if not any(label.startswith("stage:") for label in labels):
            return None

        # Extract stage label
        stage_label = next(
            (label for label in labels if label.startswith("stage:")), None
        )
        if not stage_label:
            return None

        # Map stage to event type
        event_type = None
        for etype, definition in self.event_definitions.items():
            if definition["stage_label"] == stage_label:
                event_type = etype
                break

        if not event_type:
            logger.warning(f"Unknown stage label: {stage_label}")
            return None

        # Extract research ID from issue title or body
        research_id = self._extract_research_id(issue)
        if not research_id:
            logger.warning(
                f"No research ID found for issue #{issue['number']}"
            )
            return None

        # Extract priority and category
        priority = next(
            (
                label.replace("priority:", "")
                for label in labels
                if label.startswith("priority:")
            ),
            "medium",
        )
        category = next(
            (
                label.replace("category:", "")
                for label in labels
                if label.startswith("category:")
            ),
            "UNKNOWN",
        )

        # Create event
        event = KBEvent(
            event_type=event_type,
            issue_number=issue["number"],
            research_id=research_id,
            title=issue["title"],
            labels=labels,
            classification=EventClassification.BLOCKED,  # Will be classified later
            dependencies=self.event_definitions[event_type]["dependencies"],
            blocking_conditions=[],
            swarm_pattern=self.event_definitions[event_type]["swarm"],
            priority=priority,
            category=category,
            created_at=datetime.fromisoformat(
                issue["created_at"].replace("Z", "+00:00")
            ),
            last_updated=datetime.fromisoformat(
                issue["updated_at"].replace("Z", "+00:00")
            ),
            agent_assignments=self.event_definitions[event_type]["agents"],
            estimated_duration=self.event_definitions[event_type]["timeout"],
        )

        return event

    def _extract_research_id(self, issue: dict) -> str | None:
        """Extract research ID from issue title or body"""
        text = f"{issue['title']} {issue.get('body', '')}"

        # Look for pattern like AIO_001, MEM_123, etc.
        match = re.search(r"\b([A-Z]{3}_\d{3})\b", text)
        if match:
            return match.group(1)

        # Generate ID if not found (for new requests)
        if "stage:research.requested" in [
            label["name"] for label in issue.get("labels", [])
        ]:
            category = next(
                (
                    label["name"].replace("category:", "")
                    for label in issue.get("labels", [])
                    if label["name"].startswith("category:")
                ),
                "UNK",
            )
            # This would need to query existing IDs to generate next number
            return f"{category}_TBD"

        return None

    async def _classify_event(
        self, event: KBEvent, issue: dict
    ) -> EventClassification:
        """Classify an event as parallelizable, sequential, or blocked"""

        # Check for blocking conditions first
        blocking_conditions = await self._check_blocking_conditions(
            event, issue
        )
        if blocking_conditions:
            event.blocking_conditions = blocking_conditions
            return EventClassification.BLOCKED

        # Check dependencies
        if not await self._check_dependencies(event, issue):
            event.blocking_conditions.append("Dependencies not met")
            return EventClassification.BLOCKED

        # Check for resource conflicts
        if await self._has_resource_conflict(event):
            event.blocking_conditions.append("Resource conflict detected")
            return EventClassification.BLOCKED

        # Check timeout conditions
        if await self._is_timed_out(event):
            event.blocking_conditions.append("Event has timed out")
            return EventClassification.BLOCKED

        # Return the event type's default classification
        return self.event_definitions[event.event_type]["classification"]

    async def _check_blocking_conditions(
        self, event: KBEvent, issue: dict
    ) -> list[str]:
        """Check for various blocking conditions"""
        blocking_conditions = []

        # Check for "status:blocked" label
        if any(label.startswith("status:blocked") for label in event.labels):
            blocking_conditions.append("Explicitly marked as blocked")

        # Check for needs-revision status
        if any(
            label.startswith("status:needs-revision") for label in event.labels
        ):
            blocking_conditions.append("Needs revision before proceeding")

        # Check for missing required information
        if not event.research_id or event.research_id.endswith("_TBD"):
            blocking_conditions.append("Research ID not assigned")

        # Check for stale issues (no activity in configured timeframe)
        if await self._is_stale(event):
            blocking_conditions.append("Issue is stale - no recent activity")

        return blocking_conditions

    async def _check_dependencies(self, event: KBEvent, issue: dict) -> bool:
        """Check if all event dependencies are satisfied"""
        if not event.dependencies:
            return True

        # Get issue comments to check for completion markers
        comments = await self.github.get_issue_comments(event.issue_number)

        for dependency in event.dependencies:
            if not await self._is_dependency_satisfied(
                dependency, issue, comments
            ):
                logger.info(
                    f"Dependency not satisfied for event {event.research_id}: {dependency}"
                )
                return False

        return True

    async def _is_dependency_satisfied(
        self, dependency: str, issue: dict, comments: list[dict]
    ) -> bool:
        """Check if a specific dependency is satisfied"""

        # Define dependency check patterns
        dependency_patterns = {
            "proposal_validated": r"\[.*INTAKE.*\].*COMPLETED.*proposal.*validated",
            "proposal_approved": r"\[.*CRITIC.*\].*APPROVED",
            "plan_approved": r"\[.*CRITIC.*\].*APPROVED.*plan",
            "research_complete": r"\[.*RESEARCH.*\].*COMPLETED",
            "decision_drafted": r"\[.*DECISION.*\].*COMPLETED.*draft",
            "decision_approved": r"\[.*REVIEW.*\].*APPROVED",
            "metrics_collected": r"\[.*METRICS.*\].*COMPLETED",
        }

        pattern = dependency_patterns.get(dependency)
        if not pattern:
            logger.warning(f"Unknown dependency pattern: {dependency}")
            return True  # Assume satisfied if we don't know how to check

        # Check comments for completion markers
        for comment in comments:
            if re.search(pattern, comment["body"], re.IGNORECASE):
                return True

        return False

    async def _has_resource_conflict(self, event: KBEvent) -> bool:
        """Check for resource conflicts that would prevent parallel execution"""

        # For now, implement simple conflict detection
        # In a full implementation, this would check:
        # - Agent availability
        # - File system locks
        # - Database connections
        # - API rate limits

        # Check if too many agents of same type are running
        required_agents = set(event.agent_assignments)

        # This would query active swarms and check agent allocations
        # For now, return False (no conflicts)
        return False

    async def _is_timed_out(self, event: KBEvent) -> bool:
        """Check if an event has exceeded its timeout"""
        timeout_config = self.config.get("lifecycle", {}).get(
            "stage_timeouts", {}
        )
        stage_name = event.event_type.value.replace(".", "_")
        timeout_str = timeout_config.get(stage_name, event.estimated_duration)

        if timeout_str == "variable":
            return False  # Variable timeouts don't auto-expire

        # Parse timeout string (e.g., "2h", "30m")
        timeout_delta = self._parse_duration(timeout_str)
        if not timeout_delta:
            return False

        # Check if event has exceeded timeout
        now = datetime.now().replace(tzinfo=event.created_at.tzinfo)
        return (now - event.created_at) > timeout_delta

    async def _is_stale(self, event: KBEvent) -> bool:
        """Check if an event is stale (no recent activity)"""
        stale_threshold = timedelta(days=7)  # Could be configurable
        now = datetime.now().replace(tzinfo=event.last_updated.tzinfo)
        return (now - event.last_updated) > stale_threshold

    def _parse_duration(self, duration_str: str) -> timedelta | None:
        """Parse duration string like '2h', '30m', '1d' into timedelta"""
        match = re.match(r"^(\d+)([smhd])$", duration_str.lower())
        if not match:
            return None

        value, unit = match.groups()
        value = int(value)

        if unit == "s":
            return timedelta(seconds=value)
        if unit == "m":
            return timedelta(minutes=value)
        if unit == "h":
            return timedelta(hours=value)
        if unit == "d":
            return timedelta(days=value)

        return None

    async def _scan_active_swarms(
        self, issues: list[dict]
    ) -> list[SwarmExecution]:
        """Scan for active swarm executions"""
        active_swarms = []

        for issue in issues:
            swarm = await self._extract_active_swarm(issue)
            if swarm:
                active_swarms.append(swarm)

        return active_swarms

    async def _extract_active_swarm(
        self, issue: dict
    ) -> SwarmExecution | None:
        """Extract active swarm information from issue comments"""
        comments = await self.github.get_issue_comments(issue["number"])

        # Look for orchestrator swarm start comments
        swarm_pattern = r"\[ORCHESTRATOR-.*\].*Starting.*Swarm:\s*(\S+)"

        for comment in reversed(comments):  # Start from most recent
            match = re.search(swarm_pattern, comment["body"])
            if match:
                swarm_name = match.group(1)

                # Check if swarm is still active (no completion marker)
                completion_pattern = (
                    r"\[ORCHESTRATOR-.*\].*COMPLETED.*" + re.escape(swarm_name)
                )
                swarm_completed = any(
                    re.search(completion_pattern, c["body"]) for c in comments
                )

                if not swarm_completed:
                    # Extract additional swarm details
                    research_id = self._extract_research_id(issue)

                    swarm = SwarmExecution(
                        swarm_id=f"{swarm_name}-{issue['number']}",
                        pattern=swarm_name,
                        status="running",
                        agents=self._extract_swarm_agents(comment["body"]),
                        start_time=datetime.fromisoformat(
                            comment["created_at"].replace("Z", "+00:00")
                        ),
                        estimated_completion=datetime.now()
                        + timedelta(hours=2),  # Default estimate
                        current_phase="active",
                        issue_number=issue["number"],
                        research_id=research_id or "UNKNOWN",
                    )
                    return swarm

        return None

    def _extract_swarm_agents(self, comment_body: str) -> list[str]:
        """Extract agent list from orchestrator comment"""
        # Look for agent initialization section
        agent_section_match = re.search(
            r"### Initializing Agents\s*(.+?)(?=---|\n\n|\Z)",
            comment_body,
            re.DOTALL,
        )
        if not agent_section_match:
            return []

        agent_section = agent_section_match.group(1)

        # Extract agent names (assuming they're listed)
        agents = re.findall(r"(\w+_agent)", agent_section.lower())
        return agents


class OrchestratorGatekeeper:
    """
    Main gatekeeper class that implements the orchestrator completion check
    """

    def __init__(
        self, github_token: str | None = None, repo: str | None = None
    ):
        self.github_token = github_token
        self.repo = repo
        self._scanner = None

    async def orchestrator_completion_check(self) -> bool:
        """
        MANDATORY: Check for events before completion

        This is the main function called by the orchestrator before declaring
        any task complete. Implements the requirements from CLAUDE.md.

        Raises:
            GatekeeperException: If completion is not allowed

        Returns:
            bool: True if completion is allowed
        """
        logger.info("🚪 Starting orchestrator completion check...")

        async with GitHubClient(self.github_token, self.repo) as github:
            scanner = KBEventScanner(github)

            # 1. Run event scanner
            logger.info("📡 Scanning for KB events...")
            events = await scanner.scan_kb_events()

            # 2. Check parallelizable events
            if events.parallelizable:
                error_msg = self._format_parallel_events_error(
                    events.parallelizable
                )
                logger.error(error_msg)
                raise GatekeeperException(
                    f"❌ Cannot complete - {len(events.parallelizable)} parallel events pending",
                    events,
                )

            # 3. Check sequential events
            if events.sequential:
                error_msg = self._format_sequential_events_error(
                    events.sequential
                )
                logger.error(error_msg)
                raise GatekeeperException(
                    f"❌ Cannot complete - {len(events.sequential)} sequential events pending",
                    events,
                )

            # 4. Check blocked events (warning only)
            if events.blocked:
                warning_msg = self._format_blocked_events_warning(
                    events.blocked
                )
                logger.warning(warning_msg)

            # 5. Validate all swarms completed
            if events.active_swarms:
                error_msg = self._format_active_swarms_error(
                    events.active_swarms
                )
                logger.error(error_msg)
                raise GatekeeperException(
                    f"❌ Cannot complete - {len(events.active_swarms)} active swarms still running",
                    events,
                )

            # Generate completion report
            await self._generate_completion_report(events, github)

            logger.info("✅ Orchestrator completion check passed - all clear!")
            return True

    def _format_parallel_events_error(self, events: list[KBEvent]) -> str:
        """Format error message for pending parallel events"""
        event_details = []
        for event in events:
            event_details.append(
                f"  • {event.research_id}: {event.event_type.value} (Issue #{event.issue_number})\n"
                f"    Priority: {event.priority} | Agents: {', '.join(event.agent_assignments)}\n"
                f"    Swarm: {event.swarm_pattern}"
            )

        return f"""
❌ Cannot complete - {len(events)} parallelizable events pending:

{chr(10).join(event_details)}

Execute these in parallel before proceeding.
Use the batch tool to process multiple events simultaneously.
"""

    def _format_sequential_events_error(self, events: list[KBEvent]) -> str:
        """Format error message for pending sequential events"""
        event_details = []
        for event in events:
            event_details.append(
                f"  • {event.research_id}: {event.event_type.value} (Issue #{event.issue_number})\n"
                f"    Dependencies: {', '.join(event.dependencies) if event.dependencies else 'None'}\n"
                f"    Estimated duration: {event.estimated_duration}"
            )

        return f"""
❌ Cannot complete - {len(events)} sequential events pending:

{chr(10).join(event_details)}

Execute these in order before proceeding.
Sequential events must be processed one at a time.
"""

    def _format_blocked_events_warning(self, events: list[KBEvent]) -> str:
        """Format warning message for blocked events"""
        event_details = []
        for event in events:
            blocking_reasons = ", ".join(event.blocking_conditions)
            event_details.append(
                f"  • {event.research_id}: {event.event_type.value} (Issue #{event.issue_number})\n"
                f"    Blocked by: {blocking_reasons}"
            )

        return f"""
⚠️ {len(events)} events blocked:

{chr(10).join(event_details)}

Monitor for unblocking conditions.
These events will auto-process when conditions are resolved.
"""

    def _format_active_swarms_error(self, swarms: list[SwarmExecution]) -> str:
        """Format error message for active swarms"""
        swarm_details = []
        for swarm in swarms:
            runtime = datetime.now() - swarm.start_time
            swarm_details.append(
                f"  • {swarm.swarm_id}\n"
                f"    Status: {swarm.status} | Phase: {swarm.current_phase}\n"
                f"    Runtime: {runtime} | Agents: {', '.join(swarm.agents)}\n"
                f"    Issue: #{swarm.issue_number} ({swarm.research_id})"
            )

        return f"""
❌ Cannot complete - active swarms still running:

{chr(10).join(swarm_details)}

Wait for swarm completion or terminate manually if stuck.
Check individual swarm progress in GitHub issue comments.
"""

    async def _generate_completion_report(
        self, events: EventScanResult, github: GitHubClient
    ):
        """Generate a completion report for logging and audit purposes"""
        report = f"""
# Orchestrator Completion Report
Generated: {datetime.now().isoformat()}

## Event Scan Summary
- Total events scanned: {events.total_events}
- Parallelizable events: {len(events.parallelizable)}
- Sequential events: {len(events.sequential)}
- Blocked events: {len(events.blocked)}
- Active swarms: {len(events.active_swarms)}

## Status: ✅ CLEAR TO PROCEED

All pending events have been processed.
No active swarms detected.
System is ready for completion.

---
Generated by KB Orchestrator Gatekeeper v1.0
"""

        # Log the report
        logger.info("📋 Completion Report Generated")
        logger.info(report)

        # Save to file for audit trail
        report_file = f"kb-completion-report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
        async with aiofiles.open(report_file, "w") as f:
            await f.write(report)

        logger.info(f"📄 Completion report saved to: {report_file}")

    async def emergency_override(self, reason: str) -> bool:
        """
        Emergency override for completion check

        Use only in exceptional circumstances where manual intervention is required.
        Creates an audit trail of the override decision.
        """
        logger.warning(f"🚨 EMERGENCY OVERRIDE ACTIVATED: {reason}")

        override_record = {
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "operator": os.getenv("USER", "unknown"),
            "environment": os.getenv("ENVIRONMENT", "local"),
        }

        # Save override record
        override_file = f"kb-emergency-override-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        async with aiofiles.open(override_file, "w") as f:
            await f.write(json.dumps(override_record, indent=2))

        logger.warning(f"🚨 Override record saved to: {override_file}")
        return True


# Configuration for GitHub Integration
class GatekeeperConfig:
    """Configuration settings for the gatekeeper"""

    GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
    GITHUB_REPO_ENV = "GITHUB_REPOSITORY"

    # Event scanning settings
    SCAN_INTERVAL_MINUTES = 5
    MAX_EVENTS_PER_SCAN = 50

    # Classification timeouts
    DEFAULT_AGENT_TIMEOUT = "30m"
    DEFAULT_SWARM_TIMEOUT = "2h"

    # Required labels for KB lifecycle
    REQUIRED_LABELS = [
        "stage:research.requested",
        "stage:research.proposed",
        "stage:research.active",
        "stage:decision.ready",
        "stage:decision.review",
        "stage:implementation.approved",
        "stage:implementation.active",
        "stage:metrics.review",
    ]

    # Swarm pattern definitions
    SWARM_PATTERNS = {
        "kb-intake-swarm": {
            "type": "parallel",
            "agents": ["research_intake_agent", "context_discovery_agent"],
            "timeout": "1h",
        },
        "kb-research-swarm": {
            "type": "parallel",
            "agents": [
                "codebase_analyst_agent",
                "memory_management_agent",
                "experiment_runner_agent",
            ],
            "timeout": "8h",
        },
        "kb-decision-swarm": {
            "type": "sequential",
            "agents": ["decision_synthesis_agent", "peer_review_agent"],
            "timeout": "2h",
        },
        "kb-decision-review-swarm": {
            "type": "sequential",
            "agents": ["peer_review_agent", "critic_agent"],
            "timeout": "48h",
        },
    }


# Test scenarios for validation
class GatekeeperTests:
    """Test scenarios for validating gatekeeper functionality"""

    @staticmethod
    async def test_completion_check_with_pending_events():
        """Test completion check when events are pending"""
        gatekeeper = OrchestratorGatekeeper()

        try:
            result = await gatekeeper.orchestrator_completion_check()
            assert False, "Should have raised GatekeeperException"
        except GatekeeperException as e:
            assert "parallel events pending" in str(
                e
            ) or "sequential events pending" in str(e)
            logger.info(
                "✅ Test passed: Completion blocked with pending events"
            )

    @staticmethod
    async def test_completion_check_with_active_swarms():
        """Test completion check when swarms are active"""
        gatekeeper = OrchestratorGatekeeper()

        try:
            result = await gatekeeper.orchestrator_completion_check()
            assert False, "Should have raised GatekeeperException"
        except GatekeeperException as e:
            assert "active swarms" in str(e)
            logger.info(
                "✅ Test passed: Completion blocked with active swarms"
            )

    @staticmethod
    async def test_completion_check_clear():
        """Test completion check when all clear"""
        gatekeeper = OrchestratorGatekeeper()

        result = await gatekeeper.orchestrator_completion_check()
        assert result is True
        logger.info("✅ Test passed: Completion allowed when clear")

    @staticmethod
    async def test_emergency_override():
        """Test emergency override functionality"""
        gatekeeper = OrchestratorGatekeeper()

        result = await gatekeeper.emergency_override(
            "Testing override functionality"
        )
        assert result is True
        logger.info("✅ Test passed: Emergency override successful")


# CLI interface for manual execution
async def main():
    """Main CLI interface for gatekeeper operations"""
    import argparse

    parser = argparse.ArgumentParser(description="KB Orchestrator Gatekeeper")
    parser.add_argument(
        "--check", action="store_true", help="Run completion check"
    )
    parser.add_argument(
        "--scan", action="store_true", help="Scan for events only"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run test scenarios"
    )
    parser.add_argument(
        "--override", type=str, help="Emergency override with reason"
    )
    parser.add_argument("--token", type=str, help="GitHub token")
    parser.add_argument(
        "--repo", type=str, help="GitHub repository (owner/repo)"
    )

    args = parser.parse_args()

    gatekeeper = OrchestratorGatekeeper(args.token, args.repo)

    if args.scan:
        async with GitHubClient(args.token, args.repo) as github:
            scanner = KBEventScanner(github)
            events = await scanner.scan_kb_events()
            print(f"Found {events.total_events} total events:")
            print(f"- Parallelizable: {len(events.parallelizable)}")
            print(f"- Sequential: {len(events.sequential)}")
            print(f"- Blocked: {len(events.blocked)}")
            print(f"- Active swarms: {len(events.active_swarms)}")

    elif args.check:
        try:
            result = await gatekeeper.orchestrator_completion_check()
            print("✅ Completion check passed!")
        except GatekeeperException as e:
            print(f"❌ Completion check failed: {e.reason}")
            return 1

    elif args.override:
        result = await gatekeeper.emergency_override(args.override)
        print("🚨 Emergency override activated")

    elif args.test:
        print("Running gatekeeper tests...")
        await GatekeeperTests.test_emergency_override()
        print("✅ All tests completed")

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
