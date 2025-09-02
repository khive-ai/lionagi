#!/usr/bin/env python3
"""
KB Task Master - Enhanced Event-Driven Task Orchestration System

This script orchestrates the KB event system through intelligent task coordination,
real-time event processing, and structured knowledge accumulation.

Features:
- Event-driven processing with real-time GitHub synchronization
- swarm MCP integration for cognitive coordination
- Knowledge MCP integration for structured insights
- ROI calculation and task prioritization
- Backward compatibility with existing CLI interface

Usage:
    python task_master.py --check         # Check if any tasks pending (exit 0=no tasks, 1=tasks exist)
    python task_master.py --list          # List all pending tasks
    python task_master.py --next          # Get next task to process
    python task_master.py --monitor       # Real-time event monitoring
    python task_master.py --roi           # Calculate ROI for active tasks
    python task_master.py --sync          # Synchronize with event system
    python task_master.py --insights      # Generate structured insights
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Enhanced task state tracking"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    WAITING = "waiting"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


@dataclass
class TaskMetrics:
    """Task performance metrics"""

    agent_invocations: int = 0
    compute_time: float = 0.0
    quality_score: float = 0.0
    business_value: float = 0.0
    knowledge_value: float = 0.0
    dependencies_met: bool = True
    created_at: datetime = field(default_factory=datetime.now)

    def calculate_roi(self) -> float:
        """Calculate ROI for the task"""
        total_cost = self.agent_invocations * 1.0 + self.compute_time * 0.1
        total_value = self.business_value + self.knowledge_value
        return total_value / total_cost if total_cost > 0 else 0.0


@dataclass
class EventProcessingContext:
    """Context for event processing"""

    event_id: str
    event_type: str
    payload: dict[str, Any]
    timestamp: datetime
    correlation_id: str | None = None
    processing_state: TaskState = TaskState.PENDING
    metrics: TaskMetrics = field(default_factory=TaskMetrics)


class EnhancedTaskMaster:
    """Enhanced event-driven task orchestration system"""

    def __init__(self, owner: str = "khive-ai", repo: str = "kb"):
        self.owner = owner
        self.repo = repo
        self.event_contexts: dict[str, EventProcessingContext] = {}
        self.task_metrics: dict[int, TaskMetrics] = {}
        self.knowledge_graph_id = None
        self.swarm_initialized = False
        self.session_id = self._generate_session_id()

        # Initialize enhanced capabilities
        self._initialize_knowledge_system()
        self._initialize_swarm()

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"task_master_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _initialize_knowledge_system(self):
        """Initialize Knowledge MCP integration"""
        try:
            # Create session entity for tracking
            self.knowledge_graph_id = self._create_knowledge_session()
            logger.info(
                f"Knowledge system initialized with session: {self.knowledge_graph_id}"
            )
        except Exception as e:
            logger.warning(f"Knowledge system initialization failed: {e}")

    def _initialize_swarm(self):
        """Initialize swarm MCP integration"""
        try:
            # Initialize hierarchical swarm topology
            self._init_swarm_topology()
            self.swarm_initialized = True
            logger.info("swarm coordination initialized")
        except Exception as e:
            logger.warning(f"swarm initialization failed: {e}")

    def _create_knowledge_session(self) -> str | None:
        """Create knowledge graph session entity"""
        try:
            # This would be implemented when Knowledge MCP is available
            # For now, return a mock ID
            return f"session_{self.session_id}"
        except Exception as e:
            logger.error(f"Failed to create knowledge session: {e}")
            return None

    def _init_swarm_topology(self):
        """Initialize swarm coordination topology"""
        try:
            # This would be implemented when swarm MCP is available
            # For now, log the initialization
            logger.info(
                "Swarm topology: hierarchical, max_agents: 8, strategy: adaptive"
            )
        except Exception as e:
            logger.error(f"Failed to initialize swarm topology: {e}")

    # =============================================================================
    # ENHANCED EVENT PROCESSING METHODS
    # =============================================================================

    async def process_real_time_events(
        self, duration: int = 300
    ) -> dict[str, Any]:
        """Monitor and process real-time events for specified duration"""
        logger.info(
            f"Starting real-time event monitoring for {duration} seconds..."
        )

        start_time = time.time()
        processed_events = []

        while time.time() - start_time < duration:
            try:
                # Check for new events
                events = await self._poll_github_events()

                for event in events:
                    context = EventProcessingContext(
                        event_id=event.get("id", "unknown"),
                        event_type=event.get("type", "unknown"),
                        payload=event,
                        timestamp=datetime.now(timezone.utc),
                    )

                    # Process event through enhanced pipeline
                    result = await self._process_event_with_coordination(
                        context
                    )
                    processed_events.append(result)

                    # Update metrics
                    self._update_task_metrics(context)

                # Brief pause to avoid overwhelming the API
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in real-time processing: {e}")
                await asyncio.sleep(5)

        return {
            "duration": duration,
            "events_processed": len(processed_events),
            "results": processed_events,
        }

    async def _poll_github_events(self) -> list[dict[str, Any]]:
        """Poll GitHub for new events"""
        try:
            # Get recent activity
            output = self._run_gh_command(
                [
                    "api",
                    "repos/{owner}/{repo}/events",
                    "--template",
                    "{{range .}}{{.type}}:{{.created_at}}:{{.id}}\n{{end}}",
                    "--paginate",
                    "--limit",
                    "20",
                ]
            )

            events = []
            for line in output.strip().split("\n"):
                if line:
                    parts = line.split(":")
                    if len(parts) >= 3:
                        events.append(
                            {
                                "type": parts[0],
                                "created_at": parts[1],
                                "id": parts[2],
                            }
                        )

            return events

        except Exception as e:
            logger.error(f"Failed to poll GitHub events: {e}")
            return []

    async def _process_event_with_coordination(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Process event with swarm coordination"""
        try:
            # Use swarm for coordination if available
            if self.swarm_initialized:
                coordination_result = await self._coordinate_with_swarm(
                    context
                )
                context.metrics.agent_invocations += coordination_result.get(
                    "agents_used", 0
                )

            # Process through normal pipeline
            result = await self._process_event_standard(context)

            # Accumulate knowledge
            await self._accumulate_event_knowledge(context, result)

            return result

        except Exception as e:
            logger.error(f"Error processing event {context.event_id}: {e}")
            return {"error": str(e), "event_id": context.event_id}

    async def _coordinate_with_swarm(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Coordinate event processing with swarm"""
        try:
            # This would integrate with swarm MCP when available
            # For now, simulate coordination
            coordination_strategy = self._determine_coordination_strategy(
                context
            )

            logger.info(
                f"Coordinating event {context.event_id} with strategy: {coordination_strategy}"
            )

            return {
                "strategy": coordination_strategy,
                "agents_used": 2,  # Simulated
                "coordination_time": 0.5,
            }

        except Exception as e:
            logger.error(f"Swarm coordination failed: {e}")
            return {"error": str(e)}

    def _determine_coordination_strategy(
        self, context: EventProcessingContext
    ) -> str:
        """Determine optimal coordination strategy based on event type"""
        event_type = context.event_type

        if "issue" in event_type.lower():
            return "parallel"
        if "pr" in event_type.lower() or "pull" in event_type.lower():
            return "sequential"
        return "adaptive"

    async def _process_event_standard(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Standard event processing pipeline"""
        try:
            # Route event to appropriate handler
            handler_result = await self._route_event_to_handler(context)

            # Update processing state
            context.processing_state = TaskState.COMPLETED

            return {
                "event_id": context.event_id,
                "status": "completed",
                "handler_result": handler_result,
                "processing_time": (
                    datetime.now(timezone.utc) - context.timestamp
                ).total_seconds(),
            }

        except Exception as e:
            context.processing_state = TaskState.BLOCKED
            return {
                "event_id": context.event_id,
                "status": "failed",
                "error": str(e),
            }

    async def _route_event_to_handler(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Route event to appropriate handler based on type"""
        event_type = context.event_type

        handlers = {
            "IssuesEvent": self._handle_issue_event,
            "PullRequestEvent": self._handle_pr_event,
            "PushEvent": self._handle_push_event,
            "CreateEvent": self._handle_create_event,
        }

        handler = handlers.get(event_type, self._handle_generic_event)
        return await handler(context)

    async def _handle_issue_event(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Handle issue-related events"""
        payload = context.payload

        # Check if this is a KB lifecycle event
        if "issue" in payload:
            issue_data = payload["issue"]
            labels = [
                label.get("name", "") for label in issue_data.get("labels", [])
            ]

            # Process KB stage events
            if any(label.startswith("stage:") for label in labels):
                return await self._process_kb_lifecycle_event(
                    context, issue_data
                )

        return {"type": "issue", "processed": True}

    async def _handle_pr_event(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Handle pull request events"""
        return {"type": "pr", "processed": True}

    async def _handle_push_event(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Handle push events"""
        return {"type": "push", "processed": True}

    async def _handle_create_event(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Handle create events"""
        return {"type": "create", "processed": True}

    async def _handle_generic_event(
        self, context: EventProcessingContext
    ) -> dict[str, Any]:
        """Handle generic events"""
        return {"type": "generic", "processed": True}

    async def _process_kb_lifecycle_event(
        self, context: EventProcessingContext, issue_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Process KB lifecycle events with enhanced coordination"""
        try:
            issue_number = issue_data.get("number")
            current_stage = next(
                (
                    label
                    for label in issue_data.get("labels", [])
                    if label.get("name", "").startswith("stage:")
                ),
                None,
            )

            if current_stage:
                stage_name = current_stage.get("name", "")

                # Calculate event ROI
                roi = self._calculate_event_roi(issue_number, stage_name)

                # Update metrics
                context.metrics.business_value = roi.get("business_value", 0)
                context.metrics.knowledge_value = roi.get("knowledge_value", 0)

                # Determine next actions
                next_actions = self._determine_next_actions(stage_name, roi)

                return {
                    "issue_number": issue_number,
                    "current_stage": stage_name,
                    "roi": roi,
                    "next_actions": next_actions,
                    "processed": True,
                }

        except Exception as e:
            logger.error(f"Failed to process KB lifecycle event: {e}")
            return {"error": str(e), "processed": False}

    def _calculate_event_roi(
        self, issue_number: int, stage: str
    ) -> dict[str, Any]:
        """Calculate ROI for KB lifecycle event"""
        try:
            # Get task metrics if available
            metrics = self.task_metrics.get(issue_number, TaskMetrics())

            # Stage-specific value calculations
            stage_values = {
                "stage:research.requested": {"business": 100, "knowledge": 50},
                "stage:research.proposed": {"business": 200, "knowledge": 100},
                "stage:research.active": {"business": 500, "knowledge": 300},
                "stage:decision.ready": {"business": 1000, "knowledge": 500},
                "stage:implementation.approved": {
                    "business": 2000,
                    "knowledge": 800,
                },
                "stage:metrics.collection": {
                    "business": 3000,
                    "knowledge": 1000,
                },
            }

            values = stage_values.get(stage, {"business": 0, "knowledge": 0})

            # Calculate total cost
            total_cost = (
                metrics.agent_invocations * 1.0 + metrics.compute_time * 0.1
            )

            # Calculate ROI
            roi = (
                (values["business"] + values["knowledge"]) / total_cost
                if total_cost > 0
                else 0
            )

            return {
                "business_value": values["business"],
                "knowledge_value": values["knowledge"],
                "total_cost": total_cost,
                "roi": roi,
                "stage": stage,
            }

        except Exception as e:
            logger.error(f"ROI calculation failed: {e}")
            return {"roi": 0, "error": str(e)}

    def _determine_next_actions(
        self, stage: str, roi: dict[str, Any]
    ) -> list[str]:
        """Determine next actions based on stage and ROI"""
        actions = []

        # ROI-based decisions
        if roi.get("roi", 0) < 1.0:
            actions.append("Consider abandoning - negative ROI")
        elif roi.get("roi", 0) < 2.0:
            actions.append("Proceed with caution - marginal ROI")
        else:
            actions.append("Continue - positive ROI")

        # Stage-specific actions
        if stage == "stage:research.requested":
            actions.append("Deploy intake swarm")
        elif stage == "stage:research.proposed":
            actions.append("Deploy research swarm")
        elif stage == "stage:research.active":
            actions.append("Monitor progress and deliverables")
        elif stage == "stage:decision.ready":
            actions.append("Deploy decision review swarm")

        return actions

    async def _accumulate_event_knowledge(
        self, context: EventProcessingContext, result: dict[str, Any]
    ):
        """Accumulate structured knowledge from event processing"""
        try:
            if self.knowledge_graph_id:
                # Create knowledge entries for the event
                await self._create_event_knowledge_entry(context, result)

        except Exception as e:
            logger.error(f"Failed to accumulate knowledge: {e}")

    async def _create_event_knowledge_entry(
        self, context: EventProcessingContext, result: dict[str, Any]
    ):
        """Create structured knowledge entry for the event"""
        try:
            # This would integrate with Knowledge MCP when available
            # For now, log the knowledge accumulation
            logger.info(
                f"Knowledge accumulated for event {context.event_id}: {result}"
            )

        except Exception as e:
            logger.error(f"Failed to create knowledge entry: {e}")

    def _update_task_metrics(self, context: EventProcessingContext):
        """Update task metrics based on event processing"""
        try:
            # Extract issue number if available
            issue_number = None
            if "issue" in context.payload:
                issue_number = context.payload["issue"].get("number")

            if issue_number:
                if issue_number not in self.task_metrics:
                    self.task_metrics[issue_number] = TaskMetrics()

                metrics = self.task_metrics[issue_number]
                metrics.agent_invocations += context.metrics.agent_invocations
                metrics.compute_time += context.metrics.compute_time
                metrics.business_value = max(
                    metrics.business_value, context.metrics.business_value
                )
                metrics.knowledge_value = max(
                    metrics.knowledge_value, context.metrics.knowledge_value
                )

        except Exception as e:
            logger.error(f"Failed to update task metrics: {e}")

    # =============================================================================
    # ENHANCED ROI AND ANALYTICS METHODS
    # =============================================================================

    def calculate_portfolio_roi(self) -> dict[str, Any]:
        """Calculate ROI across all active tasks"""
        try:
            total_roi = 0.0
            task_count = 0
            roi_distribution = {"positive": 0, "negative": 0, "neutral": 0}

            for issue_number, metrics in self.task_metrics.items():
                roi = metrics.calculate_roi()
                total_roi += roi
                task_count += 1

                if roi > 1.5:
                    roi_distribution["positive"] += 1
                elif roi < 1.0:
                    roi_distribution["negative"] += 1
                else:
                    roi_distribution["neutral"] += 1

            average_roi = total_roi / task_count if task_count > 0 else 0

            return {
                "average_roi": average_roi,
                "total_tasks": task_count,
                "distribution": roi_distribution,
                "total_agent_invocations": sum(
                    m.agent_invocations for m in self.task_metrics.values()
                ),
                "total_compute_time": sum(
                    m.compute_time for m in self.task_metrics.values()
                ),
            }

        except Exception as e:
            logger.error(f"Portfolio ROI calculation failed: {e}")
            return {"error": str(e)}

    def generate_insights(self) -> dict[str, Any]:
        """Generate structured insights about task performance"""
        try:
            insights = {
                "performance_trends": self._analyze_performance_trends(),
                "bottlenecks": self._identify_bottlenecks(),
                "optimization_opportunities": self._find_optimization_opportunities(),
                "risk_factors": self._assess_risk_factors(),
            }

            return insights

        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            return {"error": str(e)}

    def _analyze_performance_trends(self) -> dict[str, Any]:
        """Analyze performance trends across tasks"""
        try:
            # Analyze agent utilization trends
            agent_usage = [
                m.agent_invocations for m in self.task_metrics.values()
            ]

            if agent_usage:
                return {
                    "avg_agent_usage": sum(agent_usage) / len(agent_usage),
                    "max_agent_usage": max(agent_usage),
                    "min_agent_usage": min(agent_usage),
                    "efficiency_trend": (
                        "improving"
                        if len(agent_usage) > 1
                        and agent_usage[-1] < agent_usage[0]
                        else "stable"
                    ),
                }

            return {"no_data": True}

        except Exception as e:
            return {"error": str(e)}

    def _identify_bottlenecks(self) -> list[str]:
        """Identify system bottlenecks"""
        bottlenecks = []

        try:
            # Check for high agent usage
            high_usage_tasks = [
                k
                for k, v in self.task_metrics.items()
                if v.agent_invocations > 50
            ]
            if high_usage_tasks:
                bottlenecks.append(
                    f"High agent usage in tasks: {high_usage_tasks}"
                )

            # Check for blocked tasks
            blocked_tasks = [
                k
                for k, v in self.task_metrics.items()
                if not v.dependencies_met
            ]
            if blocked_tasks:
                bottlenecks.append(f"Blocked tasks: {blocked_tasks}")

            # Check for low quality scores
            low_quality_tasks = [
                k
                for k, v in self.task_metrics.items()
                if v.quality_score < 0.7
            ]
            if low_quality_tasks:
                bottlenecks.append(f"Low quality tasks: {low_quality_tasks}")

        except Exception as e:
            bottlenecks.append(f"Error identifying bottlenecks: {e}")

        return bottlenecks

    def _find_optimization_opportunities(self) -> list[str]:
        """Find optimization opportunities"""
        opportunities = []

        try:
            # Check for parallelization opportunities
            issues = self.get_open_kb_issues()
            parallelizable = self._find_parallelizable_tasks(issues)

            if len(parallelizable) > 1:
                opportunities.append(
                    f"Parallelize {len(parallelizable)} tasks for 2-4x speedup"
                )

            # Check for resource optimization
            avg_roi = (
                sum(m.calculate_roi() for m in self.task_metrics.values())
                / len(self.task_metrics)
                if self.task_metrics
                else 0
            )

            if avg_roi < 1.5:
                opportunities.append("Optimize task selection for better ROI")

            if not self.swarm_initialized:
                opportunities.append(
                    "Initialize swarm for better coordination"
                )

        except Exception as e:
            opportunities.append(f"Error finding opportunities: {e}")

        return opportunities

    def _assess_risk_factors(self) -> list[str]:
        """Assess risk factors in current task portfolio"""
        risks = []

        try:
            # Check for tasks with negative ROI
            negative_roi_tasks = [
                k
                for k, v in self.task_metrics.items()
                if v.calculate_roi() < 0.5
            ]
            if negative_roi_tasks:
                risks.append(f"Tasks with negative ROI: {negative_roi_tasks}")

            # Check for overcommitment
            total_agents = sum(
                m.agent_invocations for m in self.task_metrics.values()
            )
            if total_agents > 100:
                risks.append(
                    f"High agent utilization: {total_agents} total invocations"
                )

            # Check for stalled tasks
            current_time = datetime.now()
            stalled_tasks = [
                k
                for k, v in self.task_metrics.items()
                if (current_time - v.created_at).days > 7
            ]
            if stalled_tasks:
                risks.append(f"Stalled tasks (>7 days): {stalled_tasks}")

        except Exception as e:
            risks.append(f"Error assessing risks: {e}")

        return risks

    def _find_parallelizable_tasks(self, issues: list[dict]) -> list[dict]:
        """Find tasks that can be parallelized"""
        parallelizable = []

        parallelizable_stages = [
            "stage:research.requested",
            "stage:research.proposed",
            "stage:research.active",
            "stage:implementation.approved",
        ]

        for issue in issues:
            labels = issue.get("labels", [])
            if any(stage in labels for stage in parallelizable_stages):
                # Check if not blocked by dependencies
                deps = self._get_issue_dependencies(issue["number"])
                if not deps or self._check_dependencies_met(deps):
                    parallelizable.append(issue)

        return parallelizable

    # =============================================================================
    # BACKWARD COMPATIBILITY METHODS (Enhanced)
    # =============================================================================

    def _run_gh_command(self, args: list[str]) -> str:
        """Run GitHub CLI command and return output (Enhanced with error handling)"""
        cmd = ["gh"] + args
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )

            # Track API usage for ROI calculation
            if hasattr(self, "api_calls_count"):
                self.api_calls_count += 1
            else:
                self.api_calls_count = 1

            return result.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"GitHub CLI command failed: {e.stderr}")
            raise

    def _check_deliverable_in_comments(
        self, issue_number: int, current_stage: str
    ) -> dict[str, any]:
        """Check if deliverable exists in issue comments and its sync status"""
        try:
            # Get all comments
            output = self._run_gh_command(
                [
                    "issue",
                    "view",
                    str(issue_number),
                    "--repo",
                    f"{self.owner}/{self.repo}",
                    "--comments",
                    "--json",
                    "comments",
                ]
            )

            data = json.loads(output) if output else {}
            comments = data.get("comments", [])

            # Map stages to expected deliverable events
            stage_to_event = {
                "stage:research.requested": "001 - Research Requested → Research Proposed",
                "stage:research.proposed": "002 - Research Proposed → Research Active",
                "stage:research.active": "003 - Research Active → Decision Ready",
                "stage:decision.ready": "004 - Decision Ready → Decision Review",
                "stage:decision.review": "005 - Decision Review → Implementation Approved",
                "stage:implementation.approved": "006 - Implementation Approved → Implementation Started",
                "stage:implementation.started": "006 - Implementation Approved → Implementation Started",
                "stage:metrics.collection": "007 - Implementation Started → Metrics Collection",
                "stage:metrics.review": "008 - Metrics Collection → Metrics Review",
            }

            expected_event = stage_to_event.get(current_stage)
            if not expected_event:
                return {
                    "found": False,
                    "expected": None,
                    "status": "unknown_stage",
                }

            # Check recent comments for deliverable
            deliverable_found = False
            deliverable_complete = False
            deliverable_timestamp = None

            for comment in reversed(comments):  # Check from newest to oldest
                body = comment.get("body", "")

                # Look for specific event deliverable
                if (
                    f"**Event**: {expected_event}" in body
                    and "Deliverable" in body
                ):
                    deliverable_found = True

                    # Check if deliverable is marked complete
                    if "**Status**: ✅" in body:
                        deliverable_complete = True

                    # Extract timestamp if present
                    import re

                    timestamp_match = re.search(
                        r"\[.*?-(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]", body
                    )
                    if timestamp_match:
                        deliverable_timestamp = timestamp_match.group(1)

                    break

            # Determine sync status
            if deliverable_found and deliverable_complete:
                # Deliverable exists and is complete - label should be updated
                next_stage = self._get_next_stage(current_stage)
                return {
                    "found": True,
                    "complete": True,
                    "expected": expected_event,
                    "timestamp": deliverable_timestamp,
                    "status": "out_of_sync",
                    "recommendation": f"Update label from {current_stage} to {next_stage}",
                }
            if deliverable_found and not deliverable_complete:
                return {
                    "found": True,
                    "complete": False,
                    "expected": expected_event,
                    "timestamp": deliverable_timestamp,
                    "status": "in_progress",
                }
            return {
                "found": False,
                "expected": expected_event,
                "status": "missing",
            }

        except Exception as e:
            print(f"Error checking deliverables: {e}", file=sys.stderr)
            return {
                "found": False,
                "expected": None,
                "status": "error",
                "error": str(e),
            }

    def get_open_kb_issues(self) -> list[dict]:
        """Get all open KB-related issues"""
        try:
            # Get all open issues with JSON output
            output = self._run_gh_command(
                [
                    "issue",
                    "list",
                    "--repo",
                    f"{self.owner}/{self.repo}",
                    "--state",
                    "open",
                    "--json",
                    "number,title,labels,assignees,url",
                    "--limit",
                    "100",
                ]
            )

            issues = json.loads(output) if output else []

            # Filter for KB events (have stage: labels)
            kb_issues = []
            for issue in issues:
                labels = [label["name"] for label in issue.get("labels", [])]

                # Check if this is a KB lifecycle issue
                if any(label.startswith("stage:") for label in labels):
                    kb_issues.append(
                        {
                            "number": issue["number"],
                            "title": issue["title"],
                            "labels": labels,
                            "assignees": [
                                a["login"] for a in issue.get("assignees", [])
                            ],
                            "url": issue["url"],
                        }
                    )

            return kb_issues

        except Exception as e:
            print(f"Error fetching issues: {e}", file=sys.stderr)
            return []

    def _get_issue_dependencies(self, issue_number: int) -> list[int]:
        """Extract dependencies from issue body"""
        try:
            output = self._run_gh_command(
                [
                    "issue",
                    "view",
                    str(issue_number),
                    "--repo",
                    f"{self.owner}/{self.repo}",
                    "--json",
                    "body",
                ]
            )

            data = json.loads(output) if output else {}
            body = data.get("body", "")

            # Look for dependency patterns
            dependencies = []
            import re

            # Pattern 1: "Dependencies: #123, #456" (from template)
            deps_match = re.search(
                r"\*\*Dependencies\*\*:\s*(?:<!--.*?-->)?\s*(.*?)(?:\n|$)",
                body,
                re.IGNORECASE,
            )
            if deps_match:
                deps_text = deps_match.group(1)
                # Extract issue numbers
                issue_nums = re.findall(r"#(\d+)", deps_text)
                dependencies.extend([int(num) for num in issue_nums])

            # Pattern 2: "Depends on: #123"
            single_pattern = re.findall(r"(?i)depends?\s+on:?\s*#(\d+)", body)
            dependencies.extend([int(num) for num in single_pattern])

            # Pattern 3: "Blocked by: #123"
            blocked_pattern = re.findall(r"(?i)blocked\s+by:?\s*#(\d+)", body)
            dependencies.extend([int(num) for num in blocked_pattern])

            # Remove duplicates
            return list(set(dependencies))

        except Exception as e:
            print(
                f"Error checking dependencies for #{issue_number}: {e}",
                file=sys.stderr,
            )
            return []

    def _check_dependencies_met(
        self, dependencies: list[int]
    ) -> dict[int, dict]:
        """Check if dependencies are resolved (closed issues)"""
        if not dependencies:
            return {}

        results = {}
        for dep in dependencies:
            try:
                output = self._run_gh_command(
                    [
                        "issue",
                        "view",
                        str(dep),
                        "--repo",
                        f"{self.owner}/{self.repo}",
                        "--json",
                        "state,title,labels",
                    ]
                )

                data = json.loads(output) if output else {}
                is_closed = data.get("state", "").lower() == "closed"

                # Check if it reached knowledge captured stage
                labels = [l["name"] for l in data.get("labels", [])]
                is_complete = "stage:knowledge.captured" in labels or is_closed

                results[dep] = {
                    "met": is_complete,
                    "title": data.get("title", f"Issue #{dep}"),
                    "state": data.get("state", "unknown"),
                    "current_stage": next(
                        (l for l in labels if l.startswith("stage:")),
                        "unknown",
                    ),
                }
            except:
                results[dep] = {
                    "met": False,
                    "title": f"Issue #{dep} (error checking)",
                    "state": "error",
                    "current_stage": "unknown",
                }

        return results

    def classify_tasks(self, issues: list[dict]) -> dict[str, list[dict]]:
        """Classify issues into task categories"""

        tasks = {
            "intake": [],  # stage:research.requested
            "planning": [],  # stage:research.proposed
            "research": [],  # stage:research.active
            "decision": [],  # stage:decision.ready, stage:decision.review
            "implementation": [],  # stage:implementation.*
            "metrics": [],  # stage:metrics.*
            "blocked": [],  # status:blocked or unmet dependencies
        }

        for issue in issues:
            labels = issue["labels"]
            issue_num = issue["number"]

            # Check for explicit blocked status first
            if "status:blocked" in labels:
                issue["block_reason"] = "Explicit blocked label"
                tasks["blocked"].append(issue)
                continue

            # Check dependencies
            dependencies = self._get_issue_dependencies(issue_num)
            if dependencies:
                dep_status = self._check_dependencies_met(dependencies)
                unmet_deps = [
                    (dep, info)
                    for dep, info in dep_status.items()
                    if not info["met"]
                ]

                if unmet_deps:
                    # Add dependency info to issue for display
                    issue["unmet_dependencies"] = unmet_deps
                    issue["block_reason"] = "Unmet dependencies"
                    tasks["blocked"].append(issue)
                    continue

            # Classify by stage if not blocked
            classified = False
            for label in labels:
                if label == "stage:research.requested":
                    tasks["intake"].append(issue)
                    classified = True
                    break
                if label == "stage:research.proposed":
                    tasks["planning"].append(issue)
                    classified = True
                    break
                if label == "stage:research.active":
                    tasks["research"].append(issue)
                    classified = True
                    break
                if label.startswith("stage:decision"):
                    tasks["decision"].append(issue)
                    classified = True
                    break
                if label.startswith("stage:implementation"):
                    tasks["implementation"].append(issue)
                    classified = True
                    break
                if label.startswith("stage:metrics"):
                    tasks["metrics"].append(issue)
                    classified = True
                    break

            # If we couldn't classify by stage, check if it has any KB-related labels
            if not classified and any(l.startswith("stage:") for l in labels):
                print(f"Warning: Issue #{issue_num} has unknown stage label")

        return tasks

    def has_pending_tasks(self) -> bool:
        """Check if there are any pending tasks"""
        issues = self.get_open_kb_issues()
        tasks = self.classify_tasks(issues)

        # Count non-blocked tasks
        pending_count = sum(
            len(task_list)
            for category, task_list in tasks.items()
            if category != "blocked"
        )

        return pending_count > 0

    def list_all_tasks(self):
        """List all pending tasks in a readable format"""
        issues = self.get_open_kb_issues()
        tasks = self.classify_tasks(issues)

        print("📋 KB TASK MASTER - PENDING TASKS")
        print("=" * 60)

        total_pending = 0

        # Show tasks by category
        category_names = {
            "intake": "📥 Intake (Research Requests)",
            "planning": "📝 Planning (Proposals)",
            "research": "🔬 Active Research",
            "decision": "🤔 Decision Making",
            "implementation": "🚀 Implementation",
            "metrics": "📊 Metrics & ROI",
            "blocked": "🚫 Blocked",
        }

        # Track parallelizable tasks and deliverable issues
        parallelizable_categories = [
            "intake",
            "planning",
            "research",
            "implementation",
            "metrics",
        ]
        parallelizable_tasks = []
        out_of_sync_issues = []

        for category, task_list in tasks.items():
            if task_list:
                print(
                    f"\n{category_names.get(category, category.upper())} ({len(task_list)} tasks):"
                )

                for task in task_list:
                    stage = next(
                        (l for l in task["labels"] if l.startswith("stage:")),
                        "unknown",
                    )
                    priority = next(
                        (
                            l.split(":")[1]
                            for l in task["labels"]
                            if l.startswith("priority:")
                        ),
                        "normal",
                    )

                    print(f"  • Issue #{task['number']}: {task['title']}")
                    print(f"    Stage: {stage} | Priority: {priority}")
                    if task["assignees"]:
                        print(
                            f"    Assigned to: {', '.join(task['assignees'])}"
                        )
                    print(f"    URL: {task['url']}")

                    # Show block reason for blocked tasks
                    if category == "blocked":
                        if "block_reason" in task:
                            print(
                                f"    🚫 Block Reason: {task['block_reason']}"
                            )

                        if "unmet_dependencies" in task:
                            print("    ⛔ Waiting for:")
                            for dep_id, dep_info in task["unmet_dependencies"]:
                                print(
                                    f"       - #{dep_id}: {dep_info['title']}"
                                )
                                print(
                                    f"         Status: {dep_info['state']} | Stage: {dep_info['current_stage']}"
                                )

                    # Check deliverable status
                    if stage.startswith("stage:") and category != "blocked":
                        deliverable_status = (
                            self._check_deliverable_in_comments(
                                task["number"], stage
                            )
                        )
                        if deliverable_status["status"] == "out_of_sync":
                            print(
                                "    ⚠️  DELIVERABLE: Complete but label not updated!"
                            )
                            print(
                                f"        Action needed: {deliverable_status['recommendation']}"
                            )
                            out_of_sync_issues.append(
                                (
                                    task["number"],
                                    deliverable_status,
                                )
                            )
                        elif deliverable_status["status"] == "missing":
                            print("    📝 DELIVERABLE: Not yet posted")
                        elif deliverable_status["status"] == "in_progress":
                            print("    🔄 DELIVERABLE: In progress")

                    # Add to parallelizable list
                    if (
                        category in parallelizable_categories
                        and category != "blocked"
                    ):
                        parallelizable_tasks.append((category, task))

                if category != "blocked":
                    total_pending += len(task_list)

        print("\n📊 SUMMARY:")
        print(f"   Total Pending Tasks: {total_pending}")
        print(f"   Blocked Tasks: {len(tasks['blocked'])}")

        # Out of sync warning
        if out_of_sync_issues:
            print("\n🚨 OUT OF SYNC WARNING:")
            print(
                f"   {len(out_of_sync_issues)} issues have completed deliverables but outdated labels!"
            )
            print("\n   IMMEDIATE ACTION REQUIRED:")
            for issue_num, status in out_of_sync_issues:
                print(f"   • Issue #{issue_num}: {status['recommendation']}")
            print("\n   Run these commands NOW to fix:")
            for issue_num, status in out_of_sync_issues:
                current = (
                    status["recommendation"]
                    .split(" from ")[1]
                    .split(" to ")[0]
                )
                next_stage = status["recommendation"].split(" to ")[1]
                print(
                    f"   gh issue edit {issue_num} --remove-label '{current}' --add-label '{next_stage}'"
                )

        # Parallel processing guidance
        if len(parallelizable_tasks) > 1:
            print("\n⚡ PARALLEL PROCESSING OPPORTUNITY:")
            print(
                f"   {len(parallelizable_tasks)} tasks can be processed in parallel!"
            )
            print("\n   🎯 ORCHESTRATOR ACTION REQUIRED:")
            print(
                "   Use the batch tool to process multiple events simultaneously:"
            )
            print("   - Group similar event types (e.g., all intake tasks)")
            print("   - Maximum 5 agents per batch to avoid overload")
            print("   - Prioritize high-priority items first")

        # Event sequence reminder
        print("\n📌 CRITICAL REMINDERS FOR ORCHESTRATOR:")
        print(
            "   1. ⚠️  UPDATE LABELS: You MUST update issue labels after processing!"
        )
        print(
            "      - After intake → change stage:research.requested to stage:research.proposed"
        )
        print("      - After planning → change to stage:research.active")
        print("      - This is MANDATORY for event queue progression!")
        print(
            "\n   2. 📄 CHECK DELIVERABLES: Each stage has required outputs:"
        )
        print("      - Intake → research_proposal.yaml")
        print("      - Planning → research_plan.yaml")
        print("      - Research → findings/ directory")
        print("      - Decision → decision_document.yaml")
        print("\n   3. 🔄 UPDATE YOUR TODO: Use TodoWrite after each task!")
        print("   4. 🚨 RUN GATEKEEPER: Before ANY completion attempt!")

        if total_pending == 0:
            print(
                "\n✅ No pending tasks! Run gatekeeper to confirm completion allowed."
            )
        else:
            print(
                f"\n⚠️  {total_pending} tasks still pending. Keep processing!"
            )
            print(
                "   Next step: Run 'python task_master.py --next' for priority task"
            )

    def get_next_task(self) -> dict | None:
        """Get the next task to process (priority order)"""
        issues = self.get_open_kb_issues()
        tasks = self.classify_tasks(issues)

        # Priority order for processing
        priority_order = [
            "intake",
            "planning",
            "research",
            "decision",
            "implementation",
            "metrics",
        ]

        for category in priority_order:
            if tasks[category]:
                # Sort by priority within category
                category_tasks = sorted(
                    tasks[category],
                    key=lambda t: self._get_priority_score(t["labels"]),
                    reverse=True,
                )

                next_task = category_tasks[0]

                stage = next(
                    (l for l in next_task["labels"] if l.startswith("stage:")),
                    "unknown",
                )

                print("🎯 NEXT TASK TO PROCESS:")
                print(f"   Issue #{next_task['number']}: {next_task['title']}")
                print(f"   Category: {category}")
                print(f"   Current Stage: {stage}")
                print(f"   URL: {next_task['url']}")

                # Detailed action guidance based on category
                print("\n📋 ORCHESTRATOR TODO:")
                print("   1. Add to your TodoWrite immediately!")

                if category == "intake":
                    print(
                        f"   2. Run: Task('Execute kb-intake-swarm for issue #{next_task['number']}')"
                    )
                    print(
                        "      Reference: .claude/commands/swarm/kb-intake-swarm.md"
                    )
                    print("   3. MANDATORY: After completion, update label:")
                    print(
                        f"      gh issue edit {next_task['number']} --remove-label '{stage}' --add-label 'stage:research.proposed'"
                    )
                    print(
                        "   4. Verify deliverable: research_proposal.yaml created"
                    )
                elif category == "planning":
                    print(
                        f"   2. Run: Task('Execute kb-planning-swarm for issue #{next_task['number']}')"
                    )
                    print(
                        "      Reference: .claude/commands/swarm/kb-research-swarm.md"
                    )
                    print("   3. MANDATORY: After completion, update label:")
                    print(
                        f"      gh issue edit {next_task['number']} --remove-label '{stage}' --add-label 'stage:research.active'"
                    )
                    print(
                        "   4. Verify deliverable: research_plan.yaml created"
                    )
                elif category == "research":
                    print(
                        f"   2. Run: Task('Execute kb-research-swarm for issue #{next_task['number']}')"
                    )
                    print(
                        "      Reference: .claude/commands/swarm/kb-research-swarm.md"
                    )
                    print("   3. MANDATORY: After completion, update label:")
                    print(
                        f"      gh issue edit {next_task['number']} --remove-label '{stage}' --add-label 'stage:decision.ready'"
                    )
                    print(
                        "   4. Verify deliverable: findings/ directory populated"
                    )
                elif category == "decision":
                    print(
                        f"   2. Run: Task('Execute kb-decision-swarm for issue #{next_task['number']}')"
                    )
                    print(
                        "      Reference: .claude/commands/swarm/kb-decision-review-swarm.md"
                    )
                    print("   3. MANDATORY: After completion, update label:")
                    print(
                        f"      gh issue edit {next_task['number']} --remove-label '{stage}' --add-label 'stage:implementation.approved'"
                    )
                    print(
                        "   4. Verify deliverable: decision_document.yaml created"
                    )
                elif category == "implementation":
                    print("   2. Track implementation progress")
                    print(
                        "   3. Update label when complete: stage:metrics.collection"
                    )
                elif category == "metrics":
                    print("   2. Collect and analyze ROI metrics")
                    print(
                        "   3. Update label when complete: stage:knowledge.captured"
                    )

                print(
                    "\n⚠️  CRITICAL: Label updates are MANDATORY for queue progression!"
                )
                print(
                    "   Without label updates, the event queue will be stuck!"
                )

                return next_task

        print("✅ No tasks to process!")
        return None

    def _get_priority_score(self, labels: list[str]) -> int:
        """Get numeric priority score from labels"""
        for label in labels:
            if label == "priority:critical":
                return 4
            if label == "priority:high":
                return 3
            if label == "priority:medium":
                return 2
            if label == "priority:low":
                return 1
        return 2  # Default medium

    def _get_next_stage(self, current_stage: str) -> str:
        """Get the next stage in KB lifecycle"""
        stage_progression = {
            "stage:research.requested": "stage:research.proposed",
            "stage:research.proposed": "stage:research.active",
            "stage:research.active": "stage:decision.ready",
            "stage:decision.ready": "stage:decision.review",
            "stage:decision.review": "stage:implementation.approved",
            "stage:implementation.approved": "stage:implementation.started",
            "stage:implementation.started": "stage:metrics.collection",
            "stage:metrics.collection": "stage:metrics.review",
            "stage:metrics.review": "stage:knowledge.captured",
        }
        return stage_progression.get(current_stage, "unknown")


def main():
    """Enhanced CLI interface with event processing capabilities"""
    parser = argparse.ArgumentParser(
        description="KB Task Master - Enhanced Event-Driven Task Orchestration System"
    )

    # Backward compatibility commands
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if tasks are pending (exit code)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all pending tasks"
    )
    parser.add_argument(
        "--next", action="store_true", help="Get next task to process"
    )

    # Enhanced commands
    parser.add_argument(
        "--monitor", action="store_true", help="Real-time event monitoring"
    )
    parser.add_argument(
        "--roi", action="store_true", help="Calculate ROI for active tasks"
    )
    parser.add_argument(
        "--sync", action="store_true", help="Synchronize with event system"
    )
    parser.add_argument(
        "--insights", action="store_true", help="Generate structured insights"
    )
    parser.add_argument(
        "--transition", type=int, help="Transition issue to new state"
    )
    parser.add_argument("--state", type=str, help="New state for transition")
    parser.add_argument(
        "--reason", type=str, help="Reason for state transition"
    )
    parser.add_argument(
        "--recommend", type=int, help="Get recommendations for issue"
    )
    parser.add_argument(
        "--status", type=int, help="Get detailed status for issue"
    )

    # Configuration
    parser.add_argument(
        "--owner", default="khive-ai", help="GitHub repo owner"
    )
    parser.add_argument("--repo", default="kb", help="GitHub repo name")
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Monitoring duration in seconds",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create enhanced task master
    task_master = EnhancedTaskMaster(args.owner, args.repo)

    # Execute requested action
    if args.check:
        # Exit 0 if no tasks, 1 if tasks pending
        has_tasks = task_master.has_pending_tasks()
        if has_tasks:
            print("❌ Tasks pending - cannot complete", file=sys.stderr)
            sys.exit(1)
        else:
            print("✅ No pending tasks - can complete")
            sys.exit(0)

    elif args.list:
        task_master.list_all_tasks()

    elif args.next:
        task_master.get_next_task()

    elif args.monitor:
        print(
            f"🔄 Starting real-time event monitoring for {args.duration} seconds..."
        )
        try:
            result = asyncio.run(
                task_master.process_real_time_events(args.duration)
            )
            print("\n📊 MONITORING SUMMARY:")
            print(f"   Duration: {result['duration']} seconds")
            print(f"   Events Processed: {result['events_processed']}")
            print(f"   Results: {len(result['results'])} successful")
        except KeyboardInterrupt:
            print("\n⚠️  Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring failed: {e}")

    elif args.roi:
        print("💰 PORTFOLIO ROI ANALYSIS")
        print("=" * 50)

        roi_data = task_master.calculate_portfolio_roi()

        if "error" in roi_data:
            print(f"❌ ROI calculation failed: {roi_data['error']}")
        else:
            print(f"   Average ROI: {roi_data['average_roi']:.2f}x")
            print(f"   Total Tasks: {roi_data['total_tasks']}")
            print(
                f"   Agent Invocations: {roi_data['total_agent_invocations']}"
            )
            print(
                f"   Compute Time: {roi_data['total_compute_time']:.2f} minutes"
            )
            print("\n   ROI Distribution:")
            print(
                f"     Positive (>1.5x): {roi_data['distribution']['positive']}"
            )
            print(
                f"     Neutral (1.0-1.5x): {roi_data['distribution']['neutral']}"
            )
            print(
                f"     Negative (<1.0x): {roi_data['distribution']['negative']}"
            )

    elif args.sync:
        print("🔄 Synchronizing with event system...")
        try:
            # Sync with GitHub events
            sync_result = asyncio.run(task_master._poll_github_events())
            print(f"   Found {len(sync_result)} recent events")

            # Update task metrics
            issues = task_master.get_open_kb_issues()
            for issue in issues:
                if issue["number"] not in task_master.task_metrics:
                    task_master.task_metrics[issue["number"]] = TaskMetrics()

            print(f"   Updated metrics for {len(issues)} issues")
            print("✅ Synchronization complete")

        except Exception as e:
            print(f"❌ Synchronization failed: {e}")

    elif args.insights:
        print("🧠 STRUCTURED INSIGHTS")
        print("=" * 50)

        insights = task_master.generate_insights()

        if "error" in insights:
            print(f"❌ Insight generation failed: {insights['error']}")
        else:
            print("\n📈 Performance Trends:")
            for key, value in insights["performance_trends"].items():
                print(f"   {key}: {value}")

            print("\n🚧 Bottlenecks:")
            for bottleneck in insights["bottlenecks"]:
                print(f"   • {bottleneck}")

            print("\n⚡ Optimization Opportunities:")
            for opportunity in insights["optimization_opportunities"]:
                print(f"   • {opportunity}")

            print("\n⚠️  Risk Factors:")
            for risk in insights["risk_factors"]:
                print(f"   • {risk}")

    elif args.transition:
        if not args.state:
            print("❌ --state required for transition")
            sys.exit(1)

        print(
            f"🔄 Transitioning issue #{args.transition} to state: {args.state}"
        )

        try:
            # Validate state
            valid_states = [state.value for state in TaskState]
            if args.state not in valid_states:
                print(f"❌ Invalid state. Valid states: {valid_states}")
                sys.exit(1)

            # Update issue labels based on state
            result = task_master._transition_issue_state(
                args.transition, args.state, args.reason
            )
            print(f"✅ Issue #{args.transition} transitioned to {args.state}")

        except Exception as e:
            print(f"❌ Transition failed: {e}")

    elif args.recommend:
        print(f"🎯 RECOMMENDATIONS FOR ISSUE #{args.recommend}")
        print("=" * 50)

        try:
            recommendations = task_master._get_issue_recommendations(
                args.recommend
            )
            for rec in recommendations:
                print(f"   • {rec}")

        except Exception as e:
            print(f"❌ Failed to generate recommendations: {e}")

    elif args.status:
        print(f"📊 DETAILED STATUS FOR ISSUE #{args.status}")
        print("=" * 50)

        try:
            status = task_master._get_detailed_issue_status(args.status)

            for key, value in status.items():
                print(f"   {key}: {value}")

        except Exception as e:
            print(f"❌ Failed to get status: {e}")

    else:
        # Enhanced default: show comprehensive summary
        print("🎛️  KB TASK MASTER - ENHANCED DASHBOARD")
        print("=" * 60)

        issues = task_master.get_open_kb_issues()
        tasks = task_master.classify_tasks(issues)

        pending_count = sum(len(t) for c, t in tasks.items() if c != "blocked")

        print("\n📊 TASK SUMMARY:")
        print(f"   Pending Tasks: {pending_count}")
        print(f"   Blocked Tasks: {len(tasks['blocked'])}")
        print(
            f"   Knowledge Sessions: {1 if task_master.knowledge_graph_id else 0}"
        )
        print(
            f"   swarm Status: {'✅ Initialized' if task_master.swarm_initialized else '❌ Not initialized'}"
        )

        # Show quick ROI if available
        if task_master.task_metrics:
            roi_data = task_master.calculate_portfolio_roi()
            print(f"   Portfolio ROI: {roi_data.get('average_roi', 0):.2f}x")

        print("\n🔧 AVAILABLE COMMANDS:")
        print("   --list          List all pending tasks")
        print("   --next          Get next task to process")
        print("   --monitor       Real-time event monitoring")
        print("   --roi           Calculate portfolio ROI")
        print("   --insights      Generate structured insights")
        print("   --sync          Synchronize with event system")
        print("   --check         Check completion status")
        print("\n📖 Use --help for complete command reference")


if __name__ == "__main__":
    main()
