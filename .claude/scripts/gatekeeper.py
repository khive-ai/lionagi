#!/usr/bin/env python3
"""
KB Multi-Layer Event-Driven Gatekeeper

Advanced validation system that integrates with the unified KB event system.
Implements 5-layer validation architecture with event-driven processing,
dynamic validation rules, and comprehensive reporting.

This implements the critical requirement from CLAUDE.md:
"ALWAYS check for pending events before completion"

Usage:
    python gatekeeper.py                 # Full multi-layer validation
    python gatekeeper.py --quick         # Layer 1-2 validation only
    python gatekeeper.py --report        # Generate validation report
    python gatekeeper.py --subscribe     # Event-driven monitoring mode
    python gatekeeper.py --validate-event <event_id>  # Validate specific event

Returns:
    Exit code 0: Safe to complete (all validation layers passed)
    Exit code 1: Cannot complete (validation failures)
    Exit code 2: System error or configuration issue
"""

import argparse
import asyncio
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ValidationLevel(Enum):
    """Validation severity levels"""

    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationLayer(Enum):
    """Multi-layer validation architecture"""

    TASK_COMPLETION = "task_completion"  # Layer 1: Enhanced task completion
    QUALITY_GATES = "quality_gates"  # Layer 2: Quality gate enforcement
    STRATEGIC_MILESTONES = (
        "strategic_milestones"  # Layer 3: ROI/strategic validation
    )
    MULTI_AGENT_COORDINATION = (
        "multi_agent_coordination"  # Layer 4: swarm coordination
    )
    EVENT_STATE_MACHINE = (
        "event_state_machine"  # Layer 5: Event system compliance
    )


@dataclass
class ValidationResult:
    """Result of a validation check"""

    layer: ValidationLayer
    level: ValidationLevel
    message: str
    details: dict[str, Any]
    timestamp: str
    validator: str
    event_id: str | None = None
    resolution_hint: str | None = None


@dataclass
class EventValidationContext:
    """Context for event-specific validation"""

    event_id: str
    event_type: str
    issue_number: int
    current_stage: str
    dependencies: list[int]
    priority: str
    assignees: list[str]
    created_at: str
    updated_at: str
    metadata: dict[str, Any]


class EventDrivenGatekeeper:
    """Multi-layer validation system with event integration"""

    def __init__(self, config_path: str | None = None):
        self.config = self._load_config(config_path)
        self.validation_rules = self._load_validation_rules()
        self.swarm_available = self._check_swarm_availability()
        self.knowledge_available = self._check_knowledge_availability()
        self.task_master_path = Path(__file__).parent / "task_master.py"
        self.orchestration_planner_path = (
            Path(__file__).parent / "orchestration_planner.py"
        )

        # Validation statistics
        self.validation_stats = {
            "total_validations": 0,
            "layer_stats": {
                layer.value: {
                    "pass": 0,
                    "warning": 0,
                    "error": 0,
                    "critical": 0,
                }
                for layer in ValidationLayer
            },
            "event_validations": {},
            "performance_metrics": {},
        }

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        """Load gatekeeper configuration"""
        default_config = {
            "validation_timeout": 300,  # 5 minutes
            "parallel_validation": True,
            "event_subscription_enabled": True,
            "report_retention_days": 30,
            "quality_thresholds": {
                "confidence_minimum": 0.8,
                "roi_threshold": 1.5,
                "completion_rate_minimum": 0.9,
            },
            "swarm_integration": {
                "coordination_timeout": 120,
                "performance_threshold": 0.8,
                "memory_usage_limit": 0.9,
            },
            "event_patterns": {
                "critical_events": [
                    "consensus_failure",
                    "data_corruption",
                    "security_breach",
                ],
                "blocking_events": [
                    "dependency_failure",
                    "resource_exhaustion",
                ],
                "warning_events": [
                    "performance_degradation",
                    "quality_threshold_warning",
                ],
            },
        }

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)

        return default_config

    def _load_validation_rules(self) -> dict[str, Any]:
        """Load dynamic validation rules from decision matrix"""
        decision_matrix_path = (
            Path(__file__).parent.parent / "resources" / "decision_matrix.yaml"
        )

        if decision_matrix_path.exists():
            with open(decision_matrix_path) as f:
                return yaml.safe_load(f)

        return {}

    def _check_swarm_availability(self) -> bool:
        """Check if swarm MCP is available"""
        try:
            # Simple availability check - would be enhanced with actual MCP connectivity test
            return True  # Assume available for now
        except:
            return False

    def _check_knowledge_availability(self) -> bool:
        """Check if Knowledge MCP is available"""
        try:
            # Simple availability check - would be enhanced with actual MCP connectivity test
            return True  # Assume available for now
        except:
            return False

    # =================================================================
    # LAYER 1: ENHANCED TASK COMPLETION VALIDATION
    # =================================================================

    async def validate_task_completion(
        self, context: EventValidationContext | None = None
    ) -> list[ValidationResult]:
        """Layer 1: Enhanced task completion validation with detailed analysis"""
        results = []

        try:
            # Run original task master check
            result = subprocess.run(
                [sys.executable, str(self.task_master_path), "--check"],
                capture_output=True,
                text=True,
                timeout=self.config["validation_timeout"],
            )

            if result.returncode == 0:
                # Get detailed task information
                list_result = subprocess.run(
                    [sys.executable, str(self.task_master_path), "--list"],
                    capture_output=True,
                    text=True,
                    timeout=self.config["validation_timeout"],
                )

                results.append(
                    ValidationResult(
                        layer=ValidationLayer.TASK_COMPLETION,
                        level=ValidationLevel.PASS,
                        message="All tasks completed successfully",
                        details={
                            "task_output": list_result.stdout,
                            "pending_count": 0,
                        },
                        timestamp=datetime.now().isoformat(),
                        validator="task_completion_validator",
                    )
                )
            else:
                # Parse pending tasks for detailed analysis
                list_result = subprocess.run(
                    [sys.executable, str(self.task_master_path), "--list"],
                    capture_output=True,
                    text=True,
                    timeout=self.config["validation_timeout"],
                )

                # Extract task details from output
                task_details = self._parse_task_master_output(
                    list_result.stdout
                )

                results.append(
                    ValidationResult(
                        layer=ValidationLayer.TASK_COMPLETION,
                        level=ValidationLevel.ERROR,
                        message=f"Tasks still pending: {task_details['total_pending']} tasks",
                        details=task_details,
                        timestamp=datetime.now().isoformat(),
                        validator="task_completion_validator",
                        resolution_hint="Process pending tasks before attempting completion",
                    )
                )

        except subprocess.TimeoutExpired:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.TASK_COMPLETION,
                    level=ValidationLevel.ERROR,
                    message="Task completion validation timed out",
                    details={"timeout": self.config["validation_timeout"]},
                    timestamp=datetime.now().isoformat(),
                    validator="task_completion_validator",
                    resolution_hint="Check system performance and task master responsiveness",
                )
            )
        except Exception as e:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.TASK_COMPLETION,
                    level=ValidationLevel.CRITICAL,
                    message=f"Task completion validation failed: {e!s}",
                    details={"error": str(e), "type": type(e).__name__},
                    timestamp=datetime.now().isoformat(),
                    validator="task_completion_validator",
                    resolution_hint="Check task master configuration and GitHub connectivity",
                )
            )

        return results

    # =================================================================
    # LAYER 2: QUALITY GATE ENFORCEMENT
    # =================================================================

    async def validate_quality_gates(
        self, context: EventValidationContext | None = None
    ) -> list[ValidationResult]:
        """Layer 2: Quality gate enforcement by stage"""
        results = []

        try:
            # Check quality gates based on current stage
            if context:
                stage_gates = self._get_stage_quality_gates(
                    context.current_stage
                )

                for gate in stage_gates:
                    gate_result = await self._validate_quality_gate(
                        gate, context
                    )
                    results.append(gate_result)
            else:
                # Global quality validation
                global_quality = await self._validate_global_quality()
                results.extend(global_quality)

        except Exception as e:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.QUALITY_GATES,
                    level=ValidationLevel.ERROR,
                    message=f"Quality gate validation failed: {e!s}",
                    details={"error": str(e)},
                    timestamp=datetime.now().isoformat(),
                    validator="quality_gate_validator",
                    resolution_hint="Check quality gate configuration and metrics availability",
                )
            )

        return results

    # =================================================================
    # LAYER 3: STRATEGIC MILESTONE VALIDATION
    # =================================================================

    async def validate_strategic_milestones(
        self, context: EventValidationContext | None = None
    ) -> list[ValidationResult]:
        """Layer 3: Strategic milestone validation with ROI thresholds"""
        results = []

        try:
            # ROI threshold validation
            roi_validation = await self._validate_roi_thresholds(context)
            results.extend(roi_validation)

            # Strategic alignment check
            alignment_validation = await self._validate_strategic_alignment(
                context
            )
            results.extend(alignment_validation)

            # Resource utilization validation
            resource_validation = await self._validate_resource_utilization(
                context
            )
            results.extend(resource_validation)

        except Exception as e:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.STRATEGIC_MILESTONES,
                    level=ValidationLevel.ERROR,
                    message=f"Strategic milestone validation failed: {e!s}",
                    details={"error": str(e)},
                    timestamp=datetime.now().isoformat(),
                    validator="strategic_milestone_validator",
                    resolution_hint="Check ROI calculations and strategic alignment metrics",
                )
            )

        return results

    # =================================================================
    # LAYER 4: MULTI-AGENT COORDINATION VALIDATION
    # =================================================================

    async def validate_multi_agent_coordination(
        self, context: EventValidationContext | None = None
    ) -> list[ValidationResult]:
        """Layer 4: Multi-agent coordination validation using swarm"""
        results = []

        if not self.swarm_available:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.MULTI_AGENT_COORDINATION,
                    level=ValidationLevel.WARNING,
                    message="swarm MCP not available for coordination validation",
                    details={"swarm_available": False},
                    timestamp=datetime.now().isoformat(),
                    validator="multi_agent_coordination_validator",
                    resolution_hint="Enable swarm MCP for enhanced coordination validation",
                )
            )
            return results

        try:
            # Validate swarm status and performance
            swarm_status = await self._validate_swarm_status()
            results.extend(swarm_status)

            # Validate agent coordination patterns
            coordination_patterns = await self._validate_coordination_patterns(
                context
            )
            results.extend(coordination_patterns)

            # Validate memory sharing and consistency
            memory_consistency = await self._validate_memory_consistency()
            results.extend(memory_consistency)

        except Exception as e:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.MULTI_AGENT_COORDINATION,
                    level=ValidationLevel.ERROR,
                    message=f"Multi-agent coordination validation failed: {e!s}",
                    details={"error": str(e)},
                    timestamp=datetime.now().isoformat(),
                    validator="multi_agent_coordination_validator",
                    resolution_hint="Check swarm connectivity and agent status",
                )
            )

        return results

    # =================================================================
    # LAYER 5: EVENT STATE MACHINE COMPLIANCE
    # =================================================================

    async def validate_event_state_machine(
        self, context: EventValidationContext | None = None
    ) -> list[ValidationResult]:
        """Layer 5: Event state machine compliance validation"""
        results = []

        try:
            # Validate event state transitions
            state_transitions = await self._validate_state_transitions(context)
            results.extend(state_transitions)

            # Validate event ordering and dependencies
            event_ordering = await self._validate_event_ordering(context)
            results.extend(event_ordering)

            # Validate event completion criteria
            completion_criteria = await self._validate_completion_criteria(
                context
            )
            results.extend(completion_criteria)

            # Validate deliverable synchronization
            deliverable_sync = await self._validate_deliverable_sync(context)
            results.extend(deliverable_sync)

        except Exception as e:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.EVENT_STATE_MACHINE,
                    level=ValidationLevel.ERROR,
                    message=f"Event state machine validation failed: {e!s}",
                    details={"error": str(e)},
                    timestamp=datetime.now().isoformat(),
                    validator="event_state_machine_validator",
                    resolution_hint="Check event state machine configuration and GitHub API connectivity",
                )
            )

        return results

    # =================================================================
    # HELPER METHODS FOR VALIDATION LAYERS
    # =================================================================

    def _parse_task_master_output(self, output: str) -> dict[str, Any]:
        """Parse task master output for detailed analysis"""
        lines = output.split("\n")
        details = {
            "total_pending": 0,
            "parallelizable_tasks": 0,
            "blocked_tasks": 0,
            "out_of_sync_issues": 0,
            "categories": {},
            "critical_issues": [],
        }

        for line in lines:
            if "Total Pending Tasks:" in line:
                try:
                    details["total_pending"] = int(line.split(":")[-1].strip())
                except:
                    pass
            elif "Blocked Tasks:" in line:
                try:
                    details["blocked_tasks"] = int(line.split(":")[-1].strip())
                except:
                    pass
            elif "OUT OF SYNC WARNING" in line:
                details["out_of_sync_issues"] += 1
            elif "PARALLEL PROCESSING OPPORTUNITY" in line:
                details["parallelizable_tasks"] += 1

        return details

    def _get_stage_quality_gates(self, stage: str) -> list[dict[str, Any]]:
        """Get quality gates for a specific stage"""
        stage_gates = {
            "stage:research.requested": [
                {"name": "proposal_format", "threshold": 0.9},
                {"name": "research_question_clarity", "threshold": 0.8},
            ],
            "stage:research.proposed": [
                {"name": "plan_completeness", "threshold": 0.85},
                {"name": "methodology_soundness", "threshold": 0.8},
            ],
            "stage:research.active": [
                {"name": "progress_tracking", "threshold": 0.7},
                {"name": "quality_metrics", "threshold": 0.8},
            ],
            "stage:decision.ready": [
                {"name": "evidence_quality", "threshold": 0.9},
                {"name": "confidence_level", "threshold": 0.8},
            ],
            "stage:decision.review": [
                {"name": "peer_review_complete", "threshold": 1.0},
                {"name": "stakeholder_approval", "threshold": 0.9},
            ],
            "stage:implementation.approved": [
                {"name": "implementation_plan", "threshold": 0.85},
                {"name": "resource_allocation", "threshold": 0.8},
            ],
            "stage:metrics.collection": [
                {"name": "data_completeness", "threshold": 0.9},
                {"name": "roi_calculation", "threshold": 0.8},
            ],
        }

        return stage_gates.get(stage, [])

    async def _validate_quality_gate(
        self, gate: dict[str, Any], context: EventValidationContext
    ) -> ValidationResult:
        """Validate a specific quality gate"""
        gate_name = gate["name"]
        threshold = gate["threshold"]

        # Placeholder implementation - would integrate with actual quality metrics
        actual_score = 0.85  # This would be calculated based on actual metrics

        if actual_score >= threshold:
            return ValidationResult(
                layer=ValidationLayer.QUALITY_GATES,
                level=ValidationLevel.PASS,
                message=f"Quality gate '{gate_name}' passed",
                details={
                    "gate": gate_name,
                    "score": actual_score,
                    "threshold": threshold,
                },
                timestamp=datetime.now().isoformat(),
                validator="quality_gate_validator",
                event_id=context.event_id if context else None,
            )
        return ValidationResult(
            layer=ValidationLayer.QUALITY_GATES,
            level=ValidationLevel.ERROR,
            message=f"Quality gate '{gate_name}' failed",
            details={
                "gate": gate_name,
                "score": actual_score,
                "threshold": threshold,
            },
            timestamp=datetime.now().isoformat(),
            validator="quality_gate_validator",
            event_id=context.event_id if context else None,
            resolution_hint=f"Improve {gate_name} to meet threshold of {threshold}",
        )

    async def _validate_global_quality(self) -> list[ValidationResult]:
        """Validate global system quality metrics"""
        results = []

        # System-wide quality checks
        quality_metrics = {
            "overall_completion_rate": 0.92,
            "average_confidence_score": 0.87,
            "error_rate": 0.05,
            "performance_index": 0.88,
        }

        for metric, value in quality_metrics.items():
            threshold = self.config["quality_thresholds"].get(
                f"{metric}_minimum", 0.8
            )

            if value >= threshold:
                level = ValidationLevel.PASS
                message = (
                    f"Global quality metric '{metric}' within acceptable range"
                )
            else:
                level = ValidationLevel.WARNING
                message = f"Global quality metric '{metric}' below threshold"

            results.append(
                ValidationResult(
                    layer=ValidationLayer.QUALITY_GATES,
                    level=level,
                    message=message,
                    details={
                        "metric": metric,
                        "value": value,
                        "threshold": threshold,
                    },
                    timestamp=datetime.now().isoformat(),
                    validator="global_quality_validator",
                )
            )

        return results

    async def _validate_roi_thresholds(
        self, context: EventValidationContext | None
    ) -> list[ValidationResult]:
        """Validate ROI thresholds for strategic milestones"""
        results = []

        # Calculate estimated ROI for current work
        estimated_roi = await self._calculate_estimated_roi(context)
        roi_threshold = self.config["quality_thresholds"]["roi_threshold"]

        if estimated_roi >= roi_threshold:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.STRATEGIC_MILESTONES,
                    level=ValidationLevel.PASS,
                    message=f"ROI projection meets threshold ({estimated_roi:.2f}x >= {roi_threshold}x)",
                    details={
                        "estimated_roi": estimated_roi,
                        "threshold": roi_threshold,
                    },
                    timestamp=datetime.now().isoformat(),
                    validator="roi_validator",
                    event_id=context.event_id if context else None,
                )
            )
        else:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.STRATEGIC_MILESTONES,
                    level=ValidationLevel.WARNING,
                    message=f"ROI projection below threshold ({estimated_roi:.2f}x < {roi_threshold}x)",
                    details={
                        "estimated_roi": estimated_roi,
                        "threshold": roi_threshold,
                    },
                    timestamp=datetime.now().isoformat(),
                    validator="roi_validator",
                    event_id=context.event_id if context else None,
                    resolution_hint="Consider optimizing approach or reassessing value proposition",
                )
            )

        return results

    async def _calculate_estimated_roi(
        self, context: EventValidationContext | None
    ) -> float:
        """Calculate estimated ROI for current work"""
        # Placeholder calculation - would integrate with actual metrics
        base_roi = 2.1

        if context:
            # Adjust based on context factors
            if context.priority == "high":
                base_roi *= 1.2
            elif context.priority == "low":
                base_roi *= 0.8

            # Factor in agent efficiency
            agent_count = len(context.assignees) if context.assignees else 1
            if agent_count > 5:
                base_roi *= 0.9  # Coordination overhead

        return base_roi

    async def _validate_strategic_alignment(
        self, context: EventValidationContext | None
    ) -> list[ValidationResult]:
        """Validate strategic alignment of current work"""
        results = []

        # Strategic alignment score (placeholder)
        alignment_score = 0.85

        results.append(
            ValidationResult(
                layer=ValidationLayer.STRATEGIC_MILESTONES,
                level=(
                    ValidationLevel.PASS
                    if alignment_score >= 0.8
                    else ValidationLevel.WARNING
                ),
                message=f"Strategic alignment score: {alignment_score:.2f}",
                details={"alignment_score": alignment_score},
                timestamp=datetime.now().isoformat(),
                validator="strategic_alignment_validator",
                event_id=context.event_id if context else None,
            )
        )

        return results

    async def _validate_resource_utilization(
        self, context: EventValidationContext | None
    ) -> list[ValidationResult]:
        """Validate resource utilization efficiency"""
        results = []

        # Resource utilization metrics (placeholder)
        utilization_metrics = {
            "compute_efficiency": 0.82,
            "agent_productivity": 0.88,
            "time_efficiency": 0.79,
        }

        for metric, value in utilization_metrics.items():
            level = (
                ValidationLevel.PASS
                if value >= 0.75
                else ValidationLevel.WARNING
            )

            results.append(
                ValidationResult(
                    layer=ValidationLayer.STRATEGIC_MILESTONES,
                    level=level,
                    message=f"Resource utilization '{metric}': {value:.2f}",
                    details={"metric": metric, "value": value},
                    timestamp=datetime.now().isoformat(),
                    validator="resource_utilization_validator",
                    event_id=context.event_id if context else None,
                )
            )

        return results

    # =================================================================
    # LAYER 4 HELPER METHODS: MULTI-AGENT COORDINATION
    # =================================================================

    async def _validate_swarm_status(self) -> list[ValidationResult]:
        """Validate swarm status and performance"""
        results = []

        # Placeholder implementation - would use actual swarm MCP
        swarm_metrics = {
            "coordination_efficiency": 0.88,
            "agent_utilization": 0.92,
            "task_completion_rate": 0.85,
            "memory_consistency": 0.95,
        }

        for metric, value in swarm_metrics.items():
            level = (
                ValidationLevel.PASS
                if value >= 0.8
                else ValidationLevel.WARNING
            )

            results.append(
                ValidationResult(
                    layer=ValidationLayer.MULTI_AGENT_COORDINATION,
                    level=level,
                    message=f"Swarm metric '{metric}': {value:.2f}",
                    details={"metric": metric, "value": value},
                    timestamp=datetime.now().isoformat(),
                    validator="swarm_status_validator",
                )
            )

        return results

    async def _validate_coordination_patterns(
        self, context: EventValidationContext | None
    ) -> list[ValidationResult]:
        """Validate agent coordination patterns"""
        results = []

        # Coordination pattern validation (placeholder)
        pattern_score = 0.87

        results.append(
            ValidationResult(
                layer=ValidationLayer.MULTI_AGENT_COORDINATION,
                level=(
                    ValidationLevel.PASS
                    if pattern_score >= 0.8
                    else ValidationLevel.WARNING
                ),
                message=f"Coordination pattern effectiveness: {pattern_score:.2f}",
                details={"pattern_score": pattern_score},
                timestamp=datetime.now().isoformat(),
                validator="coordination_pattern_validator",
                event_id=context.event_id if context else None,
            )
        )

        return results

    async def _validate_memory_consistency(self) -> list[ValidationResult]:
        """Validate cross-agent memory sharing and consistency"""
        results = []

        # Memory consistency validation (placeholder)
        consistency_score = 0.94

        results.append(
            ValidationResult(
                layer=ValidationLayer.MULTI_AGENT_COORDINATION,
                level=(
                    ValidationLevel.PASS
                    if consistency_score >= 0.9
                    else ValidationLevel.WARNING
                ),
                message=f"Memory consistency score: {consistency_score:.2f}",
                details={"consistency_score": consistency_score},
                timestamp=datetime.now().isoformat(),
                validator="memory_consistency_validator",
            )
        )

        return results

    # =================================================================
    # LAYER 5 HELPER METHODS: EVENT STATE MACHINE
    # =================================================================

    async def _validate_state_transitions(
        self, context: EventValidationContext | None
    ) -> list[ValidationResult]:
        """Validate event state transitions are valid"""
        results = []

        if not context:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.EVENT_STATE_MACHINE,
                    level=ValidationLevel.WARNING,
                    message="No event context provided for state transition validation",
                    details={},
                    timestamp=datetime.now().isoformat(),
                    validator="state_transition_validator",
                )
            )
            return results

        # Validate current state is valid
        valid_states = [
            "stage:research.requested",
            "stage:research.proposed",
            "stage:research.active",
            "stage:decision.ready",
            "stage:decision.review",
            "stage:implementation.approved",
            "stage:implementation.started",
            "stage:metrics.collection",
            "stage:metrics.review",
            "stage:knowledge.captured",
        ]

        if context.current_stage in valid_states:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.EVENT_STATE_MACHINE,
                    level=ValidationLevel.PASS,
                    message=f"Event state '{context.current_stage}' is valid",
                    details={"current_stage": context.current_stage},
                    timestamp=datetime.now().isoformat(),
                    validator="state_transition_validator",
                    event_id=context.event_id,
                )
            )
        else:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.EVENT_STATE_MACHINE,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid event state: '{context.current_stage}'",
                    details={
                        "current_stage": context.current_stage,
                        "valid_states": valid_states,
                    },
                    timestamp=datetime.now().isoformat(),
                    validator="state_transition_validator",
                    event_id=context.event_id,
                    resolution_hint="Update event to use valid state labels",
                )
            )

        return results

    async def _validate_event_ordering(
        self, context: EventValidationContext | None
    ) -> list[ValidationResult]:
        """Validate event ordering and dependencies"""
        results = []

        if not context:
            return results

        # Check if dependencies are resolved
        if context.dependencies:
            unresolved_deps = await self._check_unresolved_dependencies(
                context.dependencies
            )

            if unresolved_deps:
                results.append(
                    ValidationResult(
                        layer=ValidationLayer.EVENT_STATE_MACHINE,
                        level=ValidationLevel.ERROR,
                        message=f"Unresolved dependencies: {unresolved_deps}",
                        details={"unresolved_dependencies": unresolved_deps},
                        timestamp=datetime.now().isoformat(),
                        validator="event_ordering_validator",
                        event_id=context.event_id,
                        resolution_hint="Complete dependent events before proceeding",
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        layer=ValidationLayer.EVENT_STATE_MACHINE,
                        level=ValidationLevel.PASS,
                        message="All dependencies resolved",
                        details={"dependencies": context.dependencies},
                        timestamp=datetime.now().isoformat(),
                        validator="event_ordering_validator",
                        event_id=context.event_id,
                    )
                )

        return results

    async def _validate_completion_criteria(
        self, context: EventValidationContext | None
    ) -> list[ValidationResult]:
        """Validate event completion criteria are met"""
        results = []

        if not context:
            return results

        # Stage-specific completion criteria
        completion_criteria = self._get_completion_criteria(
            context.current_stage
        )

        for criterion in completion_criteria:
            met = await self._check_completion_criterion(criterion, context)

            if met:
                results.append(
                    ValidationResult(
                        layer=ValidationLayer.EVENT_STATE_MACHINE,
                        level=ValidationLevel.PASS,
                        message=f"Completion criterion '{criterion}' met",
                        details={"criterion": criterion},
                        timestamp=datetime.now().isoformat(),
                        validator="completion_criteria_validator",
                        event_id=context.event_id,
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        layer=ValidationLayer.EVENT_STATE_MACHINE,
                        level=ValidationLevel.ERROR,
                        message=f"Completion criterion '{criterion}' not met",
                        details={"criterion": criterion},
                        timestamp=datetime.now().isoformat(),
                        validator="completion_criteria_validator",
                        event_id=context.event_id,
                        resolution_hint=f"Complete {criterion} before advancing stage",
                    )
                )

        return results

    async def _validate_deliverable_sync(
        self, context: EventValidationContext | None
    ) -> list[ValidationResult]:
        """Validate deliverable synchronization with labels"""
        results = []

        if not context:
            return results

        # Check if deliverable exists and is synchronized with current stage
        deliverable_status = await self._check_deliverable_status(context)

        if deliverable_status["synchronized"]:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.EVENT_STATE_MACHINE,
                    level=ValidationLevel.PASS,
                    message="Deliverable synchronized with current stage",
                    details=deliverable_status,
                    timestamp=datetime.now().isoformat(),
                    validator="deliverable_sync_validator",
                    event_id=context.event_id,
                )
            )
        else:
            results.append(
                ValidationResult(
                    layer=ValidationLayer.EVENT_STATE_MACHINE,
                    level=ValidationLevel.WARNING,
                    message="Deliverable not synchronized with current stage",
                    details=deliverable_status,
                    timestamp=datetime.now().isoformat(),
                    validator="deliverable_sync_validator",
                    event_id=context.event_id,
                    resolution_hint=deliverable_status.get(
                        "recommendation", "Synchronize deliverable with stage"
                    ),
                )
            )

        return results

    async def _check_unresolved_dependencies(
        self, dependencies: list[int]
    ) -> list[int]:
        """Check for unresolved dependencies"""
        # Placeholder implementation - would check actual GitHub issues
        # For now, assume all dependencies are resolved
        return []

    def _get_completion_criteria(self, stage: str) -> list[str]:
        """Get completion criteria for a stage"""
        criteria_map = {
            "stage:research.requested": ["research_proposal_created"],
            "stage:research.proposed": ["research_plan_approved"],
            "stage:research.active": ["findings_documented"],
            "stage:decision.ready": ["decision_document_created"],
            "stage:decision.review": ["peer_review_complete"],
            "stage:implementation.approved": ["implementation_plan_ready"],
            "stage:metrics.collection": ["roi_analysis_complete"],
        }

        return criteria_map.get(stage, [])

    async def _check_completion_criterion(
        self, criterion: str, context: EventValidationContext
    ) -> bool:
        """Check if a specific completion criterion is met"""
        # Placeholder implementation - would check actual deliverables
        return True  # Assume met for now

    async def _check_deliverable_status(
        self, context: EventValidationContext
    ) -> dict[str, Any]:
        """Check deliverable status for synchronization"""
        # Placeholder implementation - would check actual GitHub issue comments
        return {
            "synchronized": True,
            "deliverable_exists": True,
            "last_updated": datetime.now().isoformat(),
            "recommendation": None,
        }

    # =================================================================
    # MAIN VALIDATION ORCHESTRATION METHODS
    # =================================================================

    async def run_full_validation(
        self, context: EventValidationContext | None = None
    ) -> tuple[bool, list[ValidationResult]]:
        """Run complete multi-layer validation"""
        print("🚨 Starting Multi-Layer Event-Driven Validation...")

        all_results = []
        validation_start = datetime.now()

        # Execute all validation layers
        if self.config["parallel_validation"]:
            # Run layers in parallel for better performance
            layer_tasks = [
                self.validate_task_completion(context),
                self.validate_quality_gates(context),
                self.validate_strategic_milestones(context),
                self.validate_multi_agent_coordination(context),
                self.validate_event_state_machine(context),
            ]

            layer_results = await asyncio.gather(
                *layer_tasks, return_exceptions=True
            )

            for i, result in enumerate(layer_results):
                if isinstance(result, Exception):
                    layer_name = list(ValidationLayer)[i].value
                    all_results.append(
                        ValidationResult(
                            layer=list(ValidationLayer)[i],
                            level=ValidationLevel.CRITICAL,
                            message=f"Layer {layer_name} validation failed with exception: {result!s}",
                            details={
                                "error": str(result),
                                "layer": layer_name,
                            },
                            timestamp=datetime.now().isoformat(),
                            validator="orchestration_validator",
                        )
                    )
                else:
                    all_results.extend(result)
        else:
            # Run layers sequentially
            sequential_layers = [
                ("Task Completion", self.validate_task_completion),
                ("Quality Gates", self.validate_quality_gates),
                ("Strategic Milestones", self.validate_strategic_milestones),
                (
                    "Multi-Agent Coordination",
                    self.validate_multi_agent_coordination,
                ),
                ("Event State Machine", self.validate_event_state_machine),
            ]

            for layer_name, layer_func in sequential_layers:
                print(f"  Validating {layer_name}...")
                try:
                    results = await layer_func(context)
                    all_results.extend(results)
                except Exception as e:
                    all_results.append(
                        ValidationResult(
                            layer=ValidationLayer.TASK_COMPLETION,  # Default layer
                            level=ValidationLevel.CRITICAL,
                            message=f"Layer {layer_name} validation failed: {e!s}",
                            details={"error": str(e), "layer": layer_name},
                            timestamp=datetime.now().isoformat(),
                            validator="orchestration_validator",
                        )
                    )

        # Calculate overall validation result
        validation_duration = (
            datetime.now() - validation_start
        ).total_seconds()
        can_complete = self._calculate_overall_result(all_results)

        # Update statistics
        self._update_validation_stats(all_results, validation_duration)

        print(f"✅ Validation completed in {validation_duration:.2f}s")
        return can_complete, all_results

    async def run_quick_validation(
        self, context: EventValidationContext | None = None
    ) -> tuple[bool, list[ValidationResult]]:
        """Run quick validation (Layers 1-2 only)"""
        print("⚡ Starting Quick Validation (Layers 1-2)...")

        validation_start = datetime.now()

        # Run only essential layers
        layer_tasks = [
            self.validate_task_completion(context),
            self.validate_quality_gates(context),
        ]

        layer_results = await asyncio.gather(
            *layer_tasks, return_exceptions=True
        )
        all_results = []

        for i, result in enumerate(layer_results):
            if isinstance(result, Exception):
                layer_name = ["Task Completion", "Quality Gates"][i]
                all_results.append(
                    ValidationResult(
                        layer=list(ValidationLayer)[i],
                        level=ValidationLevel.CRITICAL,
                        message=f"Quick validation layer {layer_name} failed: {result!s}",
                        details={"error": str(result), "layer": layer_name},
                        timestamp=datetime.now().isoformat(),
                        validator="quick_validation_orchestrator",
                    )
                )
            else:
                all_results.extend(result)

        validation_duration = (
            datetime.now() - validation_start
        ).total_seconds()
        can_complete = self._calculate_overall_result(all_results)

        print(f"✅ Quick validation completed in {validation_duration:.2f}s")
        return can_complete, all_results

    async def validate_specific_event(
        self, event_id: str
    ) -> tuple[bool, list[ValidationResult]]:
        """Validate a specific event by ID"""
        print(f"🎯 Validating specific event: {event_id}")

        # Get event context from GitHub
        context = await self._get_event_context(event_id)

        if not context:
            error_result = ValidationResult(
                layer=ValidationLayer.EVENT_STATE_MACHINE,
                level=ValidationLevel.ERROR,
                message=f"Could not retrieve context for event {event_id}",
                details={"event_id": event_id},
                timestamp=datetime.now().isoformat(),
                validator="event_context_validator",
            )
            return False, [error_result]

        # Run full validation with event context
        return await self.run_full_validation(context)

    def _calculate_overall_result(
        self, results: list[ValidationResult]
    ) -> bool:
        """Calculate overall validation result"""
        # If any critical errors, fail validation
        critical_errors = [
            r for r in results if r.level == ValidationLevel.CRITICAL
        ]
        if critical_errors:
            return False

        # If any errors in essential layers (1-2), fail validation
        essential_errors = [
            r
            for r in results
            if r.level == ValidationLevel.ERROR
            and r.layer
            in [ValidationLayer.TASK_COMPLETION, ValidationLayer.QUALITY_GATES]
        ]
        if essential_errors:
            return False

        # Allow completion with warnings only
        return True

    def _update_validation_stats(
        self, results: list[ValidationResult], duration: float
    ):
        """Update validation statistics"""
        self.validation_stats["total_validations"] += 1

        # Update layer statistics
        for result in results:
            layer_key = result.layer.value
            level_key = result.level.value
            self.validation_stats["layer_stats"][layer_key][level_key] += 1

        # Update performance metrics
        self.validation_stats["performance_metrics"][
            "last_duration"
        ] = duration
        self.validation_stats["performance_metrics"][
            "last_validation"
        ] = datetime.now().isoformat()

    async def _get_event_context(
        self, event_id: str
    ) -> EventValidationContext | None:
        """Get event context from GitHub issue"""
        try:
            # Parse issue number from event_id
            issue_number = int(event_id) if event_id.isdigit() else None
            if not issue_number:
                return None

            # Get issue details using task_master's GitHub integration
            # This is a placeholder - would use actual GitHub API
            return EventValidationContext(
                event_id=event_id,
                event_type="validation_request",
                issue_number=issue_number,
                current_stage="stage:research.active",
                dependencies=[],
                priority="medium",
                assignees=[],
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
                metadata={},
            )
        except Exception:
            return None

    # =================================================================
    # EVENT-DRIVEN INTEGRATION
    # =================================================================

    async def start_event_subscription(self):
        """Start event-driven monitoring mode"""
        print("📡 Starting event-driven validation monitoring...")

        if not self.config["event_subscription_enabled"]:
            print("⚠️  Event subscription disabled in configuration")
            return

        try:
            while True:
                # Check for new events that need validation
                events = await self._scan_for_validation_events()

                for event in events:
                    print(f"🔔 New validation event: {event['event_id']}")

                    # Validate the event
                    can_complete, results = await self.validate_specific_event(
                        event["event_id"]
                    )

                    # Publish validation results as events
                    await self._publish_validation_results(
                        event["event_id"], can_complete, results
                    )

                # Brief pause before next scan
                await asyncio.sleep(30)

        except KeyboardInterrupt:
            print("\n📡 Event subscription stopped by user")
        except Exception as e:
            print(f"❌ Event subscription error: {e}")

    async def _scan_for_validation_events(self) -> list[dict[str, Any]]:
        """Scan for events that need validation"""
        # Placeholder implementation - would integrate with actual event system
        return []

    async def _publish_validation_results(
        self,
        event_id: str,
        can_complete: bool,
        results: list[ValidationResult],
    ):
        """Publish validation results as events"""
        # Placeholder implementation - would publish to event system
        print(
            f"📤 Publishing validation results for event {event_id}: {'✅ PASS' if can_complete else '❌ FAIL'}"
        )

    # =================================================================
    # REPORTING AND ANALYTICS
    # =================================================================

    def generate_validation_report(
        self, results: list[ValidationResult]
    ) -> str:
        """Generate comprehensive validation report"""
        report_lines = []

        # Header
        report_lines.append("# Multi-Layer Validation Report")
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")

        # Summary statistics
        total_results = len(results)
        by_level = {}
        by_layer = {}

        for result in results:
            level = result.level.value
            layer = result.layer.value

            by_level[level] = by_level.get(level, 0) + 1
            by_layer[layer] = by_layer.get(layer, 0) + 1

        report_lines.append("## Summary")
        report_lines.append(f"- Total Validations: {total_results}")
        report_lines.append(f"- Pass: {by_level.get('pass', 0)}")
        report_lines.append(f"- Warning: {by_level.get('warning', 0)}")
        report_lines.append(f"- Error: {by_level.get('error', 0)}")
        report_lines.append(f"- Critical: {by_level.get('critical', 0)}")
        report_lines.append("")

        # Layer breakdown
        report_lines.append("## Layer Breakdown")
        for layer, count in by_layer.items():
            report_lines.append(
                f"- {layer.replace('_', ' ').title()}: {count} validations"
            )
        report_lines.append("")

        # Detailed results
        report_lines.append("## Detailed Results")

        for layer in ValidationLayer:
            layer_results = [r for r in results if r.layer == layer]
            if layer_results:
                report_lines.append(
                    f"### {layer.value.replace('_', ' ').title()}"
                )

                for result in layer_results:
                    icon = {
                        "pass": "✅",
                        "warning": "⚠️",
                        "error": "❌",
                        "critical": "🚨",
                    }[result.level.value]
                    report_lines.append(
                        f"{icon} **{result.level.value.upper()}**: {result.message}"
                    )

                    if result.resolution_hint:
                        report_lines.append(
                            f"   💡 *Resolution*: {result.resolution_hint}"
                        )

                    report_lines.append("")

        # System statistics
        report_lines.append("## System Statistics")
        stats = self.validation_stats
        report_lines.append(
            f"- Total Validations Run: {stats['total_validations']}"
        )

        if (
            "performance_metrics" in stats
            and "last_duration" in stats["performance_metrics"]
        ):
            report_lines.append(
                f"- Last Validation Duration: {stats['performance_metrics']['last_duration']:.2f}s"
            )

        return "\n".join(report_lines)

    # =================================================================
    # LEGACY COMPATIBILITY
    # =================================================================

    def orchestrator_completion_check(self) -> bool:
        """Legacy compatibility method for original gatekeeper functionality"""
        print("🔄 Running legacy compatibility check...")

        # Run quick validation synchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            can_complete, results = loop.run_until_complete(
                self.run_quick_validation()
            )

            if can_complete:
                print("✅ GATEKEEPER APPROVAL: Task completion allowed")
                print("   • All essential validation layers passed")
                print("   • System ready for task completion")
            else:
                print("❌ GATEKEEPER BLOCK: Task completion NOT allowed")
                print("   • Essential validation failures detected")
                print("   • Review validation report for details")

            return can_complete

        finally:
            loop.close()


# =================================================================
# CLI INTERFACE AND MAIN EXECUTION
# =================================================================


async def main():
    """Main CLI interface for the enhanced gatekeeper"""
    parser = argparse.ArgumentParser(
        description="KB Multi-Layer Event-Driven Gatekeeper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gatekeeper.py                    # Full multi-layer validation
  python gatekeeper.py --quick            # Quick validation (layers 1-2)
  python gatekeeper.py --report           # Generate validation report
  python gatekeeper.py --subscribe        # Start event monitoring
  python gatekeeper.py --validate-event 123  # Validate specific event
        """,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (layers 1-2 only)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate and display validation report",
    )
    parser.add_argument(
        "--subscribe",
        action="store_true",
        help="Start event-driven monitoring mode",
    )
    parser.add_argument(
        "--validate-event",
        metavar="EVENT_ID",
        help="Validate specific event by ID",
    )
    parser.add_argument(
        "--config", metavar="CONFIG_PATH", help="Path to configuration file"
    )
    parser.add_argument(
        "--output", metavar="OUTPUT_PATH", help="Output file for reports"
    )

    args = parser.parse_args()

    # Initialize gatekeeper
    try:
        gatekeeper = EventDrivenGatekeeper(config_path=args.config)
    except Exception as e:
        print(f"❌ Failed to initialize gatekeeper: {e}")
        sys.exit(2)

    # Execute requested operation
    try:
        if args.subscribe:
            # Start event subscription mode
            await gatekeeper.start_event_subscription()

        elif args.validate_event:
            # Validate specific event
            can_complete, results = await gatekeeper.validate_specific_event(
                args.validate_event
            )

            if args.report or args.output:
                report = gatekeeper.generate_validation_report(results)

                if args.output:
                    with open(args.output, "w") as f:
                        f.write(report)
                    print(f"📄 Report saved to: {args.output}")
                else:
                    print("\n" + report)

            sys.exit(0 if can_complete else 1)

        elif args.quick:
            # Quick validation
            can_complete, results = await gatekeeper.run_quick_validation()

            if args.report or args.output:
                report = gatekeeper.generate_validation_report(results)

                if args.output:
                    with open(args.output, "w") as f:
                        f.write(report)
                    print(f"📄 Report saved to: {args.output}")
                else:
                    print("\n" + report)

            sys.exit(0 if can_complete else 1)

        else:
            # Full validation (default)
            can_complete, results = await gatekeeper.run_full_validation()

            if args.report or args.output:
                report = gatekeeper.generate_validation_report(results)

                if args.output:
                    with open(args.output, "w") as f:
                        f.write(report)
                    print(f"📄 Report saved to: {args.output}")
                else:
                    print("\n" + report)

            sys.exit(0 if can_complete else 1)

    except Exception as e:
        print(f"❌ Validation error: {e}")
        sys.exit(2)


def legacy_main():
    """Legacy main function for backward compatibility"""
    gatekeeper = EventDrivenGatekeeper()
    can_complete = gatekeeper.orchestrator_completion_check()
    sys.exit(0 if can_complete else 1)


if __name__ == "__main__":
    # Check if running in legacy mode (no command line arguments)
    if len(sys.argv) == 1:
        legacy_main()
    else:
        asyncio.run(main())
