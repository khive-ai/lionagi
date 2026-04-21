"""Control decisions for flow operations.

A control operation evaluates DAG state and returns a ControlDecision
that the executor applies to flow progression. This generalizes the
original ``control=True`` / ``FlowControlVerdict`` pattern to support
gates, quorums, halts, retries, and custom control types.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ControlDecision(BaseModel):
    """Result of evaluating a control operation."""

    action: Literal["proceed", "halt", "route", "retry", "iterate", "abort"] = Field(
        description=(
            "proceed: dispatch default downstream nodes. "
            "halt: pause all in-flight, surface to orchestrator. "
            "route: dispatch only 'targets' (skip others). "
            "retry: re-dispatch 'targets' with backoff. "
            "iterate: re-plan via orchestrator (legacy control=True). "
            "abort: terminate flow, write failure artifacts."
        ),
    )
    reason: str = Field(description="Human-readable explanation")
    targets: list[str] = Field(
        default_factory=list,
        description="Node IDs for route/retry actions",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Type-specific payload (quorum ratio, gate failures, etc.)",
    )
