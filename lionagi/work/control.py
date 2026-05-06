# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Control-flow decisions for operation DAG execution."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ("ControlAction", "ControlDecision")

ControlAction = Literal["proceed", "halt", "abort", "skip", "route", "retry"]


class ControlDecision(BaseModel):
    """Executor decision produced by control operations."""

    action: ControlAction = "proceed"
    reason: str = ""
    targets: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
