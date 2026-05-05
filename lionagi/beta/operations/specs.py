# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Specs for agent operations.

Action: validated tool-call request model (function + arguments).
ActionResult: validated tool-call result model (function, result, error).
get_action_spec / get_action_result_spec: Spec factories for
runtime operable composition in operate.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any, Literal

from pydantic import BaseModel, Field, JsonValue, field_validator

from lionagi.ln.types import HashableModel
from lionagi.ln._to_list import to_list
from lionagi.ln.fuzzy import extract_json, to_dict

__all__ = (
    "Action",
    "ActionResult",
    "Instruct",
    "get_action_result_spec",
    "get_action_spec",
    "get_instruct_spec",
)


class Action(HashableModel):
    """Validated tool/action request: (function, arguments) pair.

    Parsed from LLM output via fuzzy JSON extraction.
    """

    function: str = Field(
        description=(
            "Function name from tool_schemas. Never invent names outside provided schemas."
        ),
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description=("Argument dict for the function. Use only names/types from tool_schemas."),
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def _coerce_arguments(cls, value: Any) -> dict[str, Any]:
        """Coerce arguments into a dict, handling JSON strings."""
        if isinstance(value, dict):
            return value
        return to_dict(
            value,
            fuzzy_parse=True,
            recursive=True,
            recursive_python_only=False,
        )

    @classmethod
    def create(cls, content: str | dict | BaseModel) -> list[Action]:
        """Parse raw LLM output into Action instances. Returns [] on failure."""
        try:
            parsed = _parse_action_blocks(content)
            return [cls.model_validate(item) for item in parsed] if parsed else []
        except Exception:
            return []


class ActionResult(HashableModel):
    """Validated tool-call result: (function, result, error) triple.

    Produced by act stage after executing Action requests.
    """

    function: str = Field(description="Function name that was called.")
    result: Any = Field(default=None, description="Return value on success.")
    error: str | None = Field(default=None, description="Error message on failure.")

    @property
    def success(self) -> bool:
        return self.error is None


class Instruct(HashableModel):
    """Instruction bundle for orchestrated task handoff.

    Encapsulates everything needed to hand off a task: what to do,
    strategic guidance, background context, and execution preferences.
    Used as a structured handoff between orchestrator and workers.
    """

    instruction: str | None = Field(
        default=None,
        description=(
            "Clear, actionable task definition. Specify the primary goal, "
            "key constraints, and success criteria."
        ),
    )
    guidance: JsonValue | None = Field(
        default=None,
        description=(
            "Strategic direction: preferred methods, quality benchmarks, "
            "resource constraints, or compliance requirements."
        ),
    )
    context: JsonValue | None = Field(
        default=None,
        description=(
            "Background information directly relevant to the task: "
            "environment, prior outcomes, system states, or dependencies."
        ),
    )
    reason: bool = Field(
        default=False,
        description=(
            "Include reasoning: explanations of decisions, trade-offs, "
            "alternatives considered, and confidence assessment."
        ),
    )
    actions: bool = Field(
        default=False,
        description=(
            "Enable action execution via tool_schemas. "
            "True: execute tool calls. False: analysis only."
        ),
    )
    action_strategy: Literal["sequential", "concurrent"] = Field(
        default="concurrent",
        description="How to execute actions: sequential or concurrent.",
    )

    @field_validator("action_strategy", mode="before")
    @classmethod
    def _validate_action_strategy(cls, v: Any) -> str:
        if v not in ("sequential", "concurrent"):
            return "concurrent"
        return v


@cache
def get_instruct_spec():
    """Spec for instruct_model: Instruct | None."""
    from lionagi.ln.types import Spec

    return Spec(Instruct, name="instruct_model").as_nullable()


@cache
def get_action_spec():
    """Spec for action_requests: list[Action] | None."""
    from lionagi.ln.types import Spec

    return Spec(Action, name="action_requests").as_listable().as_nullable()


@cache
def get_action_result_spec():
    """Spec for action_results: list[ActionResult] | None."""
    from lionagi.ln.types import Spec

    return Spec(ActionResult, name="action_results").as_listable().as_nullable()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_action_blocks(content: str | dict | BaseModel) -> list[dict]:
    """Extract action request dicts from raw content.

    Normalizes provider-specific key names to {function, arguments}.
    """
    json_blocks: list = []

    if isinstance(content, BaseModel):
        json_blocks = [content.model_dump()]
    elif isinstance(content, str):
        json_blocks = extract_json(content, fuzzy_parse=True)
        if not json_blocks:
            # Fallback: try extracting from ```python ... ``` blocks
            matches = re.findall(r"```python\s*(.*?)\s*```", content, re.DOTALL)
            json_blocks = to_list(
                [extract_json(m, fuzzy_parse=True) for m in matches],
                dropna=True,
            )
    elif isinstance(content, dict):
        json_blocks = [content]

    if json_blocks and not isinstance(json_blocks, list):
        json_blocks = [json_blocks]

    out: list[dict] = []
    for block in json_blocks:
        if not isinstance(block, dict):
            continue
        normalized = _normalize_action_keys(block)
        if normalized:
            out.append(normalized)
    return out


def _normalize_action_keys(d: dict) -> dict | None:
    """Map provider-specific key names to canonical {function, arguments}.

    Returns None if required keys are missing.
    """
    result: dict[str, Any] = {}

    # Handle nested function.name pattern
    if "function" in d and isinstance(d["function"], dict) and "name" in d["function"]:
        d = {**d, "function": d["function"]["name"]}

    for k, v in d.items():
        # Strip common prefixes: action_name → name, recipient_name → name
        normalized = k.replace("action_", "").replace("recipient_", "").removesuffix("s")
        if normalized in ("name", "function", "recipient"):
            result["function"] = v
        elif normalized in ("parameter", "argument", "arg", "param"):
            result["arguments"] = to_dict(v, str_type="json", fuzzy_parse=True, suppress=True)

    if "function" in result:
        result.setdefault("arguments", {})
        return result
    return None
