# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""ReAct: multi-round reason-act loop built on operate.

Handler signature: react(params, ctx) -> final answer instance
Streaming variant: react_stream(params, ctx) -> AsyncGenerator[round analysis]

Each round:
  1. operate() with ReActAnalysis as request model
  2. LLM produces reasoning + planned_actions + extension_needed
  3. If actions present and allowed, operate handles execution
  4. If extension_needed and rounds remain, loop continues
  5. Final round: operate() with user's response model for the answer
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, field_validator

from lionagi.beta.rules import Validator
from lionagi.ln.types._sentinel import MaybeUnset, Unset
from lionagi.ln.types import HashableModel, ModelConfig, Params

from .generate import GenerateParams
from .operate import OperateParams

if TYPE_CHECKING:
    from lionagi.beta.resource.imodel import iModel
    from lionagi.ln.types import Operable
    from lionagi.beta.session.context import RequestContext

__all__ = (
    "Analysis",
    "PlannedAction",
    "ReActAnalysis",
    "ReActParams",
    "react",
    "react_stream",
)


# ---------------------------------------------------------------------------
# ReAct spec models
# ---------------------------------------------------------------------------


class PlannedAction(HashableModel):
    """Short descriptor for an upcoming tool invocation."""

    action_type: str | None = Field(
        default=None,
        description="Name or type of tool/action to invoke.",
    )
    description: str | None = Field(
        default=None,
        description="Concise summary of what the action entails and why.",
    )


class ReActAnalysis(HashableModel):
    """Structured reasoning output for each ReAct round.

    The LLM fills this to express its chain-of-thought, plan actions,
    and signal whether more rounds are needed.
    """

    FIRST_ROUND_PROMPT: ClassVar[str] = (
        "You can perform multiple reason-action steps for accuracy. "
        "If you are not ready to finalize, set extension_needed to True. "
        "Set extension_needed to True if the overall goal is not yet achieved. "
        "Do not set it to False if you are just providing an interim answer. "
        "You have up to {max_rounds} rounds. Strategize accordingly."
    )
    CONTINUE_PROMPT: ClassVar[str] = (
        "Another round is available. You may do multiple actions if needed. "
        "You have up to {remaining} rounds remaining. Continue."
    )
    ANSWER_PROMPT: ClassVar[str] = (
        "Given your reasoning and actions, provide the final answer "
        "to the user's request:\n\n{instruction}"
    )

    analysis: str = Field(
        ...,
        description=(
            "Free-form reasoning or chain-of-thought summary. "
            "Use for planning, reflection, and progress tracking."
        ),
    )
    extension_needed: bool = Field(
        False,
        description="True if more rounds are needed. False triggers final answer.",
    )


class Analysis(HashableModel):
    """Final answer model (default request_model for react)."""

    answer: str | None = None

    @field_validator("answer", mode="before")
    @classmethod
    def _validate_answer(cls, value: Any) -> str | None:
        if not value:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        if not isinstance(value, str):
            raise ValueError("Answer must be a non-empty string.")
        return value.strip()


# ---------------------------------------------------------------------------
# ReAct params
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReActParams(Params):
    """Parameters for ReAct loop.

    Attributes:
        instruction: The user's task/question.
        operable: Spec definition for structured output composition.
        validator: Rule-based validator. Defaults to beta.rules.Validator.
        generate_params: LLM generation config.
        max_rounds: Maximum reason-act rounds before forcing final answer.
        request_model: Model for the final answer (default: Analysis).
        invoke_actions: Enable tool execution in each round.
        action_strategy: Default strategy (overridden by LLM per-round).
        persist: Persist messages to branch.
    """

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    # Required
    instruction: str
    operable: Operable
    generate_params: GenerateParams
    validator: Validator = field(default_factory=Validator)

    # ReAct loop config
    max_rounds: int = 3
    request_model: type | None = None
    invoke_actions: bool = True
    persist: bool = True

    # Action defaults
    action_strategy: Literal["sequential", "concurrent"] = "concurrent"
    max_concurrent: int | None = None
    throttle_period: float | None = None

    # Validation
    auto_fix: bool = True
    strict: bool = True

    # Parse overrides
    parse_imodel: MaybeUnset[iModel | str] = Unset
    parse_imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    similarity_threshold: float = 0.85
    max_retries: int = 3

    # Toolkit support
    toolkits: list[Any] | None = None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def react(params: ReActParams, ctx: RequestContext) -> Any:
    """ReAct handler: collects all rounds, returns final answer.

    The final answer is extracted from the last round's Analysis.answer
    if request_model is not provided (defaults to Analysis).
    """
    result = None
    async for round_result in ctx.stream_conduct("react_stream", params):
        result = round_result

    # Last yield is the final answer
    if result is not None and hasattr(result, "answer"):
        return result.answer
    return result


async def react_stream(params: ReActParams, ctx: RequestContext) -> AsyncGenerator[Any, None]:
    """Streaming ReAct: yields each round's structured analysis.

    Yields:
        Round 1..N: ReActAnalysis instances (reasoning + action results)
        Final: request_model instance (the answer)
    """
    max_rounds = min(params.max_rounds, 100)

    # --- Round 1: Initial analysis ---
    instruction_with_prompt = (
        params.instruction + "\n\n" + ReActAnalysis.FIRST_ROUND_PROMPT.format(max_rounds=max_rounds)
    )

    analysis = await _safe_round(params, ctx, instruction_with_prompt, ReActAnalysis)
    yield analysis

    # --- Extension rounds ---
    remaining = max_rounds - 1
    while remaining > 0 and _needs_extension(analysis):
        prompt = ReActAnalysis.CONTINUE_PROMPT.format(remaining=remaining)
        analysis = await _safe_round(params, ctx, prompt, ReActAnalysis)
        yield analysis
        remaining -= 1

    # --- Final answer ---
    answer_model = params.request_model or Analysis
    answer_prompt = ReActAnalysis.ANSWER_PROMPT.format(instruction=params.instruction)
    final = await _run_round(params, ctx, answer_prompt, answer_model, invoke_actions=False)
    yield final


async def _safe_round(
    params: ReActParams,
    ctx: RequestContext,
    instruction: str,
    request_model: type,
    invoke_actions: bool | None = None,
) -> Any:
    """Execute a round, returning validation errors as context instead of raising.

    When the LLM emits malformed action requests (e.g. wrong field names),
    the error is fed back so the model can retry in the next round rather
    than crashing the entire react loop.
    """
    try:
        return await _run_round(params, ctx, instruction, request_model, invoke_actions)
    except Exception as e:
        error_msg = str(e)
        if hasattr(request_model, "model_fields"):
            try:
                return request_model.model_construct(
                    reasoning=f"Previous round failed with: {error_msg}",
                    action_requests=[],
                    extension_needed=True,
                )
            except Exception:
                raise e from None
        raise


async def _run_round(
    params: ReActParams,
    ctx: RequestContext,
    instruction: str,
    request_model: type,
    invoke_actions: bool | None = None,
) -> Any:
    """Execute a single ReAct round via operate."""
    from lionagi.ln.types import Operable

    # Build operable from the round's request model
    round_operable = Operable.from_structure(request_model)

    # Override instruction in generate params
    gen_params = params.generate_params.with_updates(copy_containers="deep", primary=instruction)

    # Resolve action strategy from previous analysis if available
    should_act = invoke_actions if invoke_actions is not None else params.invoke_actions

    operate_params = OperateParams(
        operable=round_operable,
        validator=params.validator,
        generate_params=gen_params,
        invoke_actions=should_act,
        action_strategy=params.action_strategy,
        max_concurrent=params.max_concurrent,
        throttle_period=params.throttle_period,
        persist=params.persist,
        auto_fix=params.auto_fix,
        strict=params.strict,
        parse_imodel=params.parse_imodel,
        parse_imodel_kwargs=params.parse_imodel_kwargs,
        similarity_threshold=params.similarity_threshold,
        max_retries=params.max_retries,
        toolkits=params.toolkits,
    )

    return await ctx.conduct("operate", operate_params)


def _needs_extension(analysis: Any) -> bool:
    """Check if the analysis signals more rounds are needed."""
    if hasattr(analysis, "extension_needed"):
        return bool(analysis.extension_needed)
    if isinstance(analysis, dict):
        return bool(analysis.get("extension_needed", False))
    return False
