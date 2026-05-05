# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Generate operation: stateless LLM call with message preparation.

Handler signature: generate(params, ctx) → Calling | text | raw | Message
Lowest-level operation — no message persistence, no validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, JsonValue

from lionagi.protocols.messages.instruction import InstructionContent as Instruction
from lionagi.protocols.messages.prepare import prepare_messages_for_chat
from lionagi._errors import ConfigurationError
from lionagi.beta.session.constraints import resource_must_be_accessible
from lionagi.ln.types import ModelConfig, Params
from lionagi.ln.types._sentinel import MaybeUnset, Unset
from lionagi.protocols.messages import Message

from lionagi.protocols.messages.rendering import CustomRenderer
from .utils import ReturnAs, handle_return

if TYPE_CHECKING:
    from typing import Any

    from lionagi.beta.resource.imodel import iModel
    from lionagi.beta.session.context import RequestContext
    from lionagi.beta.session.session import Branch

    Session = Any  # Session not yet migrated to beta

__all__ = ("GenerateParams", "generate", "handle_return")


@dataclass(frozen=True, slots=True)
class GenerateParams(Params):
    """LLM call parameters; provide either ``instruction`` (pre-built) or ``primary`` (string) — not both."""

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty", "dataclass", "pydantic"}))

    instruction: MaybeUnset[Instruction | Message] = Unset
    primary: MaybeUnset[str] = Unset
    context: MaybeUnset[JsonValue] = Unset
    imodel: MaybeUnset[iModel | str] = Unset
    images: MaybeUnset[list[str]] = Unset
    image_detail: MaybeUnset[Literal["low", "high", "auto"]] = Unset
    tool_schemas: MaybeUnset[list[str]] = Unset
    request_model: MaybeUnset[type[BaseModel]] = Unset
    structure_format: Literal["json", "custom", "lndl"] = "json"
    custom_renderer: MaybeUnset[CustomRenderer] = Unset
    return_as: ReturnAs = ReturnAs.CALLING
    imodel_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def instruction_message(self) -> Message:
        """Resolve to a Message, building from primary if needed."""
        if not self.is_sentinel_field("instruction"):
            if isinstance(self.instruction, Message):
                return self.instruction
            if isinstance(self.instruction, Instruction):
                return Message(content=self.instruction)

        content = Instruction.create(
            primary=self.primary,
            context=self.context,
            images=self.images,
            image_detail=self.image_detail,
            tool_schemas=self.tool_schemas,
            request_model=self.request_model,
            structure_format=self.structure_format,
            custom_renderer=self.custom_renderer,
        )
        return Message(content=content)


async def generate(params: GenerateParams, ctx: RequestContext) -> Any:
    session = await ctx.get_session()
    imodel = params.imodel if not params.is_sentinel_field("imodel") else None

    # Propagate verbose from ctx to imodel_kwargs for streaming pretty-print
    imodel_kwargs = dict(params.imodel_kwargs)
    if ctx.metadata.get("_verbose"):
        imodel_kwargs.setdefault("verbose", True)

    return await _generate(
        session=session,
        branch=ctx.branch,
        instruction=params.instruction_message,
        imodel=imodel,
        return_as=params.return_as,
        **imodel_kwargs,
    )


async def _generate(
    session: Session,
    branch: Branch | str,
    instruction: Message | Instruction | UUID,
    imodel: iModel | str | None = None,
    return_as: ReturnAs = ReturnAs.CALLING,
    **imodel_kwargs: Any,
) -> Any:
    if imodel is None:
        imodel = session.default_gen_model
    elif isinstance(imodel, str):
        imodel = session.resources.get(imodel, None)
    if imodel is None:
        raise ConfigurationError(
            "Provided imodel could not be resolved, or no default model is set."
        )

    branch = session.get_branch(branch)
    resource_must_be_accessible(branch, imodel.name)

    if isinstance(instruction, UUID):
        instruction = session.messages[instruction]
    elif isinstance(instruction, Instruction):
        instruction = Message(content=instruction)

    # CLI endpoints use streaming run path
    is_cli = (
        getattr(imodel.backend, "config", None) and imodel.backend.config.endpoint == "query_cli"
    )
    if is_cli:
        from .run import run_and_collect

        primary = ""
        if isinstance(instruction, Message) and hasattr(instruction.content, "primary"):
            primary = instruction.content.primary or ""
        elif isinstance(instruction, Message):
            primary = str(instruction.content)

        return await run_and_collect(
            session=session,
            branch=branch,
            primary=primary,
            imodel=imodel,
            **imodel_kwargs,
        )

    prepared_msgs = prepare_messages_for_chat(
        session.messages,
        branch,
        instruction,
        system_prefix=session.config.system_prefix,
        aggregate_actions=session.config.aggregate_actions,
        round_notifications=session.config.round_notifications,
        scratchpad=branch.scratchpad_summary(),
    )
    calling = await imodel.invoke(messages=prepared_msgs, **imodel_kwargs)
    return handle_return(calling, return_as)
