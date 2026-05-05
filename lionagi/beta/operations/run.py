# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Run operation: CLI streaming generate with message accumulation.

Streams chunks from CLI endpoints (Claude Code, Codex, Gemini),
accumulates text/thinking, creates ActionRequest/ActionResponse
messages, and persists to branch. The streaming counterpart of generate.

Used as the Middle for operate() when the endpoint is CLI-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lionagi.beta.core.message import (
    ActionRequest,
    ActionResponse,
    Assistant,
    Instruction,
    prepare_messages_for_chat,
)
from lionagi._errors import ConfigurationError
from lionagi.beta.session.constraints import resource_must_be_accessible
from lionagi.ln.types import ModelConfig, Params
from lionagi.ln.types._sentinel import MaybeUnset, Unset
from lionagi.protocols.messages import Message

if TYPE_CHECKING:
    from typing import Any

    from lionagi.beta.resource.imodel import iModel
    from lionagi.beta.session.context import RequestContext
    from lionagi.beta.session.session import Branch

    Session = Any  # Session not yet migrated to beta

__all__ = ("RunParams", "run", "run_and_collect")


@dataclass(frozen=True, slots=True)
class RunParams(Params):
    """Parameters for the run (CLI streaming) operation.

    Extends GenerateParams concept for streaming. The key addition is
    stream_chunk_hook for real-time chunk processing (display, persist, etc.).
    """

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty", "dataclass", "pydantic"}))

    primary: MaybeUnset[str] = Unset
    context: MaybeUnset[Any] = Unset
    imodel: MaybeUnset[iModel | str] = Unset
    stream_chunk_hook: Any | None = None
    imodel_kwargs: dict[str, Any] = field(default_factory=dict)


async def run(params: RunParams, ctx: RequestContext) -> str:
    """Run operation handler: streaming generate for CLI endpoints."""
    session = await ctx.get_session()
    imodel = params.imodel if not params.is_sentinel_field("imodel") else None

    imodel_kwargs = dict(params.imodel_kwargs)

    return await _run(
        session=session,
        branch=ctx.branch,
        primary=params.primary if not params.is_sentinel_field("primary") else "",
        context=params.context if not params.is_sentinel_field("context") else None,
        imodel=imodel,
        stream_chunk_hook=params.stream_chunk_hook,
        **imodel_kwargs,
    )


async def _run(
    session: Session,
    branch: Branch | str,
    primary: str,
    context: Any | None = None,
    imodel: iModel | str | None = None,
    stream_chunk_hook: Any | None = None,
    **imodel_kwargs: Any,
) -> str:
    """Core run: resolve model → stream → accumulate messages → return text."""
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

    instruction = Instruction.create(primary=primary, context=context)
    ins_msg = Message(content=instruction)
    prepared_msgs = prepare_messages_for_chat(
        session.messages,
        branch,
        ins_msg,
        system_prefix=session.config.system_prefix,
        aggregate_actions=session.config.aggregate_actions,
        round_notifications=session.config.round_notifications,
        scratchpad=branch.scratchpad_summary(),
    )

    # Add instruction to branch
    session.add_message(ins_msg, branches=branch)

    # Resume from previous session if available
    if sid := imodel.provider_metadata.get("session_id"):
        imodel_kwargs.setdefault("resume", sid)

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    pending_requests: dict[str, Message] = {}
    session_id: str | None = None

    async with imodel.stream(
        messages=prepared_msgs,
        stream_chunk_hook=stream_chunk_hook,
        **imodel_kwargs,
    ) as chunks:
        async for chunk in chunks:
            meta = chunk.metadata or {}
            chunk_type = meta.get("type", "")

            if chunk_type == "system":
                session_id = meta.get("session_id")

            elif chunk_type == "thinking":
                if chunk.data:
                    thinking_parts.append(chunk.data)

            elif chunk_type == "text":
                if chunk.data:
                    text_parts.append(chunk.data)

            elif chunk_type == "tool_use":
                # Flush accumulated text as assistant message before tool call
                if text_parts:
                    _add_assistant_msg(session, branch, text_parts, thinking_parts)

                act_req = ActionRequest.create(
                    function=meta.get("tool_name", ""),
                    arguments=meta.get("tool_input", {}),
                )
                req_msg = Message(content=act_req)
                tool_id = meta.get("tool_id")
                if tool_id:
                    pending_requests[tool_id] = req_msg
                session.add_message(req_msg, branches=branch)

            elif chunk_type == "tool_result":
                tool_id = meta.get("tool_id")
                orig_msg = pending_requests.pop(tool_id, None) if tool_id else None
                act_resp = ActionResponse.create(
                    request_id=str(orig_msg.id) if orig_msg else None,
                    result=chunk.data.get("content")
                    if isinstance(chunk.data, dict)
                    else chunk.data,
                    error=None if not meta.get("is_error") else str(chunk.data),
                )
                resp_msg = Message(content=act_resp)
                session.add_message(resp_msg, branches=branch)

            elif chunk_type == "result":
                pass

            elif chunk_type == "done":
                break

    # Capture before flush clears the list
    result_text = "".join(text_parts)

    # Flush remaining text as assistant message
    if text_parts:
        _add_assistant_msg(session, branch, text_parts, thinking_parts)

    # Store session_id for resume
    if session_id:
        imodel.provider_metadata["session_id"] = session_id

    return result_text


def _add_assistant_msg(
    session: Session,
    branch: Branch,
    text_parts: list[str],
    thinking_parts: list[str],
) -> Message:
    """Flush accumulated text into an Assistant message on branch."""
    text = "".join(text_parts)
    from lionagi.beta.resource.backend import Normalized

    resp = Normalized(status="success", data=text)
    content = Assistant.create(response_object=resp)
    msg = Message(content=content)
    if thinking_parts:
        msg.metadata["thinking"] = "\n".join(thinking_parts)
    session.add_message(msg, branches=branch)
    text_parts.clear()
    thinking_parts.clear()
    return msg


async def run_and_collect(
    session: Session,
    branch: Branch | str,
    primary: str,
    context: Any | None = None,
    imodel: iModel | str | None = None,
    stream_chunk_hook: Any | None = None,
    **imodel_kwargs: Any,
) -> str:
    """Convenience: _run that returns collected text. Used as Middle for operate."""
    return await _run(
        session=session,
        branch=branch,
        primary=primary,
        context=context,
        imodel=imodel,
        stream_chunk_hook=stream_chunk_hook,
        **imodel_kwargs,
    )
