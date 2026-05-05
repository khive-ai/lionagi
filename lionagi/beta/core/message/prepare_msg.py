# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from lionagi.beta.core.base.pile import Pile
from lionagi.beta.core.base.progression import Progression

from .action import ActionRequest, ActionResponse
from .assistant import Assistant
from .instruction import Instruction
from .role import RoledContent
from .system import System

if TYPE_CHECKING:
    from lionagi.protocols.messages import Message

__all__ = ("prepare_messages_for_chat",)


def _get_text(content: RoledContent, attr: str) -> str:
    """Get text from content attr, returning '' if sentinel."""
    val = getattr(content, attr)
    return "" if content._is_sentinel(val) else val


def _build_context(content: Instruction, action_outputs: list[str]) -> list[JsonValue]:
    """Build context list by appending action outputs to existing context."""
    existing = content.context
    if content._is_sentinel(existing):
        return cast("list[JsonValue]", list(action_outputs))
    return cast("list[JsonValue]", list(cast("list[JsonValue]", existing)) + action_outputs)


def _aggregate_round_actions(
    requests: list[ActionRequest],
    responses: list[ActionResponse],
    round_num: int,
) -> str:
    """Aggregate action request/response pairs into a compact round summary."""
    ts = int(time.time())
    lines = [f"round: {round_num}", f"time: {ts}"]

    if not responses:
        return "\n".join(lines)

    lines.append("actions:")
    for i, resp in enumerate(responses):
        req = requests[i] if i < len(requests) else None
        call_str = req.render_compact() if req else "unknown()"
        status = resp.render_summary()
        lines.append(f"  - {call_str}: {status}")

    return "\n".join(lines)


def _build_round_notification(
    progression: Progression | None,
    round_num: int,
    msg_count: int,
    scratchpad: dict[str, str] | None = None,
) -> str:
    """Build system notification block for agent grounding between rounds."""
    ts = int(time.time())

    capabilities: set[str] = set()
    resources: set[str] = set()

    # Branch is session-layer; import lazily to avoid circular dependency.
    try:
        from lionagi.beta.session.session import Branch  # type: ignore[import]
        if isinstance(progression, Branch):
            capabilities = getattr(progression, "capabilities", set())
            resources = getattr(progression, "resources", set())
    except ImportError:
        pass

    parts = [f'<system round="{round_num}" time="{ts}">']
    if capabilities:
        parts.append(f"  tools: [{', '.join(sorted(capabilities))}]")
    if resources:
        parts.append(f"  resources: [{', '.join(sorted(resources))}]")
    parts.append(f"  context: {msg_count} msgs | round {round_num}")
    if scratchpad:
        scratch_lines = ["  scratchpad:"]
        for k, v in scratchpad.items():
            v_str = str(v)
            scratch_lines.append(f"    {k}: {v_str}")
        parts.extend(scratch_lines)
    parts.append("</system>")
    return "\n".join(parts)


def prepare_messages_for_chat(
    messages: Pile[Message],
    progression: Progression | None = None,
    new_instruction: Message | Instruction | None = None,
    to_chat: bool = True,
    system_prefix: str | None = None,
    aggregate_actions: bool = False,
    round_notifications: bool = False,
    scratchpad: dict[str, str] | None = None,
) -> list[RoledContent] | list[dict[str, Any]]:
    """Prepare messages for chat API with intelligent content organization.

    Algorithm:
    1. Auto-detect system message from first message (if System content)
    1b. Prepend system_prefix if provided (e.g., LNDL format instructions)
    2. Collect action messages and embed into following instruction's context
       - aggregate_actions=True: correlate requests/responses, produce compact summaries
       - aggregate_actions=False: render each ActionResponse individually (legacy)
    3. Merge consecutive AssistantResponses
    4. Embed system into first instruction
    5. Append new_instruction
    """
    # Resolve message sequence — apply progression ordering if provided.
    # Pile supports __getitem__ with a Progression for ordered access.
    to_use: Pile[Message] = messages if progression is None else messages[progression]

    if len(to_use) == 0:
        if new_instruction:
            new_content = (
                new_instruction.content  # type: ignore[union-attr]
                if _is_message(new_instruction)
                else new_instruction
            )
            new_content: Instruction = new_content.with_updates(copy_containers="deep")
            if to_chat:
                chat_msg = {
                    "role": new_content.role.value,
                    "content": new_content.render(
                        new_content.structure_format, new_content.custom_renderer
                    ),
                }
                if chat_msg and chat_msg.get("content"):
                    return [chat_msg]
                return []
            return [new_content]
        return []

    # Phase 1: Extract system message (auto-detect from first message)
    system_text: str | None = None
    start_idx = 0

    first_msg = to_use[0]
    first_content = _get_content(first_msg)
    if isinstance(first_content, System):
        system_text = first_content.render()
        start_idx = 1

    # Phase 1b: Prepend system_prefix (e.g., LNDL prompt)
    if system_prefix:
        system_text = f"{system_prefix}\n\n{system_text}" if system_text else system_prefix

    # Phase 2: Process messages — collect action outputs for next instruction
    _use_msgs: list[RoledContent] = []
    pending_actions: list[str] = []
    pending_requests: list[ActionRequest] = []
    pending_responses: list[ActionResponse] = []
    round_num = 1
    msg_count = len(to_use)

    for i, msg in enumerate(to_use):
        if i < start_idx:
            continue

        content: RoledContent = _get_content(msg)

        if isinstance(content, ActionRequest):
            if aggregate_actions:
                pending_requests.append(content)
            continue

        if isinstance(content, ActionResponse):
            if aggregate_actions:
                pending_responses.append(content)
            else:
                pending_actions.append(content.render())
            continue

        # System in middle: skip
        if isinstance(content, System):
            continue

        # Instruction: embed pending action outputs
        if isinstance(content, Instruction):
            updates: dict[str, Any] = {"tool_schemas": None, "request_model": None}

            if aggregate_actions and pending_responses:
                context_parts: list[str] = []
                if round_notifications:
                    context_parts.append(
                        _build_round_notification(progression, round_num, msg_count, scratchpad)
                    )
                context_parts.append(
                    _aggregate_round_actions(pending_requests, pending_responses, round_num)
                )
                updates["context"] = _build_context(content, context_parts)
                pending_requests.clear()
                pending_responses.clear()
                round_num += 1
            elif pending_actions:
                updates["context"] = _build_context(content, pending_actions)
                pending_actions = []
                round_num += 1

            _use_msgs.append(content.with_updates(copy_containers="deep", **updates))
            continue

        # Other (Assistant, non-aggregated ActionRequest): copy as-is
        _use_msgs.append(content.with_updates(copy_containers="deep"))

    # Phase 3: Merge consecutive AssistantResponses
    if len(_use_msgs) > 1:
        merged: list[RoledContent] = [_use_msgs[0]]
        for content in _use_msgs[1:]:
            if isinstance(content, Assistant) and isinstance(merged[-1], Assistant):
                prev = _get_text(merged[-1], "response")
                curr = _get_text(content, "response")
                merged[-1] = Assistant(response=f"{prev}\n\n{curr}")
            else:
                merged.append(content)
        _use_msgs = merged

    # Phase 4: Embed system message into first instruction
    system_embedded = False
    if system_text:
        if len(_use_msgs) == 0 and new_instruction:
            # No history: embed into new_instruction
            new_content_inner = (
                new_instruction.content  # type: ignore[union-attr]
                if _is_message(new_instruction)
                else new_instruction
            )
            if isinstance(new_content_inner, Instruction):
                curr = _get_text(new_content_inner, "primary")
                system_updates: dict[str, Any] = {"primary": f"{system_text}\n\n{curr}"}
                ctx_parts: list[str] = []
                if aggregate_actions and pending_responses:
                    if round_notifications:
                        ctx_parts.append(
                            _build_round_notification(progression, round_num, msg_count, scratchpad)
                        )
                    ctx_parts.append(
                        _aggregate_round_actions(pending_requests, pending_responses, round_num)
                    )
                    pending_requests.clear()
                    pending_responses.clear()
                elif pending_actions:
                    ctx_parts.extend(pending_actions)
                    pending_actions = []
                if ctx_parts:
                    system_updates["context"] = _build_context(new_content_inner, ctx_parts)
                _use_msgs.append(
                    new_content_inner.with_updates(copy_containers="deep", **system_updates)
                )
                new_instruction = None
                system_embedded = True
        elif _use_msgs and isinstance(_use_msgs[0], Instruction):
            curr = _get_text(_use_msgs[0], "primary")
            _use_msgs[0] = _use_msgs[0].with_updates(primary=f"{system_text}\n\n{curr}")
            system_embedded = True

    # Phase 5: Append new_instruction (with any remaining action outputs)
    if new_instruction:
        final_updates: dict[str, Any] = {}
        new_content_final = (
            new_instruction.content  # type: ignore[union-attr]
            if _is_message(new_instruction)
            else new_instruction
        )
        if isinstance(new_content_final, Instruction):
            context_parts_final: list[str] = []
            if aggregate_actions and pending_responses:
                if round_notifications:
                    context_parts_final.append(
                        _build_round_notification(progression, round_num, msg_count, scratchpad)
                    )
                context_parts_final.append(
                    _aggregate_round_actions(pending_requests, pending_responses, round_num)
                )
                pending_requests.clear()
                pending_responses.clear()
            elif pending_actions:
                context_parts_final.extend(pending_actions)
                pending_actions = []
            if context_parts_final:
                final_updates["context"] = _build_context(new_content_final, context_parts_final)
            if system_text and not system_embedded:
                curr = _get_text(new_content_final, "primary")
                final_updates["primary"] = f"{system_text}\n\n{curr}"
        _use_msgs.append(
            new_content_final.with_updates(copy_containers="deep", **final_updates)
        )

    if to_chat:
        result = []
        for m in _use_msgs:
            data = {}
            if isinstance(m, Instruction):
                data = {
                    "structure_format": m.structure_format,
                    "custom_renderer": m.custom_renderer,
                }
            result.append(
                {
                    "role": m.role.value,
                    "content": m.render(**data),
                }
            )
        return result
    return _use_msgs


def _is_message(obj: Any) -> bool:
    """Check if obj is a session Message (duck-typed, no hard import)."""
    return hasattr(obj, "content") and isinstance(getattr(obj, "content", None), RoledContent)


def _get_content(msg: Any) -> RoledContent:
    """Extract RoledContent from a Message or return as-is if already RoledContent."""
    if isinstance(msg, RoledContent):
        return msg
    return msg.content  # type: ignore[union-attr]
