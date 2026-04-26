# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from lionagi.protocols.action.tool import Tool
from lionagi.service.token_calculator import TokenCalculator

from ..base import LionTool

if TYPE_CHECKING:
    from lionagi.session.branch import Branch


class ContextAction(str, Enum):
    status = "status"
    get_messages = "get_messages"
    evict = "evict"
    evict_action_results = "evict_action_results"


class ContextRequest(BaseModel):
    action: ContextAction = Field(
        ...,
        description=(
            "Action to perform. One of:\n"
            "- 'status': Get context usage — message count by type, estimated tokens.\n"
            "- 'get_messages': List messages with index, role, and content preview. "
            "Use this to decide what to evict.\n"
            "- 'evict': Remove messages by index range. Cannot evict system message (index 0). "
            "Frees context space for new work.\n"
            "- 'evict_action_results': Remove all ActionResponse messages older than "
            "keep_last N turns. Best way to reclaim space from verbose tool outputs."
        ),
    )
    start: int | None = Field(
        None,
        description=(
            "Start index (inclusive, 0-based) for 'evict' and 'get_messages'. "
            "Index 0 is the system message and cannot be evicted."
        ),
    )
    end: int | None = Field(
        None,
        description=(
            "End index (exclusive, 0-based) for 'evict' and 'get_messages'. "
            "If omitted, defaults to end of conversation for get_messages, "
            "or start+1 for evict."
        ),
    )
    keep_last: int | None = Field(
        None,
        description=(
            "For 'evict_action_results': keep the N most recent action result messages, "
            "evict all older ones. Defaults to 5."
        ),
    )


class ContextTool(LionTool):
    is_lion_system_tool = True
    system_tool_name = "context_tool"

    def bind(self, branch: Branch) -> Tool:
        from lionagi.protocols.messages import ActionResponse

        msgs = branch.msgs

        def _message_summary(idx: int, msg) -> str:
            role = msg.role if hasattr(msg, "role") else type(msg).__name__
            content = ""
            if hasattr(msg, "content"):
                c = msg.content
                if isinstance(c, dict):
                    c = str(c)
                elif isinstance(c, list):
                    c = str(c)
                if c:
                    content = c[:120].replace("\n", " ")
                    if len(c) > 120:
                        content += "..."
            return f"[{idx}] {role}: {content}"

        def _estimate_tokens(msg) -> int:
            content = ""
            if hasattr(msg, "content"):
                c = msg.content
                if isinstance(c, str):
                    content = c
                elif c:
                    content = str(c)
            return TokenCalculator.tokenize(content) if content else 0

        async def context_tool(
            action: str,
            start: int = None,
            end: int = None,
            keep_last: int = None,
        ) -> dict:
            """Manage your conversation context — check usage, list messages, evict old ones.

            Use this to stay within context limits during long tasks. Evict verbose
            tool outputs you no longer need to free space for new work.
            """
            progression = msgs.progression
            pile = msgs.messages

            if action == "status":
                total = len(progression)
                by_type: dict[str, int] = {}
                total_tokens = 0
                for uid in progression:
                    if uid in pile:
                        msg = pile[uid]
                        role = msg.role if hasattr(msg, "role") else type(msg).__name__
                        by_type[role] = by_type.get(role, 0) + 1
                        total_tokens += _estimate_tokens(msg)
                return {
                    "success": True,
                    "total_messages": total,
                    "by_type": by_type,
                    "estimated_tokens": total_tokens,
                }

            elif action == "get_messages":
                s = max(0, start or 0)
                e = min(len(progression), end or len(progression))
                summaries = []
                for i in range(s, e):
                    uid = progression[i]
                    if uid in pile:
                        summaries.append(_message_summary(i, pile[uid]))
                return {
                    "success": True,
                    "range": f"[{s}:{e}] of {len(progression)}",
                    "messages": summaries,
                }

            elif action == "evict":
                s = max(1, start or 1)
                e = end if end is not None else s + 1
                e = min(len(progression), e)
                if s >= e:
                    return {"success": False, "error": f"Invalid range [{s}:{e})"}

                uids_to_remove = []
                for i in range(s, e):
                    if i < len(progression):
                        uids_to_remove.append(progression[i])

                removed = 0
                for uid in uids_to_remove:
                    if uid in pile:
                        pile.exclude(uid)
                        removed += 1

                return {
                    "success": True,
                    "removed": removed,
                    "remaining": len(progression),
                }

            elif action == "evict_action_results":
                keep = keep_last if keep_last is not None else 5
                action_result_indices = []
                for i, uid in enumerate(progression):
                    if uid in pile and isinstance(pile[uid], ActionResponse):
                        action_result_indices.append((i, uid))

                if len(action_result_indices) <= keep:
                    return {
                        "success": True,
                        "removed": 0,
                        "message": f"Only {len(action_result_indices)} action results, keeping all.",
                    }

                to_evict = (
                    action_result_indices[:-keep] if keep > 0 else action_result_indices
                )
                removed = 0
                for _, uid in to_evict:
                    if uid in pile:
                        pile.exclude(uid)
                        removed += 1

                return {
                    "success": True,
                    "removed": removed,
                    "remaining": len(progression),
                }

            return {"success": False, "error": f"Unknown action: {action}"}

        return Tool(func_callable=context_tool, request_options=ContextRequest)

    def to_tool(self) -> Tool:
        raise NotImplementedError(
            "ContextTool requires branch context. Use context_tool.bind(branch) instead."
        )
