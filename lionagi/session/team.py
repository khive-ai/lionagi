# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Team: concurrent multi-agent collaboration via Exchange + ReAct."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from lionagi.tools.communication.messenger import LionMessenger

from .session import Session

logger = logging.getLogger(__name__)

__all__ = ("Team", "AgentState", "TeamMember")


class AgentState(str, Enum):
    ACTIVE = "active"
    DONE = "done"
    FINISHED = "finished"


class TeamMember(BaseModel):
    name: str
    role: str
    branch_id: UUID | None = None
    state: AgentState = AgentState.ACTIVE
    wake_event: Any = Field(default=None, exclude=True)


class Team:
    """Concurrent multi-agent collaboration.

    Pure setup layer: creates branches, binds LionMessenger, provides a
    ``between_rounds`` callback for ReAct message injection, then launches
    concurrent ``branch.ReAct()`` calls. No custom execution loop.

    Usage::

        team = Team(
            goal="Build a web app",
            members={
                "architect": {"role": "Design the system architecture"},
                "implementer": {"role": "Write the code"},
            },
        )
        results = await team.run(instructions={
            "architect": "Design a REST API for a todo app",
            "implementer": "Wait for the architect's design, then implement it",
        })
    """

    def __init__(
        self,
        goal: str,
        members: dict[str, dict[str, Any]],
        session: Session | None = None,
        **session_kwargs,
    ):
        self.goal = goal
        self.session = session or Session(**session_kwargs)
        self.members: dict[str, TeamMember] = {}
        self._name_to_branch: dict[str, UUID] = {}
        self._branch_to_name: dict[UUID, str] = {}
        self.messenger = LionMessenger(exchange=self.session.exchange)

        self.messenger.on("done", self._on_done)
        self.messenger.on("finished", self._on_finished)
        self.messenger.on("wakeup", self._on_wakeup)

        for name, config in members.items():
            branch = self.session.new_branch(name=name)
            member = TeamMember(
                name=name,
                role=config.get("role", name),
                branch_id=branch.id,
                wake_event=asyncio.Event(),
            )
            member.wake_event.set()
            self.members[name] = member
            self._name_to_branch[name] = branch.id
            self._branch_to_name[branch.id] = name

        for name in self.members:
            self._register_messenger(name)

    # ==================== State Callbacks ====================

    def _on_done(self, name: str, sender_id: UUID, reason: str, **_):
        if name in self.members:
            self.members[name].state = AgentState.DONE
            self.members[name].wake_event.clear()

    def _on_finished(self, name: str, sender_id: UUID, reason: str, **_):
        if name in self.members:
            self.members[name].state = AgentState.FINISHED
            self.members[name].wake_event.clear()

    def _on_wakeup(self, name: str, sender_id: UUID, target: str, message: str, **_):
        if target in self.members:
            member = self.members[target]
            if member.state == AgentState.DONE:
                member.state = AgentState.ACTIVE
                member.wake_event.set()

    # ==================== Setup ====================

    def _register_messenger(self, member_name: str):
        member = self.members[member_name]
        branch = self.session.get_branch(member.branch_id)
        roster = {
            n: m.branch_id for n, m in self.members.items() if n != member_name
        }
        tool = self.messenger.bind(
            sender_id=member.branch_id,
            roster=roster,
            sender_name=member_name,
        )
        branch.register_tools(tool)

    def _team_system_prompt(self, member_name: str) -> str:
        member = self.members[member_name]
        teammates = [
            f"- **{n}**: {m.role}"
            for n, m in self.members.items()
            if n != member_name
        ]
        return (
            f"You are **{member_name}** in a team.\n"
            f"**Team goal**: {self.goal}\n"
            f"**Your role**: {member.role}\n\n"
            f"**Your teammates**:\n" + "\n".join(teammates) + "\n\n"
            "Use the `messenger` tool to communicate:\n"
            "- `messenger(action='send', to='name', content='...')` — send a message\n"
            "- `messenger(action='wakeup', to='name', content='...')` — wake a sleeping teammate\n"
            "- `messenger(action='done', content='reason')` — you're done (can be woken)\n"
            "- `messenger(action='finished', content='reason')` — permanently done\n\n"
            "Focus on your role. Collaborate when needed."
        )

    # ==================== Message Injection ====================

    def _collect_unread(self, branch_id: UUID) -> list[dict]:
        messages = self.session.exchange.receive(branch_id)
        if not messages:
            return []

        formatted = []
        for msg in messages:
            sender_name = self._branch_to_name.get(msg.sender, str(msg.sender)[:8])
            self.session.exchange.pop_message(branch_id, msg.sender)
            formatted.append({"from": sender_name, "content": msg.content})
        return formatted

    @staticmethod
    def _format_injection(messages: list[dict]) -> str:
        lines = [
            "[begin lion system notice]",
            "New messages from teammates:",
            "",
        ]
        for msg in messages:
            lines.append(f"**{msg['from']}**: {msg['content']}")
        lines.extend([
            "",
            "[end notice]",
            "Continue your work. Use `messenger(action='done', ...)` when finished.",
        ])
        return "\n".join(lines)

    def _make_between_rounds(self, member_name: str):
        """Create a between_rounds callback for ReActStream."""
        member = self.members[member_name]

        async def between_rounds(branch, round_count) -> str | None:
            await self.session.exchange.sync()
            unread = self._collect_unread(member.branch_id)
            if unread:
                return self._format_injection(unread)
            return None

        return between_rounds

    # ==================== Run ====================

    async def _run_agent(
        self,
        member_name: str,
        instruction: str | None = None,
        max_rounds: int = 10,
        **react_kwargs,
    ) -> Any:
        member = self.members[member_name]
        branch = self.session.get_branch(member.branch_id)

        system_prompt = self._team_system_prompt(member_name)
        agent_instruction = instruction or f"Begin your work as {member_name}."

        result = await branch.ReAct(
            instruct={"instruction": agent_instruction, "guidance": system_prompt},
            tools=True,
            max_extensions=max_rounds,
            extension_allowed=True,
            between_rounds=self._make_between_rounds(member_name),
            **react_kwargs,
        )
        return result

    async def run(
        self,
        instructions: dict[str, str] | None = None,
        max_rounds: int = 10,
        **react_kwargs,
    ) -> dict[str, Any]:
        """Run all team members concurrently via ReAct.

        Args:
            instructions: Per-member instructions keyed by name.
            max_rounds: Maximum ReAct extension rounds per agent.

        Returns:
            Dict mapping member names to their results.
        """
        instructions = instructions or {}

        tasks = {
            name: asyncio.create_task(
                self._run_agent(
                    name,
                    instruction=instructions.get(name),
                    max_rounds=max_rounds,
                    **react_kwargs,
                )
            )
            for name in self.members
        }

        await asyncio.wait(tasks.values(), return_when=asyncio.ALL_COMPLETED)

        results = {}
        for name, task in tasks.items():
            try:
                results[name] = task.result()
            except Exception as e:
                results[name] = {"error": str(e)}
                logger.error(f"Agent {name} failed: {e}")

        return results

    @property
    def active_members(self) -> list[str]:
        return [n for n, m in self.members.items() if m.state == AgentState.ACTIVE]

    @property
    def all_finished(self) -> bool:
        return all(
            m.state in (AgentState.DONE, AgentState.FINISHED)
            for m in self.members.values()
        )
