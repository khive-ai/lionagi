# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Team: orchestrator-driven concurrent multi-agent collaboration."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from lionagi.tools.communication.messenger import LionMessenger

from .branch import Branch
from .session import Session

logger = logging.getLogger(__name__)

__all__ = ("Team",)


class FanoutInstruction(BaseModel):
    """Structured output from orchestrator's fanout operate call."""

    assignments: dict[str, str] = Field(
        description="Mapping of member branch names to their instructions."
    )


class Team:
    """Orchestrator-driven concurrent multi-agent collaboration.

    The orchestrator branch runs an ``operate()`` to produce structured
    fanout instructions, then member branches run ``ReAct()`` concurrently
    with ``LionMessenger`` for inter-agent communication.

    Usage::

        session = Session()

        orchestrator = session.new_branch(
            name="orchestrator",
            system="You coordinate a team of agents...",
        )
        researcher = session.new_branch(
            name="researcher",
            system="You are a research specialist...",
        )
        implementer = session.new_branch(
            name="implementer",
            system="You are a code implementer...",
        )

        team = Team(
            task="Build a REST API for a todo app",
            orchestrator=orchestrator,
            members=[researcher, implementer],
            session=session,
        )
        results = await team.run()
    """

    def __init__(
        self,
        task: str,
        orchestrator: Branch,
        members: list[Branch],
        session: Session,
    ):
        self.task = task
        self.session = session
        self.orchestrator = orchestrator
        self.members = {b.name: b for b in members}
        self.messenger = LionMessenger(exchange=session.exchange)

        all_branches = {"orchestrator": orchestrator, **self.members}
        self._name_to_id: dict[str, UUID] = {
            n: b.id for n, b in all_branches.items()
        }
        self._id_to_name: dict[UUID, str] = {
            b.id: n for n, b in all_branches.items()
        }

        for name, branch in all_branches.items():
            roster = {n: bid for n, bid in self._name_to_id.items() if n != name}
            tool = self.messenger.bind(
                sender_id=branch.id,
                roster=roster,
                sender_name=name,
            )
            branch.register_tools(tool)

    def _collect_unread(self, branch_id: UUID) -> list[dict]:
        messages = self.session.exchange.receive(branch_id)
        if not messages:
            return []

        formatted = []
        for msg in messages:
            sender_name = self._id_to_name.get(msg.sender, str(msg.sender)[:8])
            self.session.exchange.pop_message(branch_id, msg.sender)
            formatted.append({"from": sender_name, "content": msg.content})
        return formatted

    @staticmethod
    def _format_injection(messages: list[dict]) -> str:
        lines = [
            "[begin team message]",
            "",
        ]
        for msg in messages:
            lines.append(f"**{msg['from']}**: {msg['content']}")
        lines.extend([
            "",
            "[end team message]",
        ])
        return "\n".join(lines)

    def _make_between_rounds(self, branch_id: UUID):
        async def between_rounds(branch, round_count) -> str | None:
            await self.session.exchange.sync()
            unread = self._collect_unread(branch_id)
            if unread:
                return self._format_injection(unread)
            return None
        return between_rounds

    async def run(
        self,
        fanout_instruction: str | None = None,
        max_rounds: int = 10,
        **react_kwargs,
    ) -> dict[str, Any]:
        """Run the team: orchestrator fans out, members execute concurrently.

        Args:
            fanout_instruction: Override instruction for the orchestrator's
                fanout call. If None, uses self.task with member names.
            max_rounds: Maximum ReAct extension rounds per member.

        Returns:
            Dict with 'orchestrator' (fanout plan) and per-member results.
        """
        member_names = list(self.members.keys())

        prompt = fanout_instruction or (
            f"Task: {self.task}\n\n"
            f"You have these team members: {', '.join(member_names)}.\n"
            f"Assign each member a specific instruction for their part of the work.\n"
            f"Return assignments as a mapping of member name to instruction."
        )

        fanout = await self.orchestrator.operate(
            instruction=prompt,
            response_format=FanoutInstruction,
        )

        if isinstance(fanout, FanoutInstruction):
            assignments = fanout.assignments
        elif isinstance(fanout, dict):
            assignments = fanout.get("assignments", {})
        else:
            assignments = {n: f"Work on: {self.task}" for n in member_names}

        tasks = {}
        for name, branch in self.members.items():
            instruction = assignments.get(name, f"Work on your part of: {self.task}")
            tasks[name] = asyncio.create_task(
                branch.ReAct(
                    instruct={"instruction": instruction},
                    tools=True,
                    max_extensions=max_rounds,
                    extension_allowed=True,
                    between_rounds=self._make_between_rounds(branch.id),
                    **react_kwargs,
                )
            )

        await asyncio.wait(tasks.values(), return_when=asyncio.ALL_COMPLETED)

        results = {"orchestrator": assignments}
        for name, task in tasks.items():
            try:
                results[name] = task.result()
            except Exception as e:
                results[name] = {"error": str(e)}
                logger.error(f"Agent {name} failed: {e}")

        return results
