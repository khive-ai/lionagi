# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Worker - Declarative workflow definition via decorated methods.

A Worker defines workflows through:
- @work: Typed operations with assignment DSL (inputs -> outputs)
- @worklink: Conditional edges between work methods

Workers compose with Session for Branch-backed execution, where @work methods
with ``operation=...`` auto-delegate to the active Branch operation.

The Worker pattern and the Flow/Builder pattern share the same substrate:
graph-based stateful execution. Worker+Engine is the imperative interface;
Flow+Builder is the declarative interface. Both execute on the same
graph/operation primitives.

Example:
    class ResearchPipeline(Worker):
        @work(assignment="topic -> findings")
        async def survey(self, topic="", **kwargs):
            return {"findings": await self.session.default_branch.operate(...)}

        @work(assignment="findings -> analysis")
        async def analyze(self, findings="", **kwargs):
            return {"analysis": ...}

        @worklink(from_="survey", to_="analyze")
        async def survey_to_analyze(self, from_result):
            return from_result

        @worklink(from_="survey", to_="survey")
        async def deepen(self, from_result):
            questions = from_result.get("open_questions", [])
            novel = [q for q in questions if q.get("novelty", 0) > 0.7]
            if novel:
                return {"topic": novel[0]["question"]}
            return None  # no re-entry
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from lionagi.session.branch import Branch
    from lionagi.session.session import Session

    from .form import Form

__all__ = (
    "Worker",
    "WorkConfig",
    "WorkLink",
    "work",
    "worklink",
)


@dataclass
class WorkConfig:
    """Configuration for a @work decorated method.

    Attributes:
        assignment: DSL string 'inputs -> outputs' for typed I/O
        operation: Branch operation to auto-delegate to ('operate', 'communicate', 'ReAct')
        branch: Optional branch routing hint.
        resource: Optional resource hint.
        timeout: Max execution time in seconds
    """

    assignment: str = ""
    operation: str = ""
    branch: str | None = None
    resource: str | None = None
    timeout: float | None = None


@dataclass
class WorkLink:
    """Edge definition between work methods."""

    from_: str
    to_: str
    handler_name: str


class Worker:
    """Base class for declarative workflow definition.

    Subclass and decorate methods with @work and @worklink to define workflows.
    Set `session` for Session-backed execution where @work methods with
    `operation` auto-delegate to the session's default branch.

    Attributes:
        name: Worker name (default: class name)
        session: Optional Session for model-backed execution
        forms: Dict mapping form IDs to Form instances
    """

    name: str = "worker"

    def __init__(self, session: Session | None = None) -> None:
        self.session = session
        self.forms: dict[str | UUID, Form] = {}
        self._work_methods: dict[str, tuple[Callable, WorkConfig]] = {}
        self._work_links: list[WorkLink] = []
        self._operation_namespace = uuid4().hex
        self._stopped = False
        self._collect_work_metadata()

    @property
    def branch(self) -> Branch | None:
        if self.session is not None:
            return self.session.default_branch
        return None

    def _collect_work_metadata(self) -> None:
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name, None)
            if attr is None:
                continue
            if hasattr(attr, "_work_config"):
                config: WorkConfig = attr._work_config
                self._work_methods[attr_name] = (attr, config)
            if hasattr(attr, "_worklink_from") and hasattr(attr, "_worklink_to"):
                link = WorkLink(
                    from_=attr._worklink_from,
                    to_=attr._worklink_to,
                    handler_name=attr_name,
                )
                self._work_links.append(link)

    def get_links_from(self, method_name: str) -> list[WorkLink]:
        return [link for link in self._work_links if link.from_ == method_name]

    def get_links_to(self, method_name: str) -> list[WorkLink]:
        return [link for link in self._work_links if link.to_ == method_name]

    async def stop(self) -> None:
        self._stopped = True

    async def start(self) -> None:
        self._stopped = False

    def is_stopped(self) -> bool:
        return self._stopped

    def __repr__(self) -> str:
        methods = list(self._work_methods.keys())
        links = len(self._work_links)
        forms = len(self.forms)
        backed = "session-backed" if self.session else "standalone"
        return (
            f"{self.__class__.__name__}({backed}, methods={methods}, "
            f"links={links}, forms={forms})"
        )


def work(
    assignment: str = "",
    *,
    operation: str = "",
    branch: str | None = None,
    role: str | None = None,
    resource: str | None = None,
    timeout: float | None = None,
) -> Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]]:
    """Decorator for typed work methods.

    Args:
        assignment: DSL string 'inputs -> outputs' defining typed I/O.
        operation: Branch operation name ('operate', 'communicate', 'ReAct').
            When set on a session-backed Worker, the method body is ignored
            and execution auto-delegates to the session's default branch.
        branch: Optional branch routing hint for compilers.
        role: Alias for ``branch``.
        resource: Optional resource/capability hint for compilers.
        timeout: Max execution time in seconds.

    Example:
        @work(assignment="context -> code")
        async def write_code(self, context="", **kwargs):
            return {"code": await some_llm_call(context)}

        @work(assignment="question -> findings", operation="ReAct")
        async def research(self, **kwargs):
            pass  # auto-delegates to branch.ReAct()
    """

    def decorator(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        parsed_branch = branch or role
        parsed_resource = resource
        if assignment:
            from .form import parse_full_assignment

            parsed = parse_full_assignment(assignment)
            parsed_branch = parsed_branch or parsed.branch
            parsed_resource = parsed_resource or parsed.resource

        config = WorkConfig(
            assignment=assignment,
            operation=operation,
            branch=parsed_branch,
            resource=parsed_resource,
            timeout=timeout,
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            self_ref = args[0] if args else None
            if (
                config.operation
                and self_ref is not None
                and hasattr(self_ref, "branch")
                and self_ref.branch is not None
            ):
                target_branch = self_ref.branch
                meth = target_branch.get_operation(config.operation)
                if meth is None:
                    raise ValueError(
                        f"Branch has no operation '{config.operation}'"
                    )

                instruction = kwargs.pop("instruction", None)
                if instruction is None and len(args) > 1:
                    instruction = args[1]
                if instruction is None:
                    instruction = ""

                result = await meth(instruction=instruction, **kwargs)

                if config.assignment:
                    from .form import parse_assignment

                    _, outputs = parse_assignment(config.assignment)
                    if len(outputs) == 1 and not isinstance(result, dict):
                        return {outputs[0]: result}
                if isinstance(result, dict):
                    return result
                return {"result": result}

            return await func(*args, **kwargs)

        wrapper._work_config = config
        return wrapper

    return decorator


def worklink(
    from_: str,
    to_: str,
) -> Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]]:
    """Decorator for conditional edges between work methods.

    The decorated function receives the result from the 'from_' method
    and returns kwargs dict for the 'to_' method. Return None to skip
    the edge (conditional routing).

    Example:
        @worklink(from_="write_code", to_="review")
        async def write_to_review(self, from_result):
            return from_result

        @worklink(from_="review", to_="write_code")
        async def review_to_rewrite(self, from_result):
            if not from_result.get("approved"):
                return {"instruction": "Fix: " + from_result["feedback"]}
            return None  # approved, don't retry
    """

    def decorator(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._worklink_from = from_
        wrapper._worklink_to = to_
        return wrapper

    return decorator
