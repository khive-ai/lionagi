# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Worker: declarative workflow base class with @work and @worklink decorators."""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from lionagi.session.session import Branch, Session

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
    """Configuration for a @work-decorated method."""

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
    """Base class for declarative workflow definition via @work and @worklink."""

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
        seen: set[str] = set()
        for cls in type(self).__mro__:
            for attr_name, descriptor in cls.__dict__.items():
                if attr_name in seen:
                    continue
                seen.add(attr_name)
                if attr_name.startswith("_"):
                    continue

                raw = (
                    descriptor.__func__
                    if isinstance(descriptor, (classmethod, staticmethod))
                    else descriptor
                )
                if not (
                    hasattr(raw, "_work_config")
                    or (hasattr(raw, "_worklink_from") and hasattr(raw, "_worklink_to"))
                ):
                    continue

                attr = (
                    descriptor.__get__(self, type(self))
                    if hasattr(descriptor, "__get__")
                    else descriptor
                )
                if hasattr(raw, "_work_config"):
                    config: WorkConfig = raw._work_config
                    self._work_methods[attr_name] = (attr, config)
                if hasattr(raw, "_worklink_from") and hasattr(raw, "_worklink_to"):
                    link = WorkLink(
                        from_=raw._worklink_from,
                        to_=raw._worklink_to,
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
    """Decorator for typed work methods with DSL assignment and optional branch auto-delegation."""

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
            branch_id = kwargs.pop("_lionagi_branch_id", None)
            target_branch = None
            if config.operation and self_ref is not None:
                if (
                    branch_id is not None
                    and getattr(self_ref, "session", None) is not None
                ):
                    target_branch = self_ref.session.get_branch(branch_id)
                elif hasattr(self_ref, "branch"):
                    target_branch = self_ref.branch

            if config.operation and target_branch is not None:
                meth = target_branch.get_operation(config.operation)
                if meth is None:
                    raise ValueError(f"Branch has no operation '{config.operation}'")

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
    """Decorator for conditional edges between work methods; return None to suppress the edge."""

    def decorator(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._worklink_from = from_
        wrapper._worklink_to = to_
        return wrapper

    return decorator
