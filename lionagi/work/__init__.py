# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Work system - Declarative workflow orchestration.

Two complementary patterns:

**Worker** (execution capability):
    Declarative workflow via @work and @worklink decorators.
    Optional Branch backing for auto-delegation to branch.operate().

    class MyPipeline(Worker):
        @work(assignment="writer: topic -> draft", operation="operate")
        async def write(self, topic="", **kw):
            pass  # auto-delegates to branch.operate()

**Report** (artifact state):
    Multi-step workflow orchestration via form_assignments DSL.
    Tracks one job's progress through a form-based DAG.

    report = Report(
        assignment="input -> output",
        form_assignments=["input -> intermediate", "intermediate -> output"],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "WorkerEngine": ("lionagi.work.engine", "WorkerEngine"),
    "WorkerTask": ("lionagi.work.engine", "WorkerTask"),
    "Form": ("lionagi.work.form", "Form"),
    "ParsedAssignment": ("lionagi.work.form", "ParsedAssignment"),
    "parse_assignment": ("lionagi.work.form", "parse_assignment"),
    "parse_full_assignment": ("lionagi.work.form", "parse_full_assignment"),
    "Report": ("lionagi.work.report", "Report"),
    "Worker": ("lionagi.work.worker", "Worker"),
    "WorkConfig": ("lionagi.work.worker", "WorkConfig"),
    "WorkLink": ("lionagi.work.worker", "WorkLink"),
    "work": ("lionagi.work.worker", "work"),
    "worklink": ("lionagi.work.worker", "worklink"),
}

_LOADED: dict[str, object] = {}


def __getattr__(name: str) -> object:
    if name in _LOADED:
        return _LOADED[name]
    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value
    raise AttributeError(f"module 'lionagi.work' has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)


if TYPE_CHECKING:
    from lionagi.work.engine import WorkerEngine, WorkerTask
    from lionagi.work.form import (
        Form,
        ParsedAssignment,
        parse_assignment,
        parse_full_assignment,
    )
    from lionagi.work.report import Report
    from lionagi.work.worker import WorkConfig, Worker, WorkLink, work, worklink

__all__ = (
    "Form",
    "ParsedAssignment",
    "Report",
    "WorkConfig",
    "WorkLink",
    "Worker",
    "WorkerEngine",
    "WorkerTask",
    "parse_assignment",
    "parse_full_assignment",
    "work",
    "worklink",
)
