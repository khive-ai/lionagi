# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Worker engine that compiles each task wave into a Session.flow() graph."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from lionagi.beta.work.builder import OperationGraphBuilder
from lionagi.ln.concurrency import fail_after, get_cancelled_exc_class

if TYPE_CHECKING:
    from lionagi.beta.session.session import Session

    from .worker import Worker

__all__ = ("WorkerEngine", "WorkerTask")

logger = logging.getLogger(__name__)

PENDING = "PENDING"
PROCESSING = "PROCESSING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
TERMINAL = {COMPLETED, FAILED, CANCELLED}


@dataclass
class WorkerTask:

    id: UUID = field(default_factory=uuid4)
    function: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    status: str = PENDING
    result: Any = None
    error: Exception | None = None
    max_steps: int = 100
    current_step: int = 0
    history: list[tuple[str, Any]] = field(default_factory=list)
    branch_id: UUID | None = None


class WorkerEngine:
    """Drives a Worker by compiling task waves into operation graphs via Session.flow()."""

    def __init__(
        self,
        worker: Worker,
        max_concurrent: int = 10,
        on_step: Callable[[WorkerTask, str, Any], None] | None = None,
        verbose: bool = False,
    ) -> None:
        self.worker = worker
        self.max_concurrent = max_concurrent
        self.on_step = on_step
        self.verbose = verbose
        self.tasks: dict[UUID, WorkerTask] = {}
        self._task_queue: deque[UUID] = deque()
        self._stopped = False
        self._operation_names: dict[str, str] = {}
        self._registered_operation_funcs: dict[str, Callable] = {}
        self._operation_prefix = f"_work_{uuid4().hex}_"

    async def add_task(
        self,
        task_function: str,
        task_max_steps: int = 100,
        **kwargs: Any,
    ) -> WorkerTask:
        if task_function not in self.worker._work_methods:
            raise ValueError(
                f"Method '{task_function}' not found. "
                f"Available: {list(self.worker._work_methods.keys())}"
            )
        task = WorkerTask(
            function=task_function,
            kwargs=kwargs,
            max_steps=task_max_steps,
        )
        self.tasks[task.id] = task
        self._task_queue.append(task.id)
        return task

    async def execute(self) -> None:
        self._stopped = False
        await self._start_worker()
        session = self._ensure_session()
        self._register_worker_operations(session)

        try:
            while not self._stopped and self._task_queue:
                batch = self._drain_pending_batch()
                if not batch:
                    break
                await self._execute_batch(session, batch)
        finally:
            self._unregister_worker_operations(session)

    async def stop(self) -> None:
        self._stopped = True
        if self.worker.session is not None:
            self._unregister_worker_operations(self.worker.session)
        await self._stop_worker()

    async def cancel_task(self, task_id: UUID) -> bool:
        task = self.tasks.get(task_id)
        if task is None or task.status in TERMINAL:
            return False
        task.status = CANCELLED
        return True

    def _ensure_session(self) -> Session:
        if self.worker.session is not None:
            return self.worker.session
        if self._uses_branch_operations():
            raise ValueError(
                "WorkerEngine requires an explicit Session when any @work "
                "method declares operation=..."
            )

        from lionagi.beta.session.session import Session

        self.worker.session = Session()
        return self.worker.session

    def _uses_branch_operations(self) -> bool:
        return any(config.operation for _, config in self.worker._work_methods.values())

    async def _start_worker(self) -> None:
        """Call Worker.start() bypassing the @work dispatch path."""
        from .worker import Worker

        start = self.worker.start
        is_work_method = hasattr(start, "_work_config")
        if getattr(start, "__func__", None) is not Worker.start and not is_work_method:
            await start()
            return
        await Worker.start(self.worker)

    async def _stop_worker(self) -> None:
        """Call Worker.stop() bypassing the @work dispatch path."""
        from .worker import Worker

        stop = self.worker.stop
        is_work_method = hasattr(stop, "_work_config")
        if getattr(stop, "__func__", None) is not Worker.stop and not is_work_method:
            await stop()
            return
        await Worker.stop(self.worker)

    def _register_worker_operations(self, session: Session) -> None:
        for method_name in self.worker._work_methods:
            if method_name in self._operation_names:
                continue
            operation_name = f"{self._operation_prefix}{method_name}"

            async def invoke_work_method(
                params: Any | None = None,
                _ctx: Any | None = None,
                _method_name=method_name,
            ) -> Any:
                method, config = self.worker._work_methods[_method_name]
                kwargs = dict(params or {})
                if config.timeout is None:
                    return await method(**kwargs)
                with fail_after(config.timeout):
                    return await method(**kwargs)

            session.register_operation(operation_name, invoke_work_method, update=False)
            self._operation_names[method_name] = operation_name
            self._registered_operation_funcs[operation_name] = invoke_work_method

    def _unregister_worker_operations(self, session: Session) -> None:
        for operation_name, func in list(self._registered_operation_funcs.items()):
            if hasattr(session, "unregister_operation"):
                session.unregister_operation(operation_name, func=func)
                continue
            registry = getattr(
                getattr(session, "_operation_manager", None), "registry", {}
            )
            if registry.get(operation_name) is func:
                registry.pop(operation_name, None)
        self._registered_operation_funcs.clear()
        self._operation_names.clear()

    def _drain_pending_batch(self) -> list[UUID]:
        batch = []
        while self._task_queue:
            task_id = self._task_queue.popleft()
            task = self.tasks.get(task_id)
            if task is None or task.status in TERMINAL:
                continue
            batch.append(task_id)
        return batch

    async def _execute_batch(self, session: Session, batch: list[UUID]) -> None:
        builder = OperationGraphBuilder("WorkerEngineWave")
        node_to_task: dict[UUID, UUID] = {}

        for task_id in batch:
            task = self.tasks[task_id]
            if task.current_step >= task.max_steps:
                task.status = COMPLETED
                continue

            task.status = PROCESSING
            task.current_step += 1
            operation_name = self._operation_names[task.function]
            call_kwargs = self._call_kwargs_for_task(task)
            branch = self._branch_for_task(session, task)
            if task.branch_id is not None:
                call_kwargs["_lionagi_branch_id"] = str(task.branch_id)
            node_id = builder.add_operation(
                operation_name,
                branch=branch,
                parameters=call_kwargs,
                depends_on=[],
                metadata={
                    "worker_task_id": str(task.id),
                    "worker_function": task.function,
                },
            )
            node_to_task[node_id] = task.id

        if not node_to_task:
            return

        try:
            result = await session.flow(
                builder.get_graph(),
                max_concurrent=self.max_concurrent,
                verbose=self.verbose,
            )
        except get_cancelled_exc_class():
            for task_id in node_to_task.values():
                self.tasks[task_id].status = CANCELLED
            raise

        operation_results = result.get("operation_results", {})
        failed = set(result.get("failed_operations", []))
        cancelled = set(result.get("cancelled_operations", []))

        for node_id, task_id in node_to_task.items():
            task = self.tasks[task_id]
            if node_id in cancelled:
                task.status = CANCELLED
            elif node_id in failed:
                task.status = FAILED
                error_payload = operation_results.get(node_id, {})
                task.error = RuntimeError(str(error_payload.get("error", "")))
            elif node_id in operation_results:
                await self._complete_task_step(task, operation_results[node_id])
            else:
                task.status = FAILED
                task.error = RuntimeError("Worker operation produced no result")

    def _call_kwargs_for_task(self, task: WorkerTask) -> dict[str, Any]:
        _, config = self.worker._work_methods[task.function]
        call_kwargs = dict(task.kwargs)

        if config.assignment:
            from .form import parse_assignment

            inputs, _ = parse_assignment(config.assignment)
            form_id = call_kwargs.get("_form_id")
            if form_id and form_id in self.worker.forms:
                form = self.worker.forms[form_id]
                for input_name in inputs:
                    if input_name in form.available_data:
                        call_kwargs.setdefault(
                            input_name, form.available_data[input_name]
                        )
        return call_kwargs

    def _branch_for_task(self, session: Session, task: WorkerTask):
        _, config = self.worker._work_methods[task.function]
        branch = self.worker.branch
        if not config.operation:
            return branch

        if task.branch_id is not None:
            try:
                return session.get_branch(task.branch_id)
            except Exception:
                logger.warning(
                    "Branch %s not found for task %s; allocating a new branch",
                    task.branch_id,
                    task.id,
                )
                task.branch_id = None

        base_branch = branch or session.default_branch
        if hasattr(base_branch, "clone"):
            branch = base_branch.clone(sender=session.id)
            session.include_branches(branch)
        else:
            branch = base_branch
        task.branch_id = getattr(branch, "id", None)
        return branch

    async def _complete_task_step(self, task: WorkerTask, result: Any) -> None:
        task.history.append((task.function, result))
        task.result = result
        if self.on_step:
            self.on_step(task, task.function, result)

        next_steps = await self._follow_links(task, result)
        if not next_steps:
            task.status = COMPLETED
            return

        first_func, first_kwargs = next_steps[0]
        task.function = first_func
        task.kwargs = first_kwargs
        task.status = PENDING
        self._task_queue.append(task.id)

        for next_func, next_kwargs in next_steps[1:]:
            spawned = WorkerTask(
                function=next_func,
                kwargs=next_kwargs,
                max_steps=max(task.max_steps - task.current_step, 0),
            )
            self.tasks[spawned.id] = spawned
            self._task_queue.append(spawned.id)

    async def _follow_links(
        self,
        task: WorkerTask,
        result: Any,
    ) -> list[tuple[str, dict[str, Any]]]:
        next_steps: list[tuple[str, dict[str, Any]]] = []
        for link in self.worker.get_links_from(task.function):
            try:
                handler = getattr(self.worker, link.handler_name)
                next_kwargs = await handler(result)
            except Exception as exc:
                logger.exception(
                    "Worklink %s -> %s failed for task %s",
                    link.from_,
                    link.to_,
                    task.id,
                )
                task.status = FAILED
                task.error = exc
                return []

            if next_kwargs is None:
                continue
            if not isinstance(next_kwargs, dict):
                exc = TypeError(
                    f"Worklink {link.handler_name!r} must return dict or None"
                )
                task.status = FAILED
                task.error = exc
                return []
            next_steps.append((link.to_, next_kwargs))
        return next_steps

    def get_task(self, task_id: UUID) -> WorkerTask | None:
        return self.tasks.get(task_id)

    @property
    def pending_tasks(self) -> list[WorkerTask]:
        return [task for task in self.tasks.values() if task.status == PENDING]

    @property
    def completed_tasks(self) -> list[WorkerTask]:
        return [task for task in self.tasks.values() if task.status == COMPLETED]

    @property
    def failed_tasks(self) -> list[WorkerTask]:
        return [task for task in self.tasks.values() if task.status == FAILED]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for task in self.tasks.values():
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts

    def __repr__(self) -> str:
        counts = self.status_counts()
        total = len(self.tasks)
        return f"WorkerEngine(worker={self.worker.name}, tasks={total}, {counts})"
