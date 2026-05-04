# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for the work layer compiling into builder/flow."""

import pytest

from lionagi.session.session import Session
from lionagi.work import Form, Report, Worker, WorkerEngine, work, worklink


def test_form_requires_declared_outputs():
    form = Form(assignment="topic -> findings")
    form.fill(topic="LoRA")

    with pytest.raises(ValueError, match="missing fields"):
        form.set_output({"notes": "not the declared output"})

    form.set_output({"findings": "ok"})
    assert form.filled is True
    assert form.get_output_data() == {"findings": "ok"}


def test_form_partial_output_does_not_mark_filled():
    form = Form(assignment="topic -> outline, draft")

    form.set_output({"outline": "bullets"}, partial=True)

    assert form.filled is False
    assert form.missing_outputs() == ["draft"]


def test_report_rejects_duplicate_output_fields():
    report = Report(
        assignment="topic -> paper",
        form_assignments=[
            "researcher: topic -> findings",
            "analyst: topic -> findings",
            "writer: findings -> paper",
        ],
    )
    report.initialize(topic="LoRA")

    with pytest.raises(ValueError, match="Duplicate form output fields"):
        report.validate()


@pytest.mark.asyncio
async def test_report_collects_duplicate_outputs_when_allowed():
    session = Session()
    seen_findings = None

    @session.operation("complete_form")
    async def complete_form(**kwargs):
        nonlocal seen_findings
        assignment = kwargs["form_assignment"]
        inputs = kwargs.get("form_inputs") or {}
        if assignment.startswith("researcher_a"):
            return {"findings": "a"}
        if assignment.startswith("researcher_b"):
            return {"findings": "b"}

        seen_findings = inputs["findings"]
        return {"summary": ",".join(sorted(inputs["findings"]))}

    report = Report(
        assignment="topic -> summary",
        allow_duplicate_outputs=True,
        form_assignments=[
            "researcher_a: topic -> findings",
            "researcher_b: topic -> findings",
            "writer: findings -> summary",
        ],
    )
    report.initialize(topic="LoRA")

    await report.run(session, operation="complete_form", max_concurrent=2)

    assert sorted(seen_findings) == ["a", "b"]
    assert report.get_deliverable() == {"summary": "a,b"}


def test_report_compiles_field_dependencies_to_builder():
    report = Report(
        assignment="topic -> paper",
        form_assignments=[
            "writer: analysis -> paper",
            "researcher: topic -> findings",
            "analyst: findings -> analysis",
        ],
    )
    report.initialize(topic="LoRA")

    builder = report.to_builder(operation="complete_form")
    graph = builder.get_graph()
    node_by_form = report.node_by_form_id

    assert len(node_by_form) == 3
    assert len(graph.internal_nodes) == 3

    form_by_output = {
        form.output_fields[0]: form for form in report.forms if form.output_fields
    }
    writer_node = node_by_form[str(form_by_output["paper"].id)]
    analyst_node = node_by_form[str(form_by_output["analysis"].id)]

    writer_predecessors = {
        predecessor.id
        for predecessor in graph.get_predecessors(graph.internal_nodes[writer_node])
    }
    assert analyst_node in writer_predecessors


def test_report_same_branch_order_inherits_context():
    report = Report(
        assignment="topic -> article",
        form_assignments=[
            "writer: topic -> outline",
            "writer: outline -> article",
        ],
    )
    report.initialize(topic="LoRA")

    builder = report.to_builder(operation="complete_form")
    node_by_form = report.node_by_form_id
    article_form = next(
        form for form in report.forms if "article" in form.output_fields
    )
    article_node = builder.get_graph().internal_nodes[
        node_by_form[str(article_form.id)]
    ]

    assert article_node.metadata["inherit_context"] is True


@pytest.mark.asyncio
async def test_form_inputs_do_not_shadow_operation_parameters():
    session = Session()
    captured = {}

    @session.operation("capture")
    async def capture(**kwargs):
        captured.update(kwargs)
        return {"result": kwargs["form_inputs"]["instruction"]}

    report = Report(
        assignment="instruction -> result",
        form_assignments=["worker: instruction -> result"],
    )
    report.initialize(instruction="user instruction")

    await report.run(session, operation="capture")

    assert captured["instruction"] == "worker: instruction -> result"
    assert captured["form_inputs"] == {"instruction": "user instruction"}


@pytest.mark.asyncio
async def test_report_runs_through_session_flow():
    session = Session()

    @session.operation("complete_form")
    async def complete_form(**kwargs):
        inputs = kwargs.get("form_inputs") or {}
        outputs = kwargs["form_output_fields"]
        source = "|".join(str(inputs[key]) for key in sorted(inputs)) or "root"
        return {field: f"{field}:{source}" for field in outputs}

    report = Report(
        assignment="topic -> paper",
        form_assignments=[
            "researcher: topic -> findings",
            "analyst: findings -> analysis",
            "writer: findings, analysis -> paper",
        ],
    )
    report.initialize(topic="LoRA")

    result = await report.run(session, operation="complete_form", max_concurrent=2)

    assert len(result["completed_operations"]) == 3
    assert report.is_complete()
    deliverable = report.get_deliverable()
    assert set(deliverable) == {"paper"}
    assert "analysis:" in deliverable["paper"]
    assert "findings:" in deliverable["paper"]


@pytest.mark.asyncio
async def test_worker_engine_executes_waves_via_flow():
    class Pipeline(Worker):
        @work("seed -> doubled")
        async def start(self, seed, **kwargs):
            return {"doubled": seed * 2}

        @worklink("start", "finish")
        async def route(self, result):
            return {"doubled": result["doubled"]}

        @work("doubled -> done")
        async def finish(self, doubled, **kwargs):
            return {"done": doubled + 1}

    worker = Pipeline()
    engine = WorkerEngine(worker)
    task = await engine.add_task("start", seed=3)

    await engine.execute()

    assert task.status == "COMPLETED"
    assert task.result == {"done": 7}
    assert [name for name, _ in task.history] == ["start", "finish"]
    assert engine.status_counts() == {"COMPLETED": 1}


@pytest.mark.asyncio
async def test_worker_engine_requires_session_for_branch_operations():
    class Pipeline(Worker):
        @work("prompt -> answer", operation="operate")
        async def ask(self, **kwargs):
            raise AssertionError("operation-backed work should delegate")

    worker = Pipeline()
    engine = WorkerEngine(worker)
    await engine.add_task("ask", prompt="hi")

    with pytest.raises(ValueError, match="requires an explicit Session"):
        await engine.execute()


def test_worker_engine_operation_prefix_is_stable_per_worker():
    class Pipeline(Worker):
        @work("seed -> doubled")
        async def start(self, seed, **kwargs):
            return {"doubled": seed * 2}

    worker = Pipeline()

    assert (
        WorkerEngine(worker)._operation_prefix == WorkerEngine(worker)._operation_prefix
    )
