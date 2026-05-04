# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Report state and graph compilation for form-based work.

``Report`` owns the artifact state for one job. It validates form wiring,
resolves scoped inputs, and compiles forms into an ``OperationGraphBuilder``.
It does not run a second scheduler; execution is delegated to ``Session.flow()``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import Field, PrivateAttr

from lionagi.protocols.generic.element import Element

from .form import Form, parse_assignment

if TYPE_CHECKING:
    from lionagi.operations.builder import OperationGraphBuilder
    from lionagi.session.branch import Branch
    from lionagi.session.session import Session

__all__ = ("Report",)


class Report(Element):
    """Artifact state for a multi-form workflow.

    Form dependencies are inferred from declared input/output fields. Same-branch
    form order is also compiled into graph edges so branch conversation memory
    stays deterministic, but the Report itself does not keep a branch scheduler.
    """

    assignment: str = Field(default="")
    form_assignments: list[str] = Field(default_factory=list)
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    forms: list[Form] = Field(default_factory=list)

    _completed_ids: set[str] = PrivateAttr(default_factory=set)
    _initial_inputs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _node_by_form_id: dict[str, UUID] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _: Any) -> None:
        if self.assignment and not self.input_fields and not self.output_fields:
            self.input_fields, self.output_fields = parse_assignment(self.assignment)

        existing = {form.assignment for form in self.forms}
        for assignment in self.form_assignments:
            if assignment not in existing:
                self.forms.append(Form(assignment=assignment))
                existing.add(assignment)

    @property
    def initial_inputs(self) -> dict[str, Any]:
        """Initial job inputs supplied through ``initialize()``."""
        return dict(self._initial_inputs)

    @property
    def node_by_form_id(self) -> dict[str, UUID]:
        """Mapping populated when this report is compiled to a builder graph."""
        return dict(self._node_by_form_id)

    def initialize(self, **inputs: Any) -> None:
        """Set report-level inputs required by the top-level assignment."""
        missing = [field for field in self.input_fields if field not in inputs]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")
        self._initial_inputs = dict(inputs)

    def validate(self) -> None:
        """Validate field wiring before compiling or executing the report."""
        producer_by_field = self._producer_by_field()

        for field in self.output_fields:
            if field not in self._initial_inputs and field not in producer_by_field:
                raise ValueError(f"Report output field '{field}' has no producer")

        available = set(self._initial_inputs) | set(producer_by_field)
        for form in self._topological_forms():
            missing = [field for field in form.input_fields if field not in available]
            if missing:
                raise ValueError(
                    f"Form '{form.assignment}' has unresolved inputs: {missing}"
                )

        self._assert_acyclic()

    def resolve_inputs(self, form: Form) -> dict[str, Any]:
        """Resolve a form's declared inputs from completed producers or initial data."""
        resolved = {}
        for field in form.input_fields:
            producer = self._find_completed_producer(field)
            if producer is not None:
                data = producer.get_output_data()
                if field in data:
                    resolved[field] = data[field]
            elif field in self._initial_inputs:
                resolved[field] = self._initial_inputs[field]
        return resolved

    def cross_branch_inputs(self, form: Form) -> dict[str, Any]:
        """Resolve only declared inputs produced by other branches or initial input."""
        my_branch = form.branch or "_default"
        cross = {}
        for field in form.input_fields:
            producer = self._find_completed_producer(field)
            if producer is not None:
                producer_branch = producer.branch or "_default"
                if producer_branch != my_branch:
                    data = producer.get_output_data()
                    if field in data:
                        cross[field] = data[field]
            elif field in self._initial_inputs:
                cross[field] = self._initial_inputs[field]
        return cross

    def next_forms(self) -> list[Form]:
        """Return currently ready forms without performing scheduling.

        This method is retained for manual orchestration. The preferred path is
        ``to_builder()`` or ``run()``, which executes through ``Session.flow()``.
        """
        ready = []
        for form in self._topological_forms():
            if form.filled:
                continue
            resolved = self.resolve_inputs(form)
            if len(resolved) == len(form.input_fields):
                form.fill(**resolved)
                ready.append(form)
        return ready

    def complete_form(self, form: Form) -> None:
        """Record a filled form as completed."""
        if not form.filled:
            raise ValueError("Form is not filled")
        if form not in self.forms:
            raise ValueError("Form does not belong to this report")
        self._completed_ids.add(str(form.id))

    def form_dependencies(
        self,
        form: Form,
        *,
        include_branch_order: bool = True,
    ) -> list[Form]:
        """Return forms that must complete before ``form``."""
        if form not in self.forms:
            raise ValueError("Form does not belong to this report")

        producer_by_field = self._producer_by_field()
        deps: list[Form] = []
        for field in form.input_fields:
            producer = producer_by_field.get(field)
            if producer is not None and producer is not form and producer not in deps:
                deps.append(producer)

        if include_branch_order:
            previous = self._previous_same_branch_form(form)
            if previous is not None and previous not in deps:
                deps.append(previous)
        return deps

    def to_builder(
        self,
        *,
        operation: str = "operate",
        builder: OperationGraphBuilder | None = None,
        branches: Mapping[str, Branch | UUID | str] | None = None,
        parameter_factory: Callable[[Form, dict[str, Any]], dict[str, Any]]
        | None = None,
    ) -> OperationGraphBuilder:
        """Compile this report into an ``OperationGraphBuilder``.

        Args:
            operation: Branch operation name used for every form node.
            builder: Optional existing builder to append to.
            branches: Optional mapping from form branch names to Branch objects
                or IDs.
            parameter_factory: Optional callback that receives the Form and its
                base parameters and returns operation parameters.
        """
        from lionagi.operations.builder import OperationGraphBuilder

        self.validate()
        builder = builder or OperationGraphBuilder(name=f"Report:{self.id}")
        node_by_form: dict[str, UUID] = {}

        for form in self._topological_forms():
            previous = self._previous_same_branch_form(form)
            dep_forms = self.form_dependencies(form)
            if previous in dep_forms:
                dep_forms = [
                    previous,
                    *(dep for dep in dep_forms if dep is not previous),
                ]
            deps = [
                node_by_form[str(dep.id)]
                for dep in dep_forms
                if str(dep.id) in node_by_form
            ]
            branch_ref = self._resolve_branch_ref(form, branches)
            params = self._operation_parameters(form)
            if parameter_factory is not None:
                params = parameter_factory(form, params)

            node_id = builder.add_form(
                form,
                operation=operation,
                depends_on=deps,
                inherit_context=previous is not None,
                branch=branch_ref,
                parameters=params,
            )
            node_by_form[str(form.id)] = node_id
            form.metadata["operation_node_id"] = str(node_id)

        self._node_by_form_id = node_by_form
        return builder

    async def run(
        self,
        session: Session,
        *,
        operation: str = "operate",
        branches: Mapping[str, Branch | UUID | str] | None = None,
        max_concurrent: int = 5,
        verbose: bool = False,
        **flow_kwargs: Any,
    ) -> dict[str, Any]:
        """Compile and execute this report through ``Session.flow()``."""
        builder = self.to_builder(operation=operation, branches=branches)
        result = await session.flow(
            builder.get_graph(),
            context=self._initial_inputs,
            max_concurrent=max_concurrent,
            verbose=verbose,
            **flow_kwargs,
        )
        self.apply_flow_result(result)
        return result

    def apply_flow_result(self, result: dict[str, Any]) -> None:
        """Update forms from a ``Session.flow()`` result."""
        operation_results = result.get("operation_results", {})
        failed = set(result.get("failed_operations", []))
        skipped = set(result.get("skipped_operations", []))
        for form in self.forms:
            node_id = self._node_by_form_id.get(str(form.id))
            if node_id is None:
                continue
            if node_id in failed or node_id in skipped:
                continue
            if node_id in operation_results:
                form.set_output(operation_results[node_id])
                self.complete_form(form)

    def is_complete(self) -> bool:
        return all(field in self.get_deliverable() for field in self.output_fields)

    def get_deliverable(self) -> dict[str, Any]:
        result = {}
        for field in self.output_fields:
            producer = self._find_completed_producer(field)
            if producer is not None:
                data = producer.get_output_data()
                if field in data:
                    result[field] = data[field]
            elif field in self._initial_inputs:
                result[field] = self._initial_inputs[field]
        return result

    @property
    def completed_forms(self) -> list[Form]:
        return [form for form in self.forms if str(form.id) in self._completed_ids]

    @property
    def progress(self) -> tuple[int, int]:
        return len(self._completed_ids), len(self.forms)

    @property
    def branches(self) -> list[str]:
        seen = []
        for form in self.forms:
            branch = form.branch or "_default"
            if branch not in seen:
                seen.append(branch)
        return seen

    def _operation_parameters(self, form: Form) -> dict[str, Any]:
        return {
            "instruction": form.assignment,
            "form_id": str(form.id),
            "form_assignment": form.assignment,
            "form_inputs": form.get_inputs(),
            "form_input_fields": list(form.input_fields),
            "form_output_fields": list(form.output_fields),
            "form_resource": form.resource,
        }

    def _resolve_branch_ref(
        self,
        form: Form,
        branches: Mapping[str, Branch | UUID | str] | None,
    ) -> Branch | UUID | str | None:
        if not branches or not form.branch:
            return None
        return branches.get(form.branch)

    def _producer_by_field(self) -> dict[str, Form]:
        producer_by_field = {}
        duplicate_fields = defaultdict(list)
        for form in self.forms:
            for field in form.output_fields:
                if field in producer_by_field:
                    duplicate_fields[field].extend(
                        [producer_by_field[field].assignment, form.assignment]
                    )
                else:
                    producer_by_field[field] = form
        if duplicate_fields:
            details = {
                field: sorted(set(assignments))
                for field, assignments in duplicate_fields.items()
            }
            raise ValueError(f"Duplicate form output fields: {details}")
        return producer_by_field

    def _find_completed_producer(self, field: str) -> Form | None:
        producer = self._producer_by_field().get(field)
        if producer is not None and str(producer.id) in self._completed_ids:
            return producer
        return None

    def _previous_same_branch_form(self, form: Form) -> Form | None:
        idx = self.forms.index(form)
        branch = form.branch or "_default"
        for prior in reversed(self.forms[:idx]):
            if (prior.branch or "_default") == branch:
                return prior
        return None

    def _assert_acyclic(self) -> None:
        self._topological_forms()

    def _topological_forms(self) -> list[Form]:
        ordered: list[Form] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(form: Form) -> None:
            form_id = str(form.id)
            if form_id in visited:
                return
            if form_id in visiting:
                raise ValueError("Report form dependencies contain a cycle")
            visiting.add(form_id)
            for dep in self.form_dependencies(form):
                visit(dep)
            visiting.remove(form_id)
            visited.add(form_id)
            ordered.append(form)

        for form in self.forms:
            visit(form)
        return ordered

    def __repr__(self) -> str:
        completed, total = self.progress
        return f"Report('{self.assignment}', {completed}/{total} forms)"
