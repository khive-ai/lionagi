# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Report: artifact state container that compiles Forms into a Session.flow() graph."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import Field, PrivateAttr, SkipValidation, field_validator

from lionagi.beta.resource.pile import Pile
from lionagi.protocols.generic.element import Element

from .form import Form, parse_assignment

if TYPE_CHECKING:
    from lionagi.beta.session.session import Branch, Session
    from lionagi.beta.work.builder import OperationGraphBuilder

__all__ = ("Report",)

_MISSING = object()


def _coerce_form_item(item: Any) -> Form:
    if isinstance(item, Form):
        return item
    if isinstance(item, dict):
        return Form.from_dict(item)
    return item


def _forms_pile(value: Any = None) -> Pile[Form]:
    if value is None:
        return Pile(item_type={Form}, strict_type=False)
    if isinstance(value, Pile):
        return Pile(
            collections=list(value),
            item_type={Form},
            strict_type=False,
        )
    if isinstance(value, dict) and "collections" in value:
        collections = value.get("collections") or []
        if isinstance(collections, dict):
            collections = list(collections.values())
        order = value.get("order", value.get("progression"))
        return Pile(
            collections=[_coerce_form_item(item) for item in collections],
            item_type={Form},
            order=order,
            strict_type=False,
        )

    collections = list(value) if isinstance(value, (list, tuple)) else [value]
    return Pile(
        collections=[_coerce_form_item(item) for item in collections],
        item_type={Form},
        strict_type=False,
    )


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
    forms: SkipValidation[Pile[Form]] = Field(default_factory=_forms_pile)
    allow_duplicate_outputs: bool = Field(
        default=False,
        description=(
            "Allow multiple forms to produce the same output field. Consumers "
            "of duplicate fields receive a list in form_inputs."
        ),
    )

    _completed_ids: set[str] = PrivateAttr(default_factory=set)
    _initial_inputs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _node_by_form_id: dict[str, UUID] = PrivateAttr(default_factory=dict)
    _producer_cache: dict[str, list[Form]] = PrivateAttr(default_factory=dict)
    _producer_cache_fingerprint: tuple | None = PrivateAttr(default=None)
    _topological_cache: list[Form] = PrivateAttr(default_factory=list)
    _topological_cache_fingerprint: tuple | None = PrivateAttr(default=None)

    @field_validator("forms", mode="before")
    @classmethod
    def _validate_forms(cls, value: Any) -> Pile[Form]:
        return _forms_pile(value)

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
        missing = [field for field in self.input_fields if field not in inputs]
        if missing:
            raise ValueError(f"Missing required input fields: {missing}")
        self._initial_inputs = dict(inputs)

    def validate(self, external_input_fields: set[str] | None = None) -> None:
        external_input_fields = external_input_fields or set()
        producer_by_field = self._producer_map()

        for field in self.output_fields:
            if field not in self._initial_inputs and field not in producer_by_field:
                raise ValueError(f"Report output field '{field}' has no producer")

        available = set(self._initial_inputs) | set(producer_by_field)
        available.update(external_input_fields)
        for form in self._topological_forms():
            missing = [field for field in form.input_fields if field not in available]
            if missing:
                raise ValueError(
                    f"Form '{form.assignment}' has unresolved inputs: {missing}"
                )

        self._assert_acyclic()

    def resolve_inputs(self, form: Form, *, strict: bool = False) -> dict[str, Any]:
        resolved = {}
        for field in form.input_fields:
            value = self._resolved_field_value(field, consumer=form)
            if value is not _MISSING:
                resolved[field] = value
        if strict and len(resolved) != len(form.input_fields):
            missing = [field for field in form.input_fields if field not in resolved]
            raise ValueError(
                f"Form '{form.assignment}' has unresolved inputs: {missing}"
            )
        return resolved

    def cross_branch_inputs(self, form: Form) -> dict[str, Any]:
        """Resolve only declared inputs produced by other branches or initial input."""
        cross = {}
        for field in form.input_fields:
            value = self._resolved_field_value(
                field,
                consumer=form,
                cross_branch_only=True,
            )
            if value is not _MISSING:
                cross[field] = value
        return cross

    def next_forms(self) -> list[Form]:
        """Return ready forms; retained for manual orchestration (prefer to_builder()/run())."""
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

        producer_by_field = self._producer_map()
        deps: list[Form] = []
        for field in form.input_fields:
            for producer in producer_by_field.get(field, []):
                if producer is form or producer in deps:
                    continue
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
        parameter_factory: (
            Callable[[Form, dict[str, Any]], dict[str, Any]] | None
        ) = None,
    ) -> OperationGraphBuilder:
        from lionagi.beta.work.builder import OperationGraphBuilder

        builder = builder or OperationGraphBuilder(name=f"Report:{self.id}")
        external_producers = self._external_form_producers(builder)
        self.validate(external_input_fields=set(external_producers))
        node_by_form: dict[str, UUID] = {}

        for form in self._topological_forms():
            previous = self._previous_same_branch_form(form)
            dep_forms = self.form_dependencies(form)
            if previous in dep_forms:
                dep_forms = [
                    previous,
                    *(dep for dep in dep_forms if dep is not previous),
                ]
            deps = self._external_dependencies(form, external_producers)
            deps.extend(
                node_by_form[str(dep.id)]
                for dep in dep_forms
                if str(dep.id) in node_by_form and node_by_form[str(dep.id)] not in deps
            )
            branch_ref = self._resolve_branch_ref(form, branches)
            params = self._operation_parameters(form)
            params["form_collect_input_fields"] = self._collect_input_fields(form)
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
                form.set_output(operation_results[node_id], partial=True)
                if form.filled:
                    self.complete_form(form)

    def is_complete(self) -> bool:
        return all(field in self.get_deliverable() for field in self.output_fields)

    def get_deliverable(self) -> dict[str, Any]:
        result = {}
        for field in self.output_fields:
            value = self._resolved_field_value(field)
            if value is not _MISSING:
                result[field] = value
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

    def _producer_map(self) -> dict[str, list[Form]]:
        fingerprint = self._forms_fingerprint(include_inputs=False)
        if fingerprint != self._producer_cache_fingerprint:
            producer_by_field: dict[str, list[Form]] = defaultdict(list)
            for form in self.forms:
                for field in form.output_fields:
                    producer_by_field[field].append(form)
            self._producer_cache = dict(producer_by_field)
            self._producer_cache_fingerprint = fingerprint
            self._validate_duplicate_outputs(self._producer_cache)

        return self._producer_cache

    def _producer_by_field(self) -> dict[str, Form]:
        return {
            field: producers[0]
            for field, producers in self._producer_map().items()
            if producers
        }

    def _validate_duplicate_outputs(
        self,
        producer_by_field: dict[str, list[Form]],
    ) -> None:
        if self.allow_duplicate_outputs:
            return
        duplicate_fields = {}
        for field, producers in producer_by_field.items():
            if len(producers) > 1:
                duplicate_fields[field] = sorted(
                    {producer.assignment for producer in producers}
                )
        if duplicate_fields:
            raise ValueError(
                "Duplicate form output fields: "
                f"{duplicate_fields}. Set allow_duplicate_outputs=True to "
                "collect duplicate producer values."
            )

    def _completed_producers_for_field(
        self,
        field: str,
        *,
        consumer: Form | None = None,
        cross_branch_only: bool = False,
    ) -> tuple[list[Form], list[Form]]:
        producers = []
        for form in self._producer_map().get(field, []):
            if form is consumer:
                continue
            if cross_branch_only and consumer is not None:
                if (form.branch or "_default") == (consumer.branch or "_default"):
                    continue
            producers.append(form)
        completed = [
            producer
            for producer in producers
            if str(producer.id) in self._completed_ids
            and field in producer.get_output_data()
        ]
        return producers, completed

    def _resolved_field_value(
        self,
        field: str,
        *,
        consumer: Form | None = None,
        cross_branch_only: bool = False,
    ) -> Any:
        producers, completed = self._completed_producers_for_field(
            field,
            consumer=consumer,
            cross_branch_only=cross_branch_only,
        )
        if producers:
            if len(completed) != len(producers):
                return _MISSING
            values = [producer.get_output_data()[field] for producer in completed]
            return values[0] if len(values) == 1 else values
        if field in self._initial_inputs:
            return self._initial_inputs[field]
        return _MISSING

    def _find_completed_producer(self, field: str) -> Form | None:
        _, completed = self._completed_producers_for_field(field)
        return completed[0] if len(completed) == 1 else None

    def _collect_input_fields(self, form: Form) -> list[str]:
        producer_map = self._producer_map()
        return [
            field
            for field in form.input_fields
            if sum(1 for p in producer_map.get(field, []) if p is not form) > 1
        ]

    def _forms_fingerprint(self, *, include_inputs: bool = True) -> tuple:
        return (
            self.allow_duplicate_outputs,
            tuple(
                (
                    str(form.id),
                    form.branch or "_default",
                    tuple(form.input_fields) if include_inputs else (),
                    tuple(form.output_fields),
                )
                for form in self.forms
            ),
        )

    def _external_form_producers(
        self,
        builder: OperationGraphBuilder,
    ) -> dict[str, list[UUID]]:
        producers: dict[str, list[UUID]] = defaultdict(list)
        for node in builder.get_graph().internal_nodes:
            metadata = getattr(node, "metadata", {}) or {}
            parameters = getattr(node, "parameters", {}) or {}
            output_fields = metadata.get("form_output_fields")
            if not output_fields and isinstance(parameters, dict):
                output_fields = parameters.get("form_output_fields")
            for field in output_fields or []:
                producers[field].append(node.id)
        return dict(producers)

    def _external_dependencies(
        self,
        form: Form,
        external_producers: dict[str, list[UUID]],
    ) -> list[UUID]:
        deps: list[UUID] = []
        for field in form.input_fields:
            for node_id in external_producers.get(field, []):
                if node_id not in deps:
                    deps.append(node_id)
        return deps

    def _previous_same_branch_form(self, form: Form) -> Form | None:
        forms = list(self.forms)
        idx = forms.index(form)
        branch = form.branch or "_default"
        for prior in reversed(forms[:idx]):
            if (prior.branch or "_default") == branch:
                return prior
        return None

    def _assert_acyclic(self) -> None:
        self._topological_forms()

    def _topological_forms(self) -> list[Form]:
        fingerprint = self._forms_fingerprint()
        if fingerprint == self._topological_cache_fingerprint:
            return list(self._topological_cache)

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
        self._topological_cache = ordered
        self._topological_cache_fingerprint = fingerprint
        return list(ordered)

    def __repr__(self) -> str:
        completed, total = self.progress
        return f"Report('{self.assignment}', {completed}/{total} forms)"
