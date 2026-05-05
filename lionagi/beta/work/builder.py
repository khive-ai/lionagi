# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Fluent builder for operation DAGs.

Build graphs with chained .add() calls, automatic sequential linking,
and explicit dependency management. Alias: Builder = OperationGraphBuilder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from lionagi.protocols.graph.edge import Edge
from lionagi.protocols.graph.graph import Graph
from lionagi.ln._utils import to_uuid
from lionagi.ln.types._sentinel import (
    Undefined,
    UndefinedType,
    is_sentinel,
    not_sentinel,
)

from .node import Operation

if TYPE_CHECKING:
    from lionagi.beta.session.session import Branch

__all__ = ("Builder", "OperationGraphBuilder")


def _resolve_branch_ref(branch: Any) -> UUID | str:
    """Coerce a branch reference to UUID or name string; raises ValueError if invalid."""
    try:
        return to_uuid(branch)
    except (ValueError, TypeError):
        pass

    if isinstance(branch, str) and branch.strip():
        return branch.strip()

    raise ValueError(f"Invalid branch reference: {branch}")


class OperationGraphBuilder:
    """Fluent builder for operation DAGs with automatic sequential linking."""

    def __init__(
        self,
        graph: Graph | str | UndefinedType = Undefined,
        *,
        name: str | None = None,
    ):
        if isinstance(graph, str):
            name = graph
            graph = Undefined
        self.name = name or "OperationGraph"
        self.graph = Graph() if is_sentinel(graph) else graph
        self._nodes: dict[str, Operation] = {}
        self._executed: set[UUID] = set()
        self._current_heads: list[str] = []

    def add(
        self,
        name: str,
        operation: str,
        parameters: dict[str, Any] | Any | UndefinedType = Undefined,
        depends_on: list[str] | UndefinedType = Undefined,
        branch: str | UUID | Branch | UndefinedType = Undefined,
        inherit_context: bool = False,
        metadata: dict[str, Any] | UndefinedType = Undefined,
        control_type: str | None = None,
        control_policy: dict[str, Any] | None = None,
        morphism: Any | UndefinedType = Undefined,
    ) -> OperationGraphBuilder:
        """Add an operation; Undefined depends_on auto-links to current head, [] is independent."""
        if name in self._nodes:
            raise ValueError(f"Operation with name '{name}' already exists")

        resolved_params = {} if is_sentinel(parameters) else parameters
        resolved_metadata = {} if is_sentinel(metadata) else metadata
        op = Operation(
            operation_type=operation,
            parameters=resolved_params,
            metadata=resolved_metadata,
            streaming=False,
            control_type=control_type,
            control_policy=control_policy,
            morphism=None if is_sentinel(morphism) else morphism,
        )
        op.metadata["name"] = name

        if not_sentinel(branch) and branch is not None:
            op.metadata["branch"] = _resolve_branch_ref(branch)

        if inherit_context and not_sentinel(depends_on) and depends_on:
            op.metadata["inherit_context"] = True
            op.metadata["primary_dependency"] = self._nodes[depends_on[0]].id

        self.graph.add_node(op)
        self._nodes[name] = op

        if not_sentinel(depends_on):
            for dep_name in depends_on:
                if dep_name not in self._nodes:
                    raise ValueError(f"Dependency '{dep_name}' not found")
                dep_node = self._nodes[dep_name]
                edge = Edge(head=dep_node.id, tail=op.id, label=["depends_on"])
                self.graph.add_edge(edge)
        elif self._current_heads:
            for head_name in self._current_heads:
                if head_name in self._nodes:
                    head_node = self._nodes[head_name]
                    edge = Edge(head=head_node.id, tail=op.id, label=["sequential"])
                    self.graph.add_edge(edge)

        self._current_heads = [name]
        return self

    def add_operation(
        self,
        operation: str,
        *,
        parameters: dict[str, Any] | Any | UndefinedType = Undefined,
        depends_on: list[str | UUID] | UndefinedType = Undefined,
        branch: str | UUID | Branch | None | UndefinedType = Undefined,
        inherit_context: bool = False,
        metadata: dict[str, Any] | UndefinedType = Undefined,
        control_type: str | None = None,
        control_policy: dict[str, Any] | None = None,
        morphism: Any | UndefinedType = Undefined,
    ) -> UUID:
        resolved_metadata = {} if is_sentinel(metadata) else dict(metadata)
        base_name = str(resolved_metadata.pop("name", operation))
        name = base_name
        index = 1
        while name in self._nodes:
            index += 1
            name = f"{base_name}_{index}"

        string_deps: list[str] = []
        uuid_deps: list[UUID] = []
        if not_sentinel(depends_on):
            for dep in depends_on:
                if isinstance(dep, UUID):
                    uuid_deps.append(dep)
                else:
                    string_deps.append(dep)

        self.add(
            name,
            operation,
            parameters=parameters,
            depends_on=string_deps if not_sentinel(depends_on) else depends_on,
            branch=branch,
            inherit_context=inherit_context,
            metadata=resolved_metadata,
            control_type=control_type,
            control_policy=control_policy,
            morphism=morphism,
        )
        op = self._nodes[name]
        for dep_id in uuid_deps:
            if dep_id not in self.graph.internal_nodes:
                raise ValueError(f"Dependency '{dep_id}' not found")
            self.graph.add_edge(Edge(head=dep_id, tail=op.id, label=["depends_on"]))
        return op.id

    def add_control(
        self,
        name: str,
        action: str,
        *,
        targets: list[str | UUID] | UndefinedType = Undefined,
        reason: str | None = None,
        depends_on: list[str | UUID] | UndefinedType = Undefined,
        branch: str | UUID | Branch | None | UndefinedType = Undefined,
        metadata: dict[str, Any] | UndefinedType = Undefined,
    ) -> UUID:
        control_policy: dict[str, Any] = {}
        if not is_sentinel(targets):
            control_policy["targets"] = list(targets)
        if reason is not None:
            control_policy["reason"] = reason
        resolved_metadata = {} if is_sentinel(metadata) else dict(metadata)
        resolved_metadata["name"] = name
        return self.add_operation(
            f"control.{action}",
            depends_on=depends_on,
            branch=branch,
            metadata=resolved_metadata,
            control_type=action,
            control_policy=control_policy,
        )

    def add_morphism(
        self,
        name: str,
        morphism: Any,
        parameters: dict[str, Any] | Any | UndefinedType = Undefined,
        depends_on: list[str] | UndefinedType = Undefined,
        branch: str | UUID | Branch | UndefinedType = Undefined,
        inherit_context: bool = False,
        metadata: dict[str, Any] | UndefinedType = Undefined,
        control: bool = False,
    ) -> OperationGraphBuilder:
        operation_name = getattr(morphism, "name", None) or f"morphism.{name}"
        return self.add(
            name,
            operation_name,
            parameters=parameters,
            depends_on=depends_on,
            branch=branch,
            inherit_context=inherit_context,
            metadata=metadata,
            control_type="morphism" if control else None,
            morphism=morphism,
        )

    def add_form(
        self,
        form: Any,
        *,
        operation: str,
        parameters: dict[str, Any] | Any | UndefinedType = Undefined,
        depends_on: list[str | UUID] | UndefinedType = Undefined,
        branch: str | UUID | Branch | None | UndefinedType = Undefined,
        inherit_context: bool = False,
        metadata: dict[str, Any] | UndefinedType = Undefined,
    ) -> UUID:
        resolved_metadata = {} if is_sentinel(metadata) else dict(metadata)
        resolved_metadata.setdefault("form_id", str(getattr(form, "id", "")))
        resolved_metadata.setdefault("form_assignment", getattr(form, "assignment", ""))
        return self.add_operation(
            operation,
            parameters=parameters,
            depends_on=depends_on,
            branch=branch,
            inherit_context=inherit_context,
            metadata=resolved_metadata,
        )

    def depends_on(
        self,
        target: str,
        *dependencies: str,
        label: list[str] | UndefinedType = Undefined,
    ) -> OperationGraphBuilder:
        if target not in self._nodes:
            raise ValueError(f"Target operation '{target}' not found")

        target_node = self._nodes[target]
        resolved_label = (
            [] if is_sentinel(label, additions={"none", "empty"}) else label
        )

        for dep_name in dependencies:
            if dep_name not in self._nodes:
                raise ValueError(f"Dependency operation '{dep_name}' not found")

            dep_node = self._nodes[dep_name]

            # Create edge: dependency -> target
            edge = Edge(
                head=dep_node.id,
                tail=target_node.id,
                label=resolved_label,
            )
            self.graph.add_edge(edge)

        return self

    def get(self, name: str) -> Operation:
        if name not in self._nodes:
            raise ValueError(f"Operation '{name}' not found")
        return self._nodes[name]

    def get_by_id(self, operation_id: UUID) -> Operation | None:
        node = self.graph.internal_nodes.get(operation_id, None)
        if isinstance(node, Operation):
            return node
        return None

    def mark_executed(self, *names: str) -> OperationGraphBuilder:
        for name in names:
            if name in self._nodes:
                self._executed.add(self._nodes[name].id)
        return self

    def get_unexecuted_nodes(self) -> list[Operation]:
        return [op for op in self._nodes.values() if op.id not in self._executed]

    def build(self) -> Graph:
        if not self.graph.is_acyclic():
            raise ValueError("Operation graph has cycles - must be a DAG")
        return self.graph

    def get_graph(self) -> Graph:
        return self.build()

    def clear(self) -> OperationGraphBuilder:
        self.graph = Graph()
        self._nodes = {}
        self._executed = set()
        self._current_heads = []
        return self

    def __repr__(self) -> str:
        return (
            f"OperationGraphBuilder("
            f"operations={len(self._nodes)}, "
            f"edges={len(self.graph.internal_edges)}, "
            f"executed={len(self._executed)})"
        )


# Alias for convenience
Builder = OperationGraphBuilder
