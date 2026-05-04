# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Incremental construction helpers for operation DAGs.

``OperationGraphBuilder`` is intentionally a graph/spec builder, not the
runtime state machine. Runtime truth lives on ``Operation.execution`` and in
``session.flow()`` results. Builder conveniences therefore encode behavior as
real graph topology or ``EdgeCondition`` objects; edge labels are only display
metadata.
"""

from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from typing import Any
from uuid import UUID

from lionagi.models.note import Note
from lionagi.operations.node import create_operation
from lionagi.protocols.graph.edge import Edge, EdgeCondition
from lionagi.protocols.types import ID, Condition, EventStatus

__all__ = (
    "OperationGraphBuilder",
    "ExpansionStrategy",
    "ResultCondition",
)


class ExpansionStrategy(Enum):
    """Topologies for expanding one source into many operations."""

    CONCURRENT = "concurrent"
    SEQUENTIAL = "sequential"
    SEQUENTIAL_CONCURRENT_CHUNK = "sequential_concurrent_chunk"
    CONCURRENT_SEQUENTIAL_CHUNK = "concurrent_sequential_chunk"


class ResultCondition(EdgeCondition):
    """Condition that evaluates an incoming predecessor result.

    Args:
        key: Optional dot-separated key path to read from ``context["result"]``.
        equals: Value to compare against when ``use_equals`` is True.
        use_equals: If True, compare ``value == equals``. Otherwise evaluate
            truthiness.
        expected_truthy: Expected truthiness when not using equality.
        negate: Invert the final condition result.
    """

    key: str | None = None
    equals: Any = None
    use_equals: bool = False
    expected_truthy: bool | None = True
    negate: bool = False

    async def apply(self, context: dict) -> bool:
        value = context.get("result")
        if self.key:
            value = self._get_path(value, self.key)

        if self.use_equals:
            matched = value == self.equals
        elif self.expected_truthy is None:
            matched = bool(value)
        else:
            matched = bool(value) is self.expected_truthy

        return not matched if self.negate else matched

    @staticmethod
    def _get_path(value: Any, path: str) -> Any:
        for part in path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = getattr(value, part, None)
            if value is None:
                break
        return value


class OperationGraphBuilder:
    """Incremental operation DAG builder.

    The builder supports build -> execute -> expand cycles by mutating the same
    graph object between ``session.flow()`` calls. It does not track execution
    itself; methods such as ``get_unexecuted_nodes()`` read the actual
    ``Operation.execution.status`` values.

    Examples:
        >>> builder = OperationGraphBuilder()
        >>> root = builder.add_operation("operate", instruction="Generate ideas")
        >>> child = builder.add_operation(
        ...     "operate", depends_on=[root], instruction="Expand one idea"
        ... )
        >>> graph = builder.get_graph()
    """

    def __init__(self, name: str = "DynamicGraph"):
        """Initialize the incremental graph builder."""
        from lionagi.protocols.graph.graph import Graph

        self.name = name
        self.graph = Graph()

        self._operations: dict[UUID, Any] = {}
        self._refs: dict[str, UUID] = {}
        # Backward-compatible manual marks. Runtime truth remains on Operation.
        self._executed: set[UUID] = set()
        # Frontier leaves. Kept for compatibility with existing callers.
        self._current_heads: list[UUID] = []
        self.last_operation_id: UUID | None = None

    def add_operation(
        self,
        operation: str,
        node_id: str | None = None,
        depends_on: Sequence[Any] | Any | None = None,
        *,
        ref: str | None = None,
        after: Sequence[Any] | Any | None = None,
        inherit_context: bool = False,
        branch=None,
        branch_id: Any = None,
        metadata: dict[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
        auto_link: bool = False,
        condition: Condition | None = None,
        conditions: Mapping[Any, Condition] | None = None,
        edge_label: str | list[str] | None = None,
        **operation_params,
    ) -> UUID:
        """Add one operation node.

        Repeated ``add_operation()`` calls are independent by default. Use
        ``depends_on=...`` or ``after=...`` for ordering, or ``auto_link=True``
        for the legacy "attach to current leaves" behavior.

        Args:
            operation: The branch operation
            node_id: Backward-compatible reference label. Prefer ``ref``.
            depends_on: Node refs this operation depends on.
            after: Alias for ``depends_on`` when building a chain.
            inherit_context: If True and has dependencies, inherit conversation
                context from the first dependency.
            branch: Branch object or branch UUID for this operation.
            branch_id: Backward-compatible alias for ``branch``.
            metadata: Node metadata stored on the Operation.
            parameters: Operation parameters to merge with ``**operation_params``.
            auto_link: Attach from current graph leaves if no dependencies.
            condition: Edge condition for a single dependency.
            conditions: Mapping of dependency ref -> edge condition.
            edge_label: Label for dependency edges created by this call.
            **operation_params: Parameters passed to the branch operation.

        Returns:
            ID of the created node
        """
        if depends_on is not None and after is not None:
            raise ValueError("Use either depends_on or after, not both.")

        dep_refs = after if after is not None else depends_on
        dependency_ids = self._normalize_dependency_refs(dep_refs)
        if dep_refs is None and auto_link:
            dependency_ids = list(self._current_heads)

        edge_conditions = self._normalize_conditions(
            dependency_ids, condition, conditions
        )

        operation_parameters = dict(parameters or {})
        operation_parameters.update(operation_params)

        node_metadata = dict(metadata or {})
        reference_id = ref or node_id
        if reference_id:
            if reference_id in self._refs:
                raise ValueError(f"Duplicate operation reference: {reference_id}")
            node_metadata["reference_id"] = reference_id

        if inherit_context:
            if not dependency_ids:
                raise ValueError(
                    "inherit_context=True requires at least one dependency"
                )
            node_metadata["inherit_context"] = True
            node_metadata["primary_dependency"] = dependency_ids[0]

        node = create_operation(
            operation=operation,
            parameters=operation_parameters,
            metadata=node_metadata,
        )

        if branch is None and branch_id is not None:
            branch = branch_id
        if branch is not None:
            node.branch_id = self._resolve_branch_id(branch)

        self.graph.add_node(node)
        self._operations[node.id] = node
        if reference_id:
            self._refs[reference_id] = node.id

        for dep_id in dependency_ids:
            label = edge_label or (["sequential"] if auto_link else ["depends_on"])
            edge = Edge(
                head=dep_id,
                tail=node.id,
                condition=edge_conditions.get(dep_id),
                label=label,
            )
            self.graph.add_edge(edge)

        self._update_current_heads(node.id, dependency_ids)
        self.last_operation_id = node.id

        return node.id

    def then(self, previous: Any, operation: str, **kwargs) -> UUID:
        """Add an operation that explicitly depends on ``previous``."""
        return self.add_operation(operation, depends_on=[previous], **kwargs)

    def expand_from_result(
        self,
        items: Iterable[Any],
        source_node_id: Any,
        operation: str,
        strategy: ExpansionStrategy = ExpansionStrategy.CONCURRENT,
        inherit_context: bool = False,
        chain_context: bool = False,
        chunk_size: int | None = None,
        branch=None,
        **shared_params,
    ) -> list[UUID]:
        """Expand one source node into many operation nodes.

        ``CONCURRENT`` creates ``source -> child`` edges for every child.
        ``SEQUENTIAL`` creates ``source -> child0 -> child1 -> ...``. Chunked
        strategies require ``chunk_size`` and encode their ordering as real DAG
        edges rather than labels alone.

        Args:
            items: Items from result to expand (e.g., instruct_model)
            source_node_id: ID of node that produced these items
            operation: Operation to apply to each item
            strategy: How to organize the expanded operations
            inherit_context: If True, expanded operations inherit context from source
            chain_context: If True and strategy is SEQUENTIAL, each operation
                inherits from the previous.
            chunk_size: Required for chunked strategies.
            branch: Optional branch assignment for all expanded operations.
            **shared_params: Shared parameters for all operations

        Returns:
            List of new node IDs
        """
        source_id = self._resolve_operation_id(source_node_id, "Source node")
        item_list = list(items)
        if strategy in (
            ExpansionStrategy.SEQUENTIAL_CONCURRENT_CHUNK,
            ExpansionStrategy.CONCURRENT_SEQUENTIAL_CHUNK,
        ):
            if chunk_size is None or chunk_size <= 0:
                raise ValueError(f"{strategy.value} requires chunk_size > 0")

        new_node_ids = []

        for i, item in enumerate(item_list):
            if hasattr(item, "model_dump"):
                params = {**item.model_dump(), **shared_params}
            else:
                params = {**shared_params, "item_index": i, "item": item}

            params["expanded_from"] = str(source_id)
            params["expansion_strategy"] = strategy.value

            node_meta = Note(
                expansion_index=i,
                expansion_source=source_id,
                expansion_strategy=strategy.value,
            )
            if inherit_context:
                node_meta["inherit_context"] = True
                if chain_context and strategy == ExpansionStrategy.SEQUENTIAL and i > 0:
                    node_meta["primary_dependency"] = new_node_ids[i - 1]
                else:
                    node_meta["primary_dependency"] = source_id

            node = create_operation(
                operation=operation,
                parameters=params,
                metadata=node_meta.content,
            )

            self.graph.add_node(node)
            self._operations[node.id] = node
            if branch is not None:
                node.branch_id = ID.get_id(branch)
            new_node_ids.append(node.id)

        self._connect_expansion(source_id, new_node_ids, strategy, chunk_size)
        expansion_frontier = self._expansion_frontier(
            new_node_ids, strategy, chunk_size
        )
        self._current_heads = [
            head for head in self._current_heads if head != source_id
        ]
        self._current_heads.extend(expansion_frontier)

        return new_node_ids

    def add_aggregation(
        self,
        operation: str,
        node_id: str | None = None,
        source_node_ids: Sequence[Any] | None = None,
        inherit_context: bool = False,
        inherit_from_source: int = 0,
        branch=None,
        metadata: dict[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
        **operation_params,
    ) -> UUID:
        """Add a fan-in operation that depends on multiple source nodes.

        Args:
            operation: Aggregation operation
            node_id: Optional ID reference for this node
            source_node_ids: Nodes to aggregate from (defaults to current heads)
            inherit_context: If True, inherit conversation context from one source
            inherit_from_source: Index of source to inherit context from (default: 0)
            metadata: Additional node metadata.
            parameters: Operation parameters to merge with ``**operation_params``.
            **operation_params: Operation parameters.

        Returns:
            ID of aggregation node
        """
        source_refs = (
            self._current_heads if source_node_ids is None else source_node_ids
        )
        sources = self._normalize_dependency_refs(source_refs)
        if not sources:
            raise ValueError("No source nodes for aggregation")
        if inherit_from_source < 0 or inherit_from_source >= len(sources):
            raise ValueError("inherit_from_source is out of range")

        op_params = dict(parameters or {})
        op_params.update(operation_params)
        agg_params = {
            "aggregation_sources": [str(s) for s in sources],
            "aggregation_count": len(sources),
            **op_params,
        }

        agg_meta = Note(**(metadata or {}), aggregation=True)
        if node_id:
            if node_id in self._refs:
                raise ValueError(f"Duplicate operation reference: {node_id}")
            agg_meta["reference_id"] = node_id
        if inherit_context:
            source_idx = inherit_from_source
            agg_meta["inherit_context"] = True
            agg_meta["primary_dependency"] = sources[source_idx]
            agg_meta["inherit_from_source"] = source_idx

        node = create_operation(
            operation=operation,
            parameters=agg_params,
            metadata=agg_meta.content,
        )

        if branch:
            node.branch_id = self._resolve_branch_id(branch)

        self.graph.add_node(node)
        self._operations[node.id] = node
        if node_id:
            self._refs[node_id] = node.id

        for source_id in sources:
            edge = Edge(head=source_id, tail=node.id, label=["aggregate"])
            self.graph.add_edge(edge)

        self._current_heads = [node.id]
        self.last_operation_id = node.id

        return node.id

    def add_form(
        self,
        form: Any,
        *,
        operation: str = "operate",
        depends_on: Sequence[Any] | Any | None = None,
        inherit_context: bool = False,
        branch=None,
        metadata: dict[str, Any] | None = None,
        parameters: Mapping[str, Any] | None = None,
        **operation_params,
    ) -> UUID:
        """Add a work Form as an operation node.

        Forms are I/O contracts, not execution units. This helper stores the
        contract on operation metadata and parameters so ``Session.flow()`` can
        inject named form inputs from initial context and predecessor results.
        """
        caller_parameters = dict(parameters or {})
        caller_parameters.update(operation_params)
        provided_form_inputs = caller_parameters.pop("form_inputs", None)

        form_inputs = {}
        if hasattr(form, "get_inputs"):
            form_inputs = form.get_inputs()
        if provided_form_inputs is not None:
            if not isinstance(provided_form_inputs, Mapping):
                raise TypeError("form_inputs must be a mapping when provided")
            form_inputs.update(provided_form_inputs)

        form_contract = {
            "form_id": str(form.id),
            "form_assignment": getattr(form, "assignment", ""),
            "form_input_fields": list(getattr(form, "input_fields", []) or []),
            "form_output_fields": list(getattr(form, "output_fields", []) or []),
            "form_resource": getattr(form, "resource", None),
            "form_collect_input_fields": [],
        }
        for key in form_contract:
            if key in caller_parameters:
                form_contract[key] = caller_parameters[key]
        form_metadata = {
            **(metadata or {}),
            "form": True,
            **form_contract,
            "form_branch": getattr(form, "branch", None),
        }
        form_parameters = {
            **caller_parameters,
            **form_contract,
            "form_inputs": form_inputs,
        }

        return self.add_operation(
            operation,
            depends_on=depends_on,
            inherit_context=inherit_context,
            branch=branch,
            metadata=form_metadata,
            parameters=form_parameters,
        )

    def add_report(
        self,
        report: Any,
        *,
        operation: str = "operate",
        branches: Mapping[str, Any] | None = None,
        parameter_factory=None,
    ) -> dict[str, UUID]:
        """Compile a work Report into this builder.

        Returns:
            Mapping of report form ID strings to operation node IDs.
        """
        report.to_builder(
            operation=operation,
            builder=self,
            branches=branches,
            parameter_factory=parameter_factory,
        )
        return report.node_by_form_id

    def mark_executed(self, node_ids: Sequence[Any]):
        """Mark nodes as completed for subsequent resume-style flow runs.

        This is a manual override. Normal execution state is set by
        ``session.flow()``.

        Args:
            node_ids: IDs of executed nodes
        """
        for node_ref in node_ids:
            node_id = self._resolve_operation_id(node_ref, "Operation")
            self._executed.add(node_id)
            self._operations[node_id].execution.status = EventStatus.COMPLETED

    def get_unexecuted_nodes(self):
        """Return operations that are not in a terminal execution state.

        Returns:
            List of unexecuted operations
        """
        return [
            op
            for op in self._operations.values()
            if op.execution.status
            not in {
                EventStatus.COMPLETED,
                EventStatus.FAILED,
                EventStatus.SKIPPED,
                EventStatus.CANCELLED,
                EventStatus.ABORTED,
            }
        ]

    def add_conditional_branch(
        self,
        condition_check_op: str,
        true_op: str,
        false_op: str | None = None,
        true_condition: Condition | None = None,
        false_condition: Condition | None = None,
        condition_key: str | None = None,
        condition_value: Any = True,
        true_params: Mapping[str, Any] | None = None,
        false_params: Mapping[str, Any] | None = None,
        **check_params,
    ) -> dict[str, UUID]:
        """Add a conditional branch backed by real edge conditions.

        By default the true edge traverses when the check result is truthy; the
        false edge traverses when it is not. Pass ``condition_key`` and
        ``condition_value`` to test a field in the check result, or provide
        explicit ``true_condition``/``false_condition`` objects.

        Args:
            condition_check_op: Operation that evaluates condition
            true_op: Operation if condition is true
            false_op: Operation if condition is false
            true_condition: Edge condition for the true branch
            false_condition: Edge condition for the false branch
            condition_key: Optional result field path to compare
            condition_value: Expected value when using ``condition_key``
            true_params: Parameters for the true branch operation
            false_params: Parameters for the false branch operation
            **check_params: Parameters for condition check

        Returns:
            Dict with node IDs: {'check': id, 'true': id, 'false': id}
        """
        if condition_key is not None:
            true_condition = true_condition or ResultCondition(
                key=condition_key,
                equals=condition_value,
                use_equals=True,
            )
            false_condition = false_condition or ResultCondition(
                key=condition_key,
                equals=condition_value,
                use_equals=True,
                negate=True,
            )
        else:
            true_condition = true_condition or ResultCondition(expected_truthy=True)
            false_condition = false_condition or ResultCondition(
                expected_truthy=True,
                negate=True,
            )

        check_depends = list(self._current_heads)
        check_id = self.add_operation(
            condition_check_op,
            depends_on=check_depends,
            parameters={**check_params, "is_condition_check": True},
            edge_label=["to_condition"],
        )

        result = {"check": check_id}

        true_id = self.add_operation(
            true_op,
            depends_on=[check_id],
            condition=true_condition,
            parameters={"branch": "true"} if true_params is None else dict(true_params),
            edge_label=["if_true"],
        )
        result["true"] = true_id

        if false_op:
            false_id = self.add_operation(
                false_op,
                depends_on=[check_id],
                condition=false_condition,
                parameters=(
                    {"branch": "false"} if false_params is None else dict(false_params)
                ),
                edge_label=["if_false"],
            )
            result["false"] = false_id

        self._current_heads = [result["true"]]
        if "false" in result:
            self._current_heads.append(result["false"])

        return result

    def get_graph(self):
        """
        Get the current graph for execution.

        Returns:
            The graph in its current state
        """
        return self.graph

    def get_node_by_reference(self, reference_id: str):
        """Get a node by its unique reference ID.

        Args:
            reference_id: The reference ID assigned when creating the node

        Returns:
            The operation node or None
        """
        node_id = self._refs.get(reference_id)
        return self._operations.get(node_id) if node_id else None

    def visualize_state(self) -> dict[str, Any]:
        """
        Get visualization of current graph state.

        Returns:
            Dict with graph statistics and state
        """
        expansions = {}
        status_counts = {}
        for op in self._operations.values():
            source = op.metadata.get("expansion_source")
            if source:
                if source not in expansions:
                    expansions[source] = []
                expansions[source].append(op.id)
            status = op.execution.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        roots = [node.id for node in self.graph.get_heads()]
        leaves = [node.id for node in self.graph.get_tails()]

        return {
            "name": self.name,
            "total_nodes": len(self._operations),
            "executed_nodes": status_counts.get(EventStatus.COMPLETED.value, 0),
            "unexecuted_nodes": len(self.get_unexecuted_nodes()),
            "current_heads": self._current_heads,
            "roots": roots,
            "leaves": leaves,
            "status_counts": status_counts,
            "expansions": expansions,
            "edges": len(self.graph.internal_edges),
        }

    def visualize(self, title: str = "Operation Graph", figsize=(14, 10)):
        from ._visualize_graph import visualize_graph

        visualize_graph(
            self,
            title=title,
            figsize=figsize,
        )

    def _normalize_dependency_refs(
        self, refs: Sequence[Any] | Any | None
    ) -> list[UUID]:
        if refs is None:
            return []
        if isinstance(refs, (str, UUID)) or hasattr(refs, "id"):
            refs = [refs]
        return [self._resolve_operation_id(ref, "Dependency") for ref in refs]

    def _resolve_operation_id(self, ref: Any, label: str = "Operation") -> UUID:
        if isinstance(ref, str) and ref in self._refs:
            return self._refs[ref]
        try:
            node_id = ID.get_id(ref)
        except Exception:
            raise ValueError(f"{label} {ref!r} not found") from None
        if node_id not in self._operations:
            raise ValueError(f"{label} {ref!r} not found")
        return node_id

    def _normalize_conditions(
        self,
        dependency_ids: list[UUID],
        condition: Condition | None,
        conditions: Mapping[Any, Condition] | None,
    ) -> dict[UUID, Condition]:
        if condition is not None:
            if len(dependency_ids) != 1:
                raise ValueError("condition= requires exactly one dependency")
            return {dependency_ids[0]: condition}
        if not conditions:
            return {}

        normalized = {}
        for dep_ref, edge_condition in conditions.items():
            dep_id = self._resolve_operation_id(dep_ref, "Condition dependency")
            if dep_id not in dependency_ids:
                raise ValueError(
                    f"Condition dependency {dep_ref!r} is not in depends_on"
                )
            normalized[dep_id] = edge_condition
        return normalized

    def _resolve_branch_id(self, branch: Any) -> UUID:
        try:
            return ID.get_id(branch)
        except Exception:
            if hasattr(branch, "id"):
                return ID.get_id(branch.id)
            raise

    def _update_current_heads(self, node_id: UUID, dependency_ids: list[UUID]) -> None:
        """Update the builder frontier after adding one operation node.

        The frontier is the set of current leaves created through builder
        helpers. Independent roots append to it; dependent nodes replace the
        dependencies they consumed while preserving unrelated leaves.
        """
        if not dependency_ids:
            if node_id not in self._current_heads:
                self._current_heads.append(node_id)
            return

        consumed = set(dependency_ids)
        self._current_heads = [
            head for head in self._current_heads if head not in consumed
        ]
        self._current_heads.append(node_id)

    def _connect_expansion(
        self,
        source_id: UUID,
        new_node_ids: list[UUID],
        strategy: ExpansionStrategy,
        chunk_size: int | None,
    ) -> None:
        if not new_node_ids:
            return

        if strategy == ExpansionStrategy.CONCURRENT:
            for node_id in new_node_ids:
                self.graph.add_edge(
                    Edge(
                        head=source_id,
                        tail=node_id,
                        label=["expansion", strategy.value],
                    )
                )
            return

        if strategy == ExpansionStrategy.SEQUENTIAL:
            previous = source_id
            for node_id in new_node_ids:
                self.graph.add_edge(
                    Edge(
                        head=previous,
                        tail=node_id,
                        label=["expansion", strategy.value],
                    )
                )
                previous = node_id
            return

        chunks = [
            new_node_ids[i : i + chunk_size]
            for i in range(0, len(new_node_ids), chunk_size or len(new_node_ids))
        ]

        if strategy == ExpansionStrategy.SEQUENTIAL_CONCURRENT_CHUNK:
            previous_layer = [source_id]
            for chunk in chunks:
                for head_id in previous_layer:
                    for tail_id in chunk:
                        self.graph.add_edge(
                            Edge(
                                head=head_id,
                                tail=tail_id,
                                label=["expansion", strategy.value],
                            )
                        )
                previous_layer = chunk
            return

        if strategy == ExpansionStrategy.CONCURRENT_SEQUENTIAL_CHUNK:
            for chunk in chunks:
                previous = source_id
                for node_id in chunk:
                    self.graph.add_edge(
                        Edge(
                            head=previous,
                            tail=node_id,
                            label=["expansion", strategy.value],
                        )
                    )
                    previous = node_id
            return

        raise ValueError(f"Unsupported expansion strategy: {strategy}")

    def _expansion_frontier(
        self,
        new_node_ids: list[UUID],
        strategy: ExpansionStrategy,
        chunk_size: int | None,
    ) -> list[UUID]:
        if not new_node_ids:
            return []
        if strategy == ExpansionStrategy.CONCURRENT:
            return new_node_ids
        if strategy == ExpansionStrategy.SEQUENTIAL:
            return [new_node_ids[-1]]
        chunks = [
            new_node_ids[i : i + chunk_size]
            for i in range(0, len(new_node_ids), chunk_size or len(new_node_ids))
        ]
        if strategy == ExpansionStrategy.SEQUENTIAL_CONCURRENT_CHUNK:
            return chunks[-1]
        if strategy == ExpansionStrategy.CONCURRENT_SEQUENTIAL_CHUNK:
            return [chunk[-1] for chunk in chunks]
        return new_node_ids
