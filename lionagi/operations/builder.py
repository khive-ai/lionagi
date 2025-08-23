# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

"""
OperationGraphBuilder: Incremental graph builder for multi-stage operations.

Build → Execute → Expand → Execute → ...
"""

from enum import Enum
from typing import Any

from lionagi.operations.node import BranchOperations, Operation
from lionagi.protocols.graph.edge import Edge
from lionagi.protocols.graph.graph import Graph
from lionagi.protocols.types import ID

__all__ = (
    "OperationGraphBuilder",
    "ExpansionStrategy",
)


class ExpansionStrategy(Enum):
    """Strategies for expanding operations."""

    CONCURRENT = "concurrent"
    SEQUENTIAL = "sequential"
    SEQUENTIAL_CONCURRENT_CHUNK = "sequential_concurrent_chunk"
    CONCURRENT_SEQUENTIAL_CHUNK = "concurrent_sequential_chunk"


class OperationGraphBuilder:
    """
    Incremental graph builder that supports build → execute → expand cycles.

    Unlike static builders, this maintains state and allows expanding the graph
    based on execution results.

    Examples:
        >>> # Build initial graph
        >>> builder = OperationGraphBuilder()
        >>> builder.add_operation("operate", instruction="Generate ideas", num_instruct=5)
        >>> graph = builder.get_graph()
        >>>
        >>> # Execute with session
        >>> result = await session.flow(graph)
        >>>
        >>> # Expand based on results
        >>> if hasattr(result, 'instruct_models'):
        ...     builder.expand_from_result(
        ...         result.instruct_models,
        ...         source_node_id=builder.last_operation_id,
        ...         operation="instruct"
        ...     )
        >>>
        >>> # Get expanded graph and continue execution
        >>> graph = builder.get_graph()
        >>> final_result = await session.flow(graph)
    """

    def __init__(self, name: str = "DynamicGraph"):
        """Initialize the incremental graph builder."""
        self.name = name
        self.graph = Graph()

        # Track state
        self._operations: dict[str, Operation] = {}  # All operations by ID
        self._executed: set[str] = set()  # IDs of executed operations
        self._current_heads: list[str] = []  # Current head nodes for linking
        self.last_operation_id: str | None = None

    def add_operation(
        self,
        operation: BranchOperations,
        node_id: str | None = None,
        depends_on: list[str] | None = None,
        inherit_context: bool = False,
        branch=None,
        **parameters,
    ) -> str:
        """
        Add an operation to the graph.

        Args:
            operation: The branch operation
            node_id: Optional ID reference for this node
            depends_on: List of node IDs this depends on
            inherit_context: If True and has dependencies, inherit conversation
                           context from primary (first) dependency
            **parameters: Operation parameters

        Returns:
            ID of the created node
        """
        # Create operation node
        node = Operation(operation=operation, parameters=parameters)

        # Store context inheritance strategy
        if inherit_context and depends_on:
            node.metadata["inherit_context"] = True
            node.metadata["primary_dependency"] = depends_on[0]

        self.graph.add_node(node)
        self._operations[node.id] = node

        # Store reference if provided
        if node_id:
            # Add as metadata for easy lookup
            node.metadata["reference_id"] = node_id

        if branch:
            node.branch_id = ID.get_id(branch)

        # Handle dependencies
        if depends_on:
            for dep_id in depends_on:
                if dep_id in self._operations:
                    edge = Edge(
                        head=dep_id, tail=node.id, label=["depends_on"]
                    )
                    self.graph.add_edge(edge)
        elif self._current_heads:
            # Auto-link from current heads
            for head_id in self._current_heads:
                edge = Edge(head=head_id, tail=node.id, label=["sequential"])
                self.graph.add_edge(edge)

        # Update state
        self._current_heads = [node.id]
        self.last_operation_id = node.id

        return node.id

    def expand_from_result(
        self,
        items: list[Any],
        source_node_id: str,
        operation: BranchOperations,
        strategy: ExpansionStrategy = ExpansionStrategy.CONCURRENT,
        inherit_context: bool = False,
        chain_context: bool = False,
        **shared_params,
    ) -> list[str]:
        """
        Expand the graph based on execution results.

        This is called after executing the graph to add new operations
        based on results.

        Args:
            items: Items from result to expand (e.g., instruct_models)
            source_node_id: ID of node that produced these items
            operation: Operation to apply to each item
            strategy: How to organize the expanded operations
            inherit_context: If True, expanded operations inherit context from source
            chain_context: If True and strategy is SEQUENTIAL, each operation
                          inherits from the previous (only applies to SEQUENTIAL)
            **shared_params: Shared parameters for all operations

        Returns:
            List of new node IDs
        """
        if source_node_id not in self._operations:
            raise ValueError(f"Source node {source_node_id} not found")

        new_node_ids = []

        # Create operation for each item
        for i, item in enumerate(items):
            # Extract parameters from item if it's a model
            if hasattr(item, "model_dump"):
                params = {**item.model_dump(), **shared_params}
            else:
                params = {**shared_params, "item_index": i, "item": str(item)}

            # Add metadata about expansion
            params["expanded_from"] = source_node_id
            params["expansion_strategy"] = strategy.value

            node = Operation(
                operation=operation,
                parameters=params,
                metadata={
                    "expansion_index": i,
                    "expansion_source": source_node_id,
                    "expansion_strategy": strategy.value,
                },
            )

            # Handle context inheritance for expanded operations
            if inherit_context:
                if (
                    chain_context
                    and strategy == ExpansionStrategy.SEQUENTIAL
                    and i > 0
                ):
                    # Chain context: inherit from previous expanded operation
                    node.metadata["inherit_context"] = True
                    node.metadata["primary_dependency"] = new_node_ids[i - 1]
                else:
                    # Inherit from source node
                    node.metadata["inherit_context"] = True
                    node.metadata["primary_dependency"] = source_node_id

            self.graph.add_node(node)
            self._operations[node.id] = node
            new_node_ids.append(node.id)

            # Link from source
            edge = Edge(
                head=source_node_id,
                tail=node.id,
                label=["expansion", strategy.value],
            )
            self.graph.add_edge(edge)

        # Update current heads based on strategy
        if strategy in [
            ExpansionStrategy.CONCURRENT,
            ExpansionStrategy.SEQUENTIAL,
        ]:
            self._current_heads = new_node_ids

        return new_node_ids

    def add_aggregation(
        self,
        operation: BranchOperations,
        node_id: str | None = None,
        source_node_ids: list[str] | None = None,
        inherit_context: bool = False,
        inherit_from_source: int = 0,
        branch=None,
        **parameters,
    ) -> str:
        """
        Add an aggregation operation that collects from multiple sources.

        Args:
            operation: Aggregation operation
            node_id: Optional ID reference for this node
            source_node_ids: Nodes to aggregate from (defaults to current heads)
            inherit_context: If True, inherit conversation context from one source
            inherit_from_source: Index of source to inherit context from (default: 0)
            **parameters: Operation parameters

        Returns:
            ID of aggregation node
        """
        sources = source_node_ids or self._current_heads
        if not sources:
            raise ValueError("No source nodes for aggregation")

        # Add aggregation metadata - convert IDType to strings for JSON serialization
        agg_params = {
            "aggregation_sources": [
                str(s) for s in sources
            ],  # Convert IDType to strings
            "aggregation_count": len(sources),
            **parameters,
        }

        node = Operation(
            operation=operation,
            parameters=agg_params,
            metadata={"aggregation": True},
        )

        # Store node reference if provided
        if node_id:
            node.metadata["reference_id"] = node_id

        if branch:
            node.branch_id = ID.get_id(branch)

        # Store context inheritance for aggregations
        if inherit_context and sources:
            node.metadata["inherit_context"] = True
            # Use the specified source index (bounded by available sources)
            source_idx = min(inherit_from_source, len(sources) - 1)
            node.metadata["primary_dependency"] = sources[source_idx]
            node.metadata["inherit_from_source"] = source_idx

        self.graph.add_node(node)
        self._operations[node.id] = node

        # Connect all sources
        for source_id in sources:
            edge = Edge(head=source_id, tail=node.id, label=["aggregate"])
            self.graph.add_edge(edge)

        # Update state
        self._current_heads = [node.id]
        self.last_operation_id = node.id

        return node.id

    def mark_executed(self, node_ids: list[str]):
        """
        Mark nodes as executed.

        This helps track which parts of the graph have been run.

        Args:
            node_ids: IDs of executed nodes
        """
        self._executed.update(node_ids)

    def get_unexecuted_nodes(self) -> list[Operation]:
        """
        Get nodes that haven't been executed yet.

        Returns:
            List of unexecuted operations
        """
        return [
            op
            for op_id, op in self._operations.items()
            if op_id not in self._executed
        ]

    def add_conditional_branch(
        self,
        condition_check_op: BranchOperations,
        true_op: BranchOperations,
        false_op: BranchOperations | None = None,
        **check_params,
    ) -> dict[str, str]:
        """
        Add a conditional branch structure.

        Args:
            condition_check_op: Operation that evaluates condition
            true_op: Operation if condition is true
            false_op: Operation if condition is false
            **check_params: Parameters for condition check

        Returns:
            Dict with node IDs: {'check': id, 'true': id, 'false': id}
        """
        # Add condition check node
        check_node = Operation(
            operation=condition_check_op,
            parameters={**check_params, "is_condition_check": True},
        )
        self.graph.add_node(check_node)
        self._operations[check_node.id] = check_node

        # Link from current heads
        for head_id in self._current_heads:
            edge = Edge(
                head=head_id, tail=check_node.id, label=["to_condition"]
            )
            self.graph.add_edge(edge)

        result = {"check": check_node.id}

        # Add true branch
        true_node = Operation(operation=true_op, parameters={"branch": "true"})
        self.graph.add_node(true_node)
        self._operations[true_node.id] = true_node
        result["true"] = true_node.id

        # Connect with condition label
        true_edge = Edge(
            head=check_node.id, tail=true_node.id, label=["if_true"]
        )
        self.graph.add_edge(true_edge)

        # Add false branch if specified
        if false_op:
            false_node = Operation(
                operation=false_op, parameters={"branch": "false"}
            )
            self.graph.add_node(false_node)
            self._operations[false_node.id] = false_node
            result["false"] = false_node.id

            false_edge = Edge(
                head=check_node.id, tail=false_node.id, label=["if_false"]
            )
            self.graph.add_edge(false_edge)

            self._current_heads = [true_node.id, false_node.id]
        else:
            self._current_heads = [true_node.id]

        return result

    def get_graph(self) -> Graph:
        """
        Get the current graph for execution.

        Returns:
            The graph in its current state
        """
        return self.graph

    def get_node_by_reference(self, reference_id: str) -> Operation | None:
        """
        Get a node by its reference ID.

        Args:
            reference_id: The reference ID assigned when creating the node

        Returns:
            The operation node or None
        """
        for op in self._operations.values():
            if op.metadata.get("reference_id") == reference_id:
                return op
        return None

    def visualize_state(self) -> dict[str, Any]:
        """
        Get visualization of current graph state.

        Returns:
            Dict with graph statistics and state
        """
        # Group nodes by expansion source
        expansions = {}
        for op in self._operations.values():
            source = op.metadata.get("expansion_source")
            if source:
                if source not in expansions:
                    expansions[source] = []
                expansions[source].append(op.id)

        return {
            "name": self.name,
            "total_nodes": len(self._operations),
            "executed_nodes": len(self._executed),
            "unexecuted_nodes": len(self._operations) - len(self._executed),
            "current_heads": self._current_heads,
            "expansions": expansions,
            "edges": len(self.graph.internal_edges),
        }

    def visualize(self, title: str = "Operation Graph", figsize=(14, 10)):
        visualize_graph(
            self,
            title=title,
            figsize=figsize,
        )


def visualize_graph(
    builder: OperationGraphBuilder,
    title: str = "Operation Graph",
    figsize=(14, 10),
):
    """Visualization with improved layout for complex graphs."""
    from lionagi.protocols.graph.graph import (
        _MATPLIB_AVAILABLE,
        _NETWORKX_AVAILABLE,
    )

    if _MATPLIB_AVAILABLE is not True:
        raise _MATPLIB_AVAILABLE
    if _NETWORKX_AVAILABLE is not True:
        raise _NETWORKX_AVAILABLE

    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    graph = builder.get_graph()

    # Convert to networkx
    G = nx.DiGraph()

    # Track node positions for hierarchical layout
    node_levels = {}
    node_labels = {}
    node_colors = []
    node_sizes = []

    # First pass: add nodes and determine levels
    for node in graph.internal_nodes.values():
        node_id = str(node.id)[:8]
        G.add_node(node_id)

        # Determine level based on dependencies
        in_edges = [
            e
            for e in graph.internal_edges.values()
            if str(e.tail)[:8] == node_id
        ]
        if not in_edges:
            level = 0  # Root nodes
        else:
            # Get max level of predecessors + 1
            pred_levels = []
            for edge in in_edges:
                pred_id = str(edge.head)[:8]
                if pred_id in node_levels:
                    pred_levels.append(node_levels[pred_id])
            level = max(pred_levels, default=0) + 1

        node_levels[node_id] = level

        # Create label
        ref_id = node.metadata.get("reference_id", "")
        if ref_id:
            label = f"{node.operation}\n[{ref_id}]"
        else:
            label = f"{node.operation}\n{node_id}"
        node_labels[node_id] = label

        # Color and size based on status and type
        if node.id in builder._executed:
            node_colors.append("#90EE90")  # Light green
            node_sizes.append(4000)
        elif node.metadata.get("expansion_source"):
            node_colors.append("#87CEEB")  # Sky blue
            node_sizes.append(3500)
        elif node.metadata.get("aggregation"):
            node_colors.append("#FFD700")  # Gold
            node_sizes.append(4500)
        elif node.metadata.get("is_condition_check"):
            node_colors.append("#DDA0DD")  # Plum
            node_sizes.append(3500)
        else:
            node_colors.append("#E0E0E0")  # Light gray
            node_sizes.append(3000)

    # Add edges
    edge_colors = []
    edge_styles = []
    edge_widths = []
    edge_labels = {}

    for edge in graph.internal_edges.values():
        head_id = str(edge.head)[:8]
        tail_id = str(edge.tail)[:8]
        G.add_edge(head_id, tail_id)

        # Style edges based on type
        edge_label = edge.label[0] if edge.label else ""
        edge_labels[(head_id, tail_id)] = edge_label

        if "expansion" in edge_label:
            edge_colors.append("#4169E1")  # Royal blue
            edge_styles.append("dashed")
            edge_widths.append(2)
        elif "aggregate" in edge_label:
            edge_colors.append("#FF6347")  # Tomato
            edge_styles.append("dotted")
            edge_widths.append(2.5)
        else:
            edge_colors.append("#808080")  # Gray
            edge_styles.append("solid")
            edge_widths.append(1.5)

    # Create improved hierarchical layout
    pos = {}
    nodes_by_level = {}

    for node_id, level in node_levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node_id)

    # Position nodes with better spacing algorithm
    y_spacing = 2.5
    max_width = 16  # Maximum horizontal spread

    for level, nodes in nodes_by_level.items():
        num_nodes = len(nodes)

        if num_nodes <= 6:
            # Normal spacing for small levels
            x_spacing = 2.5
            x_offset = -(num_nodes - 1) * x_spacing / 2
            for i, node_id in enumerate(nodes):
                pos[node_id] = (x_offset + i * x_spacing, -level * y_spacing)
        else:
            # Multi-row layout for large levels
            nodes_per_row = min(6, int(np.ceil(np.sqrt(num_nodes * 1.5))))
            rows = int(np.ceil(num_nodes / nodes_per_row))

            for i, node_id in enumerate(nodes):
                row = i // nodes_per_row
                col = i % nodes_per_row

                # Calculate row width
                nodes_in_row = min(
                    nodes_per_row, num_nodes - row * nodes_per_row
                )
                x_spacing = 2.5
                x_offset = -(nodes_in_row - 1) * x_spacing / 2

                # Add slight y offset for different rows
                y_offset = row * 0.8

                pos[node_id] = (
                    x_offset + col * x_spacing,
                    -level * y_spacing - y_offset,
                )

    # Create figure
    plt.figure(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=2,
        edgecolors="black",
    )

    # Draw edges with different styles - use curved edges for better visibility
    for i, (u, v) in enumerate(G.edges()):
        # Calculate curve based on node positions
        u_pos = pos[u]
        v_pos = pos[v]

        # Determine connection style based on relative positions
        if abs(u_pos[0] - v_pos[0]) > 5:  # Far apart horizontally
            connectionstyle = "arc3,rad=0.2"
        else:
            connectionstyle = "arc3,rad=0.1"

        nx.draw_networkx_edges(
            G,
            pos,
            [(u, v)],
            edge_color=[edge_colors[i]],
            style=edge_styles[i],
            width=edge_widths[i],
            alpha=0.7,
            arrows=True,
            arrowsize=20,
            arrowstyle="-|>",
            connectionstyle=connectionstyle,
        )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        node_labels,
        font_size=9,
        font_weight="bold",
        font_family="monospace",
    )

    # Draw edge labels (only for smaller graphs)
    if len(G.edges()) < 20:
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels,
            font_size=7,
            font_color="darkblue",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
            ),
        )

    plt.title(title, fontsize=18, fontweight="bold", pad=20)
    plt.axis("off")

    # Enhanced legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    legend_elements = [
        Patch(facecolor="#90EE90", edgecolor="black", label="Executed"),
        Patch(facecolor="#87CEEB", edgecolor="black", label="Expanded"),
        Patch(facecolor="#FFD700", edgecolor="black", label="Aggregation"),
        Patch(facecolor="#DDA0DD", edgecolor="black", label="Condition"),
        Patch(facecolor="#E0E0E0", edgecolor="black", label="Pending"),
        Line2D([0], [0], color="#808080", linewidth=2, label="Sequential"),
        Line2D(
            [0],
            [0],
            color="#4169E1",
            linewidth=2,
            linestyle="dashed",
            label="Expansion",
        ),
        Line2D(
            [0],
            [0],
            color="#FF6347",
            linewidth=2,
            linestyle="dotted",
            label="Aggregate",
        ),
    ]

    plt.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=2,
    )

    # Add statistics box
    stats_text = f"Nodes: {len(G.nodes())}\nEdges: {len(G.edges())}\nExecuted: {len(builder._executed)}"
    if nodes_by_level:
        max_level = max(nodes_by_level.keys())
        stats_text += f"\nLevels: {max_level + 1}"

    plt.text(
        0.98,
        0.02,
        stats_text,
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=10,
        fontfamily="monospace",
    )

    plt.tight_layout()
    plt.show()
