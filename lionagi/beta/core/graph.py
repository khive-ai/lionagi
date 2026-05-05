# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""OpNode and OpGraph: the DAG execution model for lionagi.beta.

Distinct from core/base/Graph (rich Pydantic graph for knowledge/content).
OpGraph is minimal: nodes → morphisms + deps. Formally analyzable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from .policy import covers

if TYPE_CHECKING:
    from .morphism import Morphism

__all__ = (
    "OpGraph",
    "OpNode",
)


@dataclass(slots=True)
class OpNode:
    id: UUID = field(default_factory=uuid4)
    m: Morphism = field(default=None)  # type: ignore[assignment]
    deps: set[UUID] = field(default_factory=set)
    params: dict[str, Any] = field(default_factory=dict)
    control: bool = False


@dataclass(slots=True)
class OpGraph:
    nodes: dict[UUID, OpNode] = field(default_factory=dict)
    roots: set[UUID] = field(default_factory=set)

    def validate_dag(self) -> list[UUID]:
        """Kahn topological sort. Returns execution order. Raises on cycle."""
        indeg: dict[UUID, int] = {k: 0 for k in self.nodes}
        for nid, node in self.nodes.items():
            for d in node.deps:
                if d not in self.nodes:
                    raise ValueError(f"OpGraph: missing dependency node {d}")
                indeg[nid] += 1

        if self.roots:
            q: list[UUID] = [
                n for n, deg in indeg.items() if deg == 0 and n in self.roots
            ]
        else:
            q = [n for n, deg in indeg.items() if deg == 0]

        order: list[UUID] = []
        while q:
            u = q.pop()
            order.append(u)
            for v, node in self.nodes.items():
                if u in node.deps:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)

        if len(order) != len(self.nodes):
            raise ValueError(
                f"OpGraph: cycle detected. Sorted {len(order)}/{len(self.nodes)} nodes."
            )
        return order

    def check_satisfiability(
        self, ambient: frozenset[str] = frozenset()
    ) -> list[tuple[UUID, frozenset[str]]]:
        """Static capability satisfiability check (#7).

        For each node, checks that morphism.requires ⊆ (ambient ∪ predecessors.provides).
        Returns list of (node_id, unsatisfied_requirements) for failing nodes.
        Empty list means the graph is satisfiable.

        Args:
            ambient: Capabilities available without any predecessor providing them
                (e.g., from Principal.rights()).
        """
        order = self.validate_dag()
        available: dict[UUID, frozenset[str]] = {}
        failures: list[tuple[UUID, frozenset[str]]] = []

        for nid in order:
            node = self.nodes[nid]
            pred_provides = frozenset().union(*(
                available.get(d, frozenset()) for d in node.deps
            )) if node.deps else frozenset()

            effective = ambient | pred_provides
            requires = frozenset(getattr(node.m, "requires", frozenset()))
            unsatisfied = frozenset(
                r for r in requires
                if not any(covers(h, r) for h in effective)
            )

            if unsatisfied:
                failures.append((nid, unsatisfied))

            provides = frozenset(getattr(node.m, "provides", frozenset()))
            available[nid] = pred_provides | provides

        return failures
