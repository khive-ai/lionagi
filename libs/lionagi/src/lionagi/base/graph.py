from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from .morphism import Morphism


@dataclass(slots=True)
class OpNode:
    id: UUID = field(default_factory=uuid4)
    m: Morphism = field(default=None)  # type: ignore
    deps: set[UUID] = field(default_factory=set)
    params: dict[str, Any] = field(default_factory=dict)  # <-- new


@dataclass(slots=True)
class OpGraph:
    nodes: dict[UUID, OpNode] = field(default_factory=dict)
    roots: set[UUID] = field(default_factory=set)

    def validate_dag(self) -> list[UUID]:
        """Kahn's algorithm for topological sorting; raises on cycle or invalid roots.

        Implements Kahn's 1962 algorithm for topological sort using in-degree counting.
        Guarantees DAG validation and returns nodes in dependency-respecting order.

        Algorithm reference:
        - Kahn, A. B. (1962). "Topological sorting of large networks"
        - Wikipedia: https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm

        Returns:
            Topologically sorted list of node IDs

        Raises:
            ValueError: If cycle detected or missing dependency nodes
        """
        indeg: dict[UUID, int] = {k: 0 for k in self.nodes}
        for nid, node in self.nodes.items():
            for d in node.deps:
                if d not in self.nodes:
                    raise ValueError(f"Missing dependency node: {d}")
                indeg[nid] += 1
        q: list[UUID] = [n for n, deg in indeg.items() if deg == 0 and n in self.roots]
        if not q and self.nodes:
            # if roots not provided, allow any 0-indegree as start
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
            raise ValueError("Cycle detected or invalid roots")
        return order
