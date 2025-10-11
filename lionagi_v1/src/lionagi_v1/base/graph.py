from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set
from uuid import UUID, uuid4

from .morphism import Morphism


@dataclass(slots=True)
class OpNode:
    id: UUID = field(default_factory=uuid4)
    m: Morphism = field(default=None)  # type: ignore
    deps: Set[UUID] = field(default_factory=set)
    params: Dict[str, Any] = field(default_factory=dict)  # <-- new


@dataclass(slots=True)
class OpGraph:
    nodes: Dict[UUID, OpNode] = field(default_factory=dict)
    roots: Set[UUID] = field(default_factory=set)

    def validate_dag(self) -> List[UUID]:
        """Kahn topological sort; raises on cycle or invalid roots."""
        indeg: Dict[UUID, int] = {k: 0 for k in self.nodes}
        for nid, node in self.nodes.items():
            for d in node.deps:
                if d not in self.nodes:
                    raise ValueError(f"Missing dependency node: {d}")
                indeg[nid] += 1
        q: List[UUID] = [
            n for n, deg in indeg.items() if deg == 0 and n in self.roots
        ]
        if not q and self.nodes:
            # if roots not provided, allow any 0-indegree as start
            q = [n for n, deg in indeg.items() if deg == 0]
        order: List[UUID] = []
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
