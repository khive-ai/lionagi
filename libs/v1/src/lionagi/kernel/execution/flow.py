"""Flow: Composable execution patterns.

Ocean's v0 wisdom: Flows are reusable execution patterns.
"""

from typing import Any, List, Callable, Optional
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from ..foundation.contracts import Observable, Morphism, Invariant


@dataclass
class FlowStep:
    """A single step in a flow"""
    name: str
    func: Callable
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    capabilities_required: set[str] = field(default_factory=set)


class Flow(Observable):
    """Composable execution pattern.
    
    Flows define reusable patterns of execution that can be:
    - Composed together
    - Validated through IPU
    - Executed in branches
    """
    
    def __init__(self, name: str = None):
        self._id = uuid4()
        self.name = name or f"flow_{self._id}"
        self.steps: List[FlowStep] = []
        self.context: dict[str, Any] = {}
    
    @property
    def id(self) -> UUID:
        return self._id
    
    def add_step(self, step: FlowStep):
        """Add a step to the flow"""
        self.steps.append(step)
    
    def compose(self, other: 'Flow') -> 'Flow':
        """Compose with another flow"""
        new_flow = Flow(name=f"{self.name}+{other.name}")
        new_flow.steps = self.steps + other.steps
        return new_flow
    
    async def execute(self, branch: Any) -> Any:
        """Execute flow in a branch"""
        # TODO: Implement flow execution
        pass