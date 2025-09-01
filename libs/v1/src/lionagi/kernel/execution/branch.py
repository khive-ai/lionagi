"""Branch: Isolated execution context with capabilities.

Ocean's v0 wisdom: Branches provide isolated contexts for agent execution.
"""

from typing import Any, Optional, Set
from uuid import UUID, uuid4

from ..foundation.contracts import Observable
from ..foundation.capability import Capability
from ..safety.ipu import IPU
from ...domain.generic.progression import Progression


class Branch(Observable):
    """Isolated execution context with capability management.
    
    Each branch has:
    - Its own capabilities
    - Isolated state
    - Progress tracking
    - IPU validation
    """
    
    def __init__(self,
                 name: str = None,
                 capabilities: Set[str] = None,
                 parent: Optional['Branch'] = None,
                 ipu: IPU = None):
        self._id = uuid4()
        self.name = name or f"branch_{self._id}"
        self.capabilities = capabilities or set()
        self.parent = parent
        self.ipu = ipu or IPU()
        self.state = {}
        self.progression = Progression()
    
    @property
    def id(self) -> UUID:
        return self._id
    
    def has_capability(self, capability: str) -> bool:
        """Check if branch has a capability"""
        return capability in self.capabilities
    
    def grant_capability(self, capability: str):
        """Grant a new capability"""
        # TODO: Validate through IPU
        self.capabilities.add(capability)
    
    def revoke_capability(self, capability: str):
        """Revoke a capability"""
        self.capabilities.discard(capability)
    
    async def operate(self, instruction: Any, **kwargs) -> Any:
        """Execute an operation within this branch"""
        # Check capabilities
        required_caps = kwargs.get('required_capabilities', set())
        for cap in required_caps:
            if not self.has_capability(cap):
                raise PermissionError(f"Missing capability: {cap}")
        
        # TODO: Implement operation execution
        return "TODO: Implement operation"
    
    def fork(self) -> 'Branch':
        """Create a child branch"""
        return Branch(
            name=f"{self.name}_fork",
            capabilities=self.capabilities.copy(),
            parent=self,
            ipu=self.ipu
        )
    
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None
