"""Element: The fundamental building block.

Ocean's v0 wisdom: Everything is an Element with identity and metadata.
"""

from typing import Any, Optional
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from ...kernel.foundation.contracts import Observable


@dataclass
class Element(Observable):
    """The atomic unit - an observable with metadata.
    
    V0 wisdom elevated:
    - Everything has identity
    - Elements can be composed
    - Metadata is typed and validated
    """
    
    id: UUID = field(default_factory=uuid4)
    content: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()
    
    def with_metadata(self, key: str, value: Any) -> 'Element':
        """Add metadata - returns new element (immutable pattern)"""
        new_meta = {**self.metadata, key: value}
        return Element(
            id=self.id,
            content=self.content,
            metadata=new_meta
        )
    
    def compose(self, other: 'Element') -> 'Element':
        """Compose with another element"""
        # TODO: Implement composition logic
        pass