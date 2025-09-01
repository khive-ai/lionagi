"""Event: Things that happen in the system.

Ocean's v0 wisdom: Events are immutable records with causality chains.
"""

from typing import Any, Optional
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import time

from ...kernel.foundation.contracts import Observable


@dataclass(frozen=True)
class Event(Observable):
    """Immutable record of something that happened.
    
    V0 wisdom elevated:
    - Events are immutable
    - Events have causality chains
    - Events trigger effects
    - Guaranteed delivery through Observable protocol
    """
    
    id: UUID = field(default_factory=uuid4)
    type: str = "event"
    source: UUID = None
    target: UUID = None
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)
    causality: tuple[UUID, ...] = field(default_factory=tuple)
    
    def caused_by(self, event: 'Event') -> 'Event':
        """Create new event with causality chain"""
        new_causality = (*self.causality, event.id)
        return Event(
            type=self.type,
            source=self.source,
            target=self.target,
            data=self.data,
            causality=new_causality
        )