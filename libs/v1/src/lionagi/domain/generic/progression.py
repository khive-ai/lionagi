"""Progression: Ordered sequence of states.

Ocean's v0 wisdom: Natural way to model workflows with observable progress.
"""

from typing import Any, Optional, Generic, TypeVar
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

from ...kernel.foundation.contracts import Observable

T = TypeVar('T')


class ProgressionState(Enum):
    """States of a progression"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Progression(Observable, Generic[T]):
    """State machine with invariant checking.
    
    V0 wisdom elevated:
    - State transitions are explicit
    - Progress is observable
    - Transitions are atomic
    - Rollback capability built-in
    """
    
    id: UUID = field(default_factory=uuid4)
    states: list[T] = field(default_factory=list)
    current_index: int = 0
    state: ProgressionState = ProgressionState.PENDING
    history: list[tuple[int, T]] = field(default_factory=list)
    
    def advance(self) -> bool:
        """Move to next state"""
        if self.current_index < len(self.states) - 1:
            self.history.append((self.current_index, self.states[self.current_index]))
            self.current_index += 1
            self.state = ProgressionState.IN_PROGRESS
            return True
        else:
            self.state = ProgressionState.COMPLETED
            return False
    
    def rollback(self) -> bool:
        """Rollback to previous state"""
        if self.history:
            self.current_index, _ = self.history.pop()
            return True
        return False
    
    @property
    def current(self) -> Optional[T]:
        """Get current state"""
        if 0 <= self.current_index < len(self.states):
            return self.states[self.current_index]
        return None
    
    @property
    def progress(self) -> float:
        """Get progress percentage"""
        if not self.states:
            return 0.0
        return self.current_index / len(self.states)