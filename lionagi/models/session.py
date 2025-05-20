"""Session and Branch models for conversation state.

This module contains models for managing conversation state, including
Session (which manages multiple branches) and Branch (which contains
a sequence of messages).
"""

from typing import Any, Dict, List, Optional, Union

import lionfuncs.utils as utils
from pydantic import Field

from lionagi.models.base import BaseModel
from lionagi.models.message import Message


class Branch(BaseModel):
    """Model for a conversation branch containing messages."""

    id: str = Field(default_factory=lambda: utils.generate_uuid())
    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the branch.

        Args:
            message: The message to add.
        """
        self.messages.append(message)

    def get_messages(self) -> List[Message]:
        """Get all messages in the branch.

        Returns:
            A list of all messages in the branch.
        """
        return self.messages

    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the branch.

        Returns:
            The last message in the branch, or None if the branch is empty.
        """
        if not self.messages:
            return None
        return self.messages[-1]


class Session(BaseModel):
    """Model for a conversation session containing multiple branches."""

    id: str = Field(default_factory=lambda: utils.generate_uuid())
    branches: Dict[str, Branch] = Field(default_factory=dict)
    current_branch_id: Optional[str] = None
    default_service_config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def create_branch(self, **kwargs) -> Branch:
        """Create a new branch in the session.

        Args:
            **kwargs: Additional arguments to pass to the Branch constructor.

        Returns:
            The newly created branch.
        """
        branch = Branch(**kwargs)
        self.branches[branch.id] = branch
        if self.current_branch_id is None:
            self.current_branch_id = branch.id
        return branch

    def get_branch(self, branch_id: str) -> Branch:
        """Get a branch by ID.

        Args:
            branch_id: The ID of the branch to retrieve.

        Returns:
            The branch with the specified ID.

        Raises:
            KeyError: If the branch does not exist.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Branch with ID {branch_id} not found")
        return self.branches[branch_id]

    def get_current_branch(self) -> Optional[Branch]:
        """Get the current branch.

        Returns:
            The current branch, or None if no branch is set as current.
        """
        if self.current_branch_id is None:
            return None
        return self.branches.get(self.current_branch_id)

    def set_current_branch(self, branch_id: str) -> None:
        """Set the current branch.

        Args:
            branch_id: The ID of the branch to set as current.

        Raises:
            KeyError: If the branch does not exist.
        """
        if branch_id not in self.branches:
            raise KeyError(f"Branch with ID {branch_id} not found")
        self.current_branch_id = branch_id

    def add_message_to_current_branch(self, message: Message) -> None:
        """Add a message to the current branch.

        Args:
            message: The message to add.

        Raises:
            ValueError: If no branch is set as current.
        """
        current_branch = self.get_current_branch()
        if current_branch is None:
            raise ValueError("No current branch set")
        current_branch.add_message(message)


# Register adapters for the models
Branch.register_adapters()
Session.register_adapters()