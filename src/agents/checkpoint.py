"""
Checkpointing mechanism for agent sessions.
Allows saving and loading agent state to support long-running, stateful interactions.
"""
from __future__ import annotations # For Python 3.7, 3.8 compatibility

from abc import ABC, abstractmethod
from typing import Optional, Dict

from .state import SessionState # Import SessionState from the new state.py

class BaseCheckpointManager(ABC):
    """
    Abstract base class for checkpoint managers.
    Defines the interface for saving and loading agent session states.
    """

    @abstractmethod
    async def load_state(self, thread_id: str) -> Optional[SessionState]:
        """
        Loads the session state for a given thread_id.

        Args:
            thread_id: The unique identifier for the conversation thread.

        Returns:
            The SessionState if found, otherwise None.
        """
        pass

    @abstractmethod
    async def save_state(self, thread_id: str, state: SessionState) -> None:
        """
        Saves the session state for a given thread_id.

        Args:
            thread_id: The unique identifier for the conversation thread.
            state: The SessionState object to save.
        """
        pass

    @abstractmethod
    async def delete_state(self, thread_id: str) -> bool:
        """
        Deletes the session state for a given thread_id.

        Args:
            thread_id: The unique identifier for the conversation thread.

        Returns:
            True if state was deleted, False if not found.
        """
        pass


class InMemoryCheckpointManager(BaseCheckpointManager):
    """
    An in-memory implementation of the checkpoint manager.
    Stores checkpoints in a Python dictionary. Suitable for single-instance deployments
    and testing. Not persistent across application restarts.
    """
    def __init__(self):
        self._checkpoints: Dict[str, SessionState] = {}
        # For potential concurrency control if needed, though Pydantic models are generally not thread-safe for mutation
        # self._lock = asyncio.Lock()

    async def load_state(self, thread_id: str) -> Optional[SessionState]:
        """Loads state, returning a deep copy to prevent shared mutable state issues."""
        # async with self._lock: # Uncomment if direct mutation of _checkpoints items is a concern elsewhere
        state = self._checkpoints.get(thread_id)
        if state:
            # Pydantic's model_copy(deep=True) is the new way for deep copies in v2
            # For older Pydantic, state.copy(deep=True)
            return state.model_copy(deep=True)
        return None

    async def save_state(self, thread_id: str, state: SessionState) -> None:
        """Saves state, storing a deep copy."""
        # async with self._lock:
        if not isinstance(state, SessionState):
            raise TypeError("state must be an instance of SessionState")
        # Store a deep copy to prevent issues if the original 'state' object is modified elsewhere
        self._checkpoints[thread_id] = state.model_copy(deep=True)

    async def delete_state(self, thread_id: str) -> bool:
        """Deletes state for the given thread_id."""
        # async with self._lock:
        if thread_id in self._checkpoints:
            del self._checkpoints[thread_id]
            return True
        return False

```
