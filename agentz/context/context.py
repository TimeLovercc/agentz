from __future__ import annotations

from agentz.context.conversation import BaseIterationRecord, ConversationState


class Context:
    """Central coordinator for conversation state and iteration management."""

    def __init__(self, state: ConversationState) -> None:
        """Initialize context engine with conversation state.

        Args:
            state: The conversation state to manage
        """
        self._state = state

    @property
    def state(self) -> ConversationState:
        return self._state

    def begin_iteration(self) -> BaseIterationRecord:
        """Start a new iteration and return its record."""
        return self._state.begin_iteration()

    def mark_iteration_complete(self) -> None:
        """Mark the current iteration as complete."""
        self._state.mark_iteration_complete()
