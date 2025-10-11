from __future__ import annotations

from typing import Dict, List, Optional, Union

from agentz.context.conversation import BaseIterationRecord, ConversationState, create_conversation_state
from agentz.profiles.base import Profile, load_all_profiles


class Context:
    """Central coordinator for conversation state and iteration management."""

    def __init__(self, components: Union[ConversationState, List[str]]) -> None:
        """Initialize context engine with conversation state.

        Args:
            components: Either a ConversationState object (for backward compatibility)
                       or a list of component names to automatically initialize:
                       - "profiles": loads all profiles via load_all_profiles()
                       - "states": creates conversation state via create_conversation_state()

        Examples:
            # Automatic initialization
            context = Context(["profiles", "states"])

            # Manual initialization (backward compatible)
            state = create_conversation_state(profiles)
            context = Context(state)
        """
        self.profiles: Optional[Dict[str, Profile]] = None

        if isinstance(components, ConversationState):
            # Backward compatible: direct state initialization
            self._state = components
        elif isinstance(components, list):
            # Automatic initialization from component list
            if "profiles" in components:
                self.profiles = load_all_profiles()

            if "states" in components:
                if self.profiles is None:
                    raise ValueError("'states' requires 'profiles' to be initialized first. Include 'profiles' in the component list.")
                self._state = create_conversation_state(self.profiles)
            elif not hasattr(self, '_state'):
                # If no state requested, create empty state
                self._state = ConversationState()
        else:
            raise TypeError(f"components must be ConversationState or list, got {type(components)}")

    @property
    def state(self) -> ConversationState:
        return self._state

    def begin_iteration(self) -> BaseIterationRecord:
        """Start a new iteration and return its record.

        Automatically starts the conversation state timer on first iteration.
        """
        # Lazy timer start: start on first iteration if not already started
        if self._state.started_at is None:
            self._state.start_timer()
        return self._state.begin_iteration()

    def mark_iteration_complete(self) -> None:
        """Mark the current iteration as complete."""
        self._state.mark_iteration_complete()
