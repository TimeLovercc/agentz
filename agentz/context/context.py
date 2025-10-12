from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from agentz.context.conversation import BaseIterationRecord, ConversationState, create_conversation_state
from agentz.profiles.base import Profile, load_all_profiles


class Context:
    """Central coordinator for conversation state and iteration management."""

    # Constants for iteration group IDs
    ITERATION_GROUP_PREFIX = "iter"
    FINAL_GROUP_ID = "iter-final"

    def __init__(
        self,
        components: Union[ConversationState, List[str]]
    ) -> None:
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

    def begin_iteration(self) -> Tuple[BaseIterationRecord, str]:
        """Start a new iteration and return its record with group_id.

        Automatically starts the conversation state timer on first iteration.

        Returns:
            Tuple of (iteration_record, group_id) where group_id follows the pattern "iter-{index}"
        """
        # Lazy timer start: start on first iteration if not already started
        if self._state.started_at is None:
            self._state.start_timer()

        iteration = self._state.begin_iteration()
        group_id = f"{self.ITERATION_GROUP_PREFIX}-{iteration.index}"

        return iteration, group_id

    def mark_iteration_complete(self) -> None:
        """Mark the current iteration as complete."""
        self._state.mark_iteration_complete()

    def begin_final_report(self) -> Tuple[None, str]:
        """Begin final report phase and return group_id.

        Returns:
            Tuple of (None, group_id) where group_id is the final report group ID
        """
        return None, self.FINAL_GROUP_ID

    def mark_final_complete(self) -> None:
        """Mark final report as complete."""
        pass  # No state change needed for final report
