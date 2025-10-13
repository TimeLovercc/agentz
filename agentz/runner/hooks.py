"""Hook registry system for event-driven pipeline extensibility."""

import asyncio
from typing import Any, Callable, Dict, List, Tuple

from loguru import logger


class HookRegistry:
    """Registry for managing pipeline event hooks.

    Supports registering callbacks for various lifecycle events and triggering
    them with priority ordering.
    """

    DEFAULT_EVENTS = [
        "before_execution",
        "after_execution",
        "before_iteration",
        "after_iteration",
        "before_agent_step",
        "after_agent_step",
    ]

    def __init__(self, events: List[str] = None):
        """Initialize hook registry.

        Args:
            events: List of event names to support (defaults to DEFAULT_EVENTS)
        """
        event_list = events or self.DEFAULT_EVENTS
        self._hooks: Dict[str, List[Tuple[int, Callable]]] = {
            event: [] for event in event_list
        }

    def register(
        self,
        event: str,
        callback: Callable,
        priority: int = 0
    ) -> None:
        """Register a hook callback for an event.

        Args:
            event: Event name
            callback: Callable or async callable
            priority: Execution priority (higher = earlier)

        Raises:
            ValueError: If event is not recognized

        Example:
            def log_iteration(pipeline, iteration, group_id):
                logger.info(f"Starting iteration {iteration.index}")

            registry.register("before_iteration", log_iteration)
        """
        if event not in self._hooks:
            raise ValueError(
                f"Unknown hook event: {event}. Valid events: {list(self._hooks.keys())}"
            )

        self._hooks[event].append((priority, callback))
        # Sort by priority (descending)
        self._hooks[event].sort(key=lambda x: -x[0])

    async def trigger(self, event: str, context: Any, **kwargs) -> None:
        """Trigger all registered hooks for an event.

        Args:
            event: Event name
            context: Context object to pass to callbacks (typically the pipeline)
            **kwargs: Additional arguments to pass to hook callbacks
        """
        if event not in self._hooks:
            return

        for priority, callback in self._hooks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(context, **kwargs)
                else:
                    callback(context, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {callback.__name__} for {event} failed: {e}")

    def has_hooks(self, event: str) -> bool:
        """Check if any hooks are registered for an event.

        Args:
            event: Event name

        Returns:
            True if hooks are registered, False otherwise
        """
        return event in self._hooks and len(self._hooks[event]) > 0

    def clear_event(self, event: str) -> None:
        """Clear all hooks for a specific event.

        Args:
            event: Event name
        """
        if event in self._hooks:
            self._hooks[event].clear()

    def clear_all(self) -> None:
        """Clear all registered hooks."""
        for event in self._hooks:
            self._hooks[event].clear()
