"""Iteration management for iterative workflows."""

from typing import Any, Callable, Optional

from agentz.context.engine import ContextEngine


class IterationManager:
    """Manages iteration lifecycle for iterative workflows.

    Handles:
    - Iteration begin/end
    - Loop condition checking
    - Printer group visualization
    - State management

    This class is purely about iteration control - it has no knowledge
    of agents or behaviors, maintaining clean separation of concerns.
    """

    def __init__(
        self,
        engine: ContextEngine,
        loop_condition: Callable[[ContextEngine], bool],
        pipeline: Any,  # Pipeline for printer access
        after_iteration: Optional[Callable[[ContextEngine], None]] = None,
    ):
        """Initialize iteration manager.

        Args:
            engine: ContextEngine for state management
            loop_condition: Callable that determines if iteration should continue
            pipeline: Pipeline instance for printer updates
            after_iteration: Optional callback after each iteration
        """
        self.engine = engine
        self.loop_condition = loop_condition
        self.pipeline = pipeline
        self.after_iteration = after_iteration

    def should_continue(self) -> bool:
        """Check if iteration should continue.

        Returns:
            True if loop should continue, False otherwise
        """
        return self.loop_condition(self.engine)

    def begin_iteration(self) -> Any:
        """Begin a new iteration.

        Returns:
            The new iteration object from engine
        """
        iteration = self.engine.begin_iteration()
        iteration_group = f"iter-{iteration.index}"

        # Update pipeline iteration tracking
        self.pipeline.iteration = iteration.index

        # Start printer group
        self.pipeline.start_group(
            iteration_group,
            title=f"Iteration {iteration.index}",
            border_style="white",
            iteration=iteration.index,
        )

        return iteration, iteration_group

    def end_iteration(self, iteration_group: str) -> None:
        """End the current iteration.

        Args:
            iteration_group: The iteration group identifier
        """
        self.engine.mark_iteration_complete()

        # Call optional callback
        if self.after_iteration:
            self.after_iteration(self.engine)

        # End printer group
        self.pipeline.end_group(iteration_group, is_done=True)

    def start_final_group(self) -> str:
        """Start the final group for post-iteration work.

        Returns:
            The final group identifier
        """
        final_group = "iter-final"
        self.pipeline.start_group(
            final_group,
            title="Final Report",
            border_style="white",
        )
        return final_group

    def end_final_group(self, final_group: str) -> None:
        """End the final group.

        Args:
            final_group: The final group identifier
        """
        self.pipeline.end_group(final_group, is_done=True)
