"""Iteration management for iterative pipeline workflows."""

import time
from typing import Any, Optional, Tuple

from loguru import logger


class IterationManager:
    """Manages iteration lifecycle for iterative pipelines.

    Handles:
    - Starting/ending iterations with printer groups
    - Constraint checking (max iterations, max time)
    - Final group management
    """

    # Constants for iteration group IDs
    ITERATION_GROUP_PREFIX = "iter"
    FINAL_GROUP_ID = "iter-final"

    def __init__(
        self,
        context: Any,
        max_iterations: int = 5,
        max_time_minutes: float = 10,
        start_group_callback: Optional[callable] = None,
        end_group_callback: Optional[callable] = None,
    ):
        """Initialize iteration manager.

        Args:
            context: Pipeline context with state and begin_iteration/mark_iteration_complete methods
            max_iterations: Maximum number of iterations allowed
            max_time_minutes: Maximum time in minutes
            start_group_callback: Optional callback(group_id, title, border_style, iteration)
            end_group_callback: Optional callback(group_id, is_done, title)
        """
        self.context = context
        self.max_iterations = max_iterations
        self.max_time_minutes = max_time_minutes
        self.start_group_callback = start_group_callback
        self.end_group_callback = end_group_callback

        self.start_time: Optional[float] = None
        self.current_iteration: int = 0

    def start_timer(self) -> None:
        """Start the iteration timer for constraint checking."""
        self.start_time = time.time()

    def begin_iteration(self) -> Tuple[Any, str]:
        """Begin a new iteration with printer group.

        Returns:
            Tuple of (iteration_record, group_id)
        """
        iteration = self.context.begin_iteration()
        group_id = f"{self.ITERATION_GROUP_PREFIX}-{iteration.index}"

        self.current_iteration = iteration.index

        if self.start_group_callback:
            self.start_group_callback(
                group_id,
                title=f"Iteration {iteration.index}",
                border_style="white",
                iteration=iteration.index,
            )

        return iteration, group_id

    def end_iteration(self, group_id: str) -> None:
        """End the current iteration and close printer group.

        Args:
            group_id: Group ID to close
        """
        self.context.mark_iteration_complete()

        if self.end_group_callback:
            self.end_group_callback(group_id, is_done=True)

    def start_final_group(self) -> str:
        """Start final group for post-iteration work.

        Returns:
            Final group ID
        """
        if self.start_group_callback:
            self.start_group_callback(
                self.FINAL_GROUP_ID,
                title="Final Report",
                border_style="white",
            )
        return self.FINAL_GROUP_ID

    def end_final_group(self, group_id: str) -> None:
        """End final group.

        Args:
            group_id: Group ID to close
        """
        if self.end_group_callback:
            self.end_group_callback(group_id, is_done=True)

    def should_continue_iteration(self) -> bool:
        """Check if iteration should continue.

        Checks:
        - State not complete
        - Within max iterations
        - Within max time

        Returns:
            True if should continue, False otherwise
        """
        # Check state completion
        if hasattr(self.context, 'state') and self.context.state and self.context.state.complete:
            return False

        return self.check_constraints()

    def check_constraints(self) -> bool:
        """Check if we've exceeded our constraints (max iterations or time).

        Returns:
            True if within constraints, False if exceeded
        """
        if self.current_iteration >= self.max_iterations:
            logger.info("\n=== Ending Iteration Loop ===")
            logger.info(f"Reached maximum iterations ({self.max_iterations})")
            return False

        if self.start_time is not None:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.max_time_minutes:
                logger.info("\n=== Ending Iteration Loop ===")
                logger.info(f"Reached maximum time ({self.max_time_minutes} minutes)")
                return False

        return True
