"""High-level runtime execution patterns for agent orchestration."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

from loguru import logger

from agentz.flow.context import ExecutionContext
from agentz.flow.executor import AgentExecutor, AgentStep


class ExecutionPattern(ABC):
    """Base class for execution patterns."""

    def __init__(self, context: ExecutionContext):
        """Initialize pattern with execution context.

        Args:
            context: ExecutionContext for tracing, printing, etc.
        """
        self.context = context
        self.executor = AgentExecutor(context)

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Run the pattern. Subclasses must implement."""
        pass


class SequentialPattern(ExecutionPattern):
    """Execute agent steps one after another, optionally passing output to next.

    Example:
        steps = [
            AgentStep(agent=loader, instructions="Load data", span_name="load"),
            AgentStep(agent=analyzer, instructions="Analyze data", span_name="analyze")
        ]
        pattern = SequentialPattern(context)
        results = await pattern.run(steps)
    """

    async def run(self, steps: List[AgentStep], pass_output: bool = False) -> List[Any]:
        """Execute steps sequentially.

        Args:
            steps: List of AgentStep instances to execute
            pass_output: If True, pass each step's output to next step's instructions

        Returns:
            List of results from each step
        """
        results = []
        previous_output = None

        for i, step in enumerate(steps):
            # Optionally inject previous output into instructions
            if pass_output and previous_output and i > 0:
                if callable(step.instructions):
                    original_fn = step.instructions
                    step.instructions = lambda: f"{original_fn()}\n\nPrevious output:\n{previous_output}"
                else:
                    step.instructions = f"{step.instructions}\n\nPrevious output:\n{previous_output}"

            result = await self.executor.execute_step(step)
            results.append(result)
            previous_output = result

        return results


class ParallelPattern(ExecutionPattern):
    """Execute multiple agent steps concurrently and aggregate results.

    Example:
        steps = [
            AgentStep(agent=agent1, instructions="Task 1", span_name="task1"),
            AgentStep(agent=agent2, instructions="Task 2", span_name="task2")
        ]
        pattern = ParallelPattern(context)
        results = await pattern.run(steps)
    """

    async def run(self, steps: List[AgentStep]) -> List[Any]:
        """Execute steps in parallel.

        Args:
            steps: List of AgentStep instances to execute concurrently

        Returns:
            List of results in the same order as input steps
        """
        tasks = [self.executor.execute_step(step) for step in steps]
        return await asyncio.gather(*tasks)


class ConditionalPattern(ExecutionPattern):
    """Execute agents based on conditions/predicates.

    Example:
        def check_data_size(data):
            return len(data) > 1000

        pattern = ConditionalPattern(context)
        result = await pattern.run(
            condition=check_data_size,
            condition_args=(data,),
            if_true=AgentStep(agent=big_data_agent, ...),
            if_false=AgentStep(agent=small_data_agent, ...)
        )
    """

    async def run(
        self,
        condition: Callable[..., bool],
        if_true: AgentStep,
        if_false: Optional[AgentStep] = None,
        condition_args: tuple = (),
        condition_kwargs: dict = None
    ) -> Any:
        """Execute step based on condition.

        Args:
            condition: Callable that returns bool
            if_true: Step to execute if condition is True
            if_false: Optional step to execute if condition is False
            condition_args: Positional args for condition callable
            condition_kwargs: Keyword args for condition callable

        Returns:
            Result from executed step, or None if condition False and no if_false
        """
        condition_kwargs = condition_kwargs or {}
        should_execute_true = condition(*condition_args, **condition_kwargs)

        if should_execute_true:
            return await self.executor.execute_step(if_true)
        elif if_false:
            return await self.executor.execute_step(if_false)
        return None


class LoopPattern(ExecutionPattern):
    """Execute agent step repeatedly until condition met.

    This pattern is useful for iterative research or refinement workflows.

    Example:
        def should_continue(iteration, result):
            return iteration < 5 and not result.get("complete")

        pattern = LoopPattern(context)
        results = await pattern.run(
            step=AgentStep(agent=research_agent, ...),
            should_continue=should_continue,
            max_iterations=10
        )
    """

    async def run(
        self,
        step: AgentStep,
        should_continue: Callable[[int, Any], bool],
        max_iterations: int = 10,
        max_time_minutes: Optional[float] = None
    ) -> List[Any]:
        """Execute step in a loop until condition met.

        Args:
            step: AgentStep to execute repeatedly
            should_continue: Callable(iteration, last_result) -> bool
            max_iterations: Maximum number of iterations
            max_time_minutes: Optional time limit in minutes

        Returns:
            List of results from each iteration
        """
        results = []
        start_time = time.time()
        iteration = 0

        while iteration < max_iterations:
            # Check time constraint
            if max_time_minutes:
                elapsed_minutes = (time.time() - start_time) / 60
                if elapsed_minutes >= max_time_minutes:
                    logger.info(f"Loop terminated: reached time limit ({max_time_minutes} min)")
                    break

            # Update context iteration
            self.context.iteration = iteration

            # Execute step
            result = await self.executor.execute_step(step)
            results.append(result)

            # Check continuation condition
            iteration += 1
            if not should_continue(iteration, result):
                logger.info(f"Loop terminated: condition not met after iteration {iteration}")
                break

        return results


class RetryPattern(ExecutionPattern):
    """Retry agent execution with exponential backoff on failure.

    Example:
        pattern = RetryPattern(context)
        result = await pattern.run(
            step=AgentStep(agent=flaky_agent, ...),
            max_retries=3,
            backoff_base=2.0
        )
    """

    async def run(
        self,
        step: AgentStep,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        exceptions: tuple = (Exception,)
    ) -> Any:
        """Execute step with retry logic.

        Args:
            step: AgentStep to execute
            max_retries: Maximum number of retry attempts
            backoff_base: Base for exponential backoff (seconds)
            exceptions: Tuple of exception types to catch and retry

        Returns:
            Result from successful execution

        Raises:
            Last exception if all retries exhausted
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await self.executor.execute_step(step)
            except exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = backoff_base ** attempt
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time:.1f}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed")

        raise last_exception


class PipelinePattern(ExecutionPattern):
    """Compose multiple patterns together.

    This allows building complex execution flows by combining simpler patterns.

    Example:
        # Create a pipeline that runs parallel steps, then sequential refinement
        pattern = PipelinePattern(context)
        result = await pattern.run([
            ParallelPattern(context).run(parallel_steps),
            SequentialPattern(context).run(refinement_steps)
        ])
    """

    async def run(self, patterns: List[Union[ExecutionPattern, Callable]]) -> List[Any]:
        """Execute multiple patterns in sequence.

        Args:
            patterns: List of pattern run coroutines or callables

        Returns:
            List of results from each pattern
        """
        results = []

        for pattern_or_callable in patterns:
            if asyncio.iscoroutine(pattern_or_callable):
                result = await pattern_or_callable
            elif callable(pattern_or_callable):
                result = await pattern_or_callable()
            else:
                result = pattern_or_callable

            results.append(result)

        return results
