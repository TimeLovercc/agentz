"""Workflow control flow helpers for pipeline execution."""

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union


class WorkflowHelpers:
    """Collection of workflow control flow helpers.

    These helpers provide common patterns for pipeline execution:
    - Iterative loops
    - Custom groups
    - Parallel execution
    - Conditional execution
    - Loop-until patterns
    """

    def __init__(
        self,
        iteration_manager: Any,
        hook_registry: Any,
        start_group_callback: Optional[Callable] = None,
        end_group_callback: Optional[Callable] = None,
        update_printer_callback: Optional[Callable] = None,
    ):
        """Initialize workflow helpers.

        Args:
            iteration_manager: IterationManager instance
            hook_registry: HookRegistry instance
            start_group_callback: Optional callback(group_id, title, border_style)
            end_group_callback: Optional callback(group_id, is_done)
            update_printer_callback: Optional callback(key, message, is_done, group_id)
        """
        self.iteration_manager = iteration_manager
        self.hook_registry = hook_registry
        self.start_group = start_group_callback
        self.end_group = end_group_callback
        self.update_printer = update_printer_callback

    async def run_iterative_loop(
        self,
        iteration_body: Callable[[Any, str], Awaitable[Any]],
        final_body: Optional[Callable[[str], Awaitable[Any]]] = None,
        should_continue: Optional[Callable[[], bool]] = None,
    ) -> Any:
        """Execute standard iterative loop pattern.

        Args:
            iteration_body: Async function(iteration, group_id) -> result
            final_body: Optional async function(final_group_id) -> result
            should_continue: Optional custom condition (default: iteration_manager.should_continue_iteration)

        Returns:
            Result from final_body if provided, else None

        Example:
            async def my_iteration(iteration, group):
                observations = await observe_agent(...)
                evaluations = await evaluate_agent(...)
                await route_and_execute(evaluations, group)

            async def my_final(group):
                return await writer_agent(...)

            result = await helpers.run_iterative_loop(
                iteration_body=my_iteration,
                final_body=my_final
            )
        """
        should_continue_fn = should_continue or self.iteration_manager.should_continue_iteration

        while should_continue_fn():
            iteration, group_id = self.iteration_manager.begin_iteration()

            await self.hook_registry.trigger(
                "before_iteration",
                context=self.iteration_manager.context,
                iteration=iteration,
                group_id=group_id
            )

            try:
                await iteration_body(iteration, group_id)
            finally:
                await self.hook_registry.trigger(
                    "after_iteration",
                    context=self.iteration_manager.context,
                    iteration=iteration,
                    group_id=group_id
                )
                self.iteration_manager.end_iteration(group_id)

            # Check if state indicates completion
            context = self.iteration_manager.context
            if hasattr(context, 'state') and context.state and context.state.complete:
                break

        result = None
        if final_body:
            final_group = self.iteration_manager.start_final_group()
            result = await final_body(final_group)
            self.iteration_manager.end_final_group(final_group)

        return result

    async def run_custom_group(
        self,
        group_id: str,
        title: str,
        body: Callable[[], Awaitable[Any]],
        border_style: str = "white",
    ) -> Any:
        """Execute code within a custom printer group.

        Args:
            group_id: Unique group identifier
            title: Display title for the group
            body: Async function to execute within group
            border_style: Border color for printer

        Returns:
            Result from body()

        Example:
            exploration = await helpers.run_custom_group(
                "exploration",
                "Exploration Phase",
                self._explore
            )

            analysis = await helpers.run_custom_group(
                "analysis",
                "Deep Analysis",
                lambda: self._analyze(exploration)
            )
        """
        if self.start_group:
            self.start_group(group_id, title=title, border_style=border_style)
        try:
            result = await body()
            return result
        finally:
            if self.end_group:
                self.end_group(group_id, is_done=True)

    async def run_parallel_steps(
        self,
        steps: Dict[str, Callable[[], Awaitable[Any]]],
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute multiple steps in parallel.

        Args:
            steps: Dict mapping step_name -> async callable
            group_id: Optional group to nest steps in

        Returns:
            Dict mapping step_name -> result

        Example:
            results = await helpers.run_parallel_steps({
                "data_loading": self.load_data,
                "validation": self.validate_inputs,
                "model_init": self.initialize_models,
            })

            data = results["data_loading"]
        """
        async def run_step(name: str, fn: Callable):
            key = f"{group_id}:{name}" if group_id else name
            if self.update_printer:
                self.update_printer(key, f"Running {name}...", group_id=group_id)
            result = await fn()
            if self.update_printer:
                self.update_printer(key, f"Completed {name}", is_done=True, group_id=group_id)
            return name, result

        tasks = [run_step(name, fn) for name, fn in steps.items()]
        completed = await asyncio.gather(*tasks)
        return dict(completed)

    async def run_if(
        self,
        condition: Union[bool, Callable[[], bool]],
        body: Callable[[], Awaitable[Any]],
        else_body: Optional[Callable[[], Awaitable[Any]]] = None,
    ) -> Any:
        """Conditional execution helper.

        Args:
            condition: Boolean or callable returning bool
            body: Execute if condition is True
            else_body: Optional execute if condition is False

        Returns:
            Result from executed body

        Example:
            initial = await quick_check()

            return await helpers.run_if(
                condition=initial.needs_deep_analysis,
                body=lambda: deep_analysis(initial),
                else_body=lambda: simple_report(initial)
            )
        """
        cond_result = condition() if callable(condition) else condition
        if cond_result:
            return await body()
        elif else_body:
            return await else_body()
        return None

    async def run_until(
        self,
        condition: Callable[[], bool],
        body: Callable[[int], Awaitable[Any]],
        max_iterations: Optional[int] = None,
    ) -> List[Any]:
        """Execute body repeatedly until condition is met.

        Args:
            condition: Callable returning True to stop
            body: Async function(iteration_number) -> result
            max_iterations: Optional max iterations (default: unlimited)

        Returns:
            List of results from each iteration

        Example:
            results = await helpers.run_until(
                condition=lambda: context.state.complete,
                body=self._exploration_step,
                max_iterations=10
            )
            return aggregate(results)
        """
        results = []
        iteration = 0

        while not condition():
            if max_iterations and iteration >= max_iterations:
                break

            result = await body(iteration)
            results.append(result)
            iteration += 1

        return results
