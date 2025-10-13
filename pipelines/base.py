import asyncio
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple, Union

from loguru import logger
from rich.console import Console

from agents.tracing.create import function_span
from agentz.utils.config import BaseConfig, resolve_config
from agentz.runner import (
    AgentExecutor,
    RuntimeTracker,
    HookRegistry,
    execute_tool_plan,
    execute_tools,
    run_manager_tool_loop,
    record_structured_payload,
    serialize_output,
)
from agentz.artifacts import RunReporter
from agentz.utils import Printer, get_experiment_timestamp
from pydantic import BaseModel


class BasePipeline:
    """Base class for all pipelines with common configuration and setup."""

    # Constants for iteration group IDs
    ITERATION_GROUP_PREFIX = "iter"
    FINAL_GROUP_ID = "iter-final"

    def __init__(self, config: Union[str, Path, Mapping[str, Any], BaseConfig]):
        """Initialize the pipeline using a single configuration input.

        Args:
            spec: Configuration specification:
                - str/Path: Load YAML/JSON file
                - dict with 'config_path': Load file, then deep-merge dict on top (dict wins)
                - dict without 'config_path': Use as-is
                - BaseConfig: Use as-is
            strict: Whether to strictly validate configuration (default: True).

        Examples:
            # Load from file
            BasePipeline("pipelines/configs/data_science.yaml")

            # Dict without config_path
            BasePipeline({"provider": "openai", "data": {"path": "data.csv"}})

            # Dict that patches a file (use 'config_path')
            BasePipeline({
                "config_path": "pipelines/configs/data_science.yaml",
                "data": {"path": "data/banana_quality.csv"},
                "user_prompt": "Custom prompt..."
            })

            # BaseConfig object
            BasePipeline(BaseConfig(provider="openai", data={"path": "data.csv"}))
        """
        self.console = Console()
        self._printer: Optional[Printer] = None
        self.reporter: Optional[RunReporter] = None

        # Resolve configuration using the new unified API
        self.config = resolve_config(config)

        # Generic pipeline settings
        self.experiment_id = get_experiment_timestamp()

        pipeline_settings = self.config.pipeline
        default_slug = self.__class__.__name__.replace("Pipeline", "").lower()
        self.pipeline_slug = (
            pipeline_settings.get("slug")
            or pipeline_settings.get("name")
            or default_slug
        )
        self.workflow_name = (
            pipeline_settings.get("workflow_name")
            or pipeline_settings.get("name")
        )
        if not self.workflow_name:
            # Default pattern: use class name + experiment_id
            pipeline_name = self.__class__.__name__.replace("Pipeline", "").lower()
            self.workflow_name = f"{pipeline_name}_{self.experiment_id}"

        self.verbose = pipeline_settings.get("verbose", True)
        self.max_iterations = pipeline_settings.get("max_iterations", 5)
        self.max_time_minutes = pipeline_settings.get("max_time_minutes", 10)

        # Research workflow name (optional, for pipelines with research components)
        self.research_workflow_name = pipeline_settings.get(
            "research_workflow_name",
            f"researcher_{self.experiment_id}",
        )

        # Iterative pipeline state
        self.iteration = 0
        self.start_time: Optional[float] = None
        self.should_continue = True
        self.constraint_reason = ""

        # Setup tracing configuration and logging
        self._setup_tracing()

        # Initialize runtime tracker and executor
        self._runtime_tracker: Optional[RuntimeTracker] = None
        self._executor: Optional[AgentExecutor] = None

        # Initialize hook registry
        self._hook_registry = HookRegistry()

    # ============================================
    # Core Properties
    # ============================================

    @property
    def enable_tracing(self) -> bool:
        """Get tracing enabled flag from config."""
        return self.config.pipeline.get("enable_tracing", True)

    @property
    def trace_sensitive(self) -> bool:
        """Get trace sensitive data flag from config."""
        return self.config.pipeline.get("trace_include_sensitive_data", False)

    @property
    def state(self) -> Optional[Any]:
        """Get pipeline state if available."""
        if hasattr(self, 'context') and hasattr(self.context, 'state'):
            return self.context.state
        return None

    @property
    def printer(self) -> Optional[Printer]:
        return self._printer

    @property
    def runtime_tracker(self) -> RuntimeTracker:
        """Get or create the runtime tracker."""
        if self._runtime_tracker is None:
            self._runtime_tracker = RuntimeTracker(
                printer=self.printer,
                enable_tracing=self.enable_tracing,
                trace_sensitive=self.trace_sensitive,
                iteration=self.iteration,
                experiment_id=self.experiment_id,
                reporter=self.reporter,
            )
        else:
            # Update iteration in existing tracker
            self._runtime_tracker.iteration = self.iteration
            self._runtime_tracker.printer = self.printer
            self._runtime_tracker.reporter = self.reporter
        return self._runtime_tracker

    @property
    def executor(self) -> AgentExecutor:
        """Get or create the agent executor."""
        # Refresh runtime tracker so iteration/printer stay in sync across loops
        tracker = self.runtime_tracker

        if self._executor is None:
            self._executor = AgentExecutor(tracker)
        else:
            # Executor holds a reference to the tracker; update it in case it changed
            self._executor.context = tracker
        return self._executor

    # ============================================
    # Printer & Reporter Management
    # ============================================

    def start_printer(self) -> Printer:
        if self._printer is None:
            self._printer = Printer(self.console)
        return self._printer

    def stop_printer(self) -> None:
        """Stop the live printer and finalize reporter if active."""
        if self._printer is not None:
            self._printer.end()
            self._printer = None
        if self.reporter is not None:
            self.reporter.finalize()
            self.reporter.print_terminal_report()

    def start_group(
        self,
        group_id: str,
        *,
        title: Optional[str] = None,
        border_style: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """Start a printer group and notify the reporter."""
        if self.reporter:
            self.reporter.record_group_start(
                group_id=group_id,
                title=title,
                border_style=border_style,
                iteration=iteration,
            )
        if self.printer:
            self.printer.start_group(
                group_id,
                title=title,
                border_style=border_style,
            )

    def end_group(
        self,
        group_id: str,
        *,
        is_done: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Mark a printer group complete and notify the reporter."""
        if self.reporter:
            self.reporter.record_group_end(
                group_id=group_id,
                is_done=is_done,
                title=title,
            )
        if self.printer:
            self.printer.end_group(
                group_id,
                is_done=is_done,
                title=title,
            )

    # ============================================
    # Initialization & Setup
    # ============================================

    def _initialize_run(self, additional_logging=None):
        """Initialize a pipeline run with logging, printer, and tracing.

        Args:
            additional_logging: Optional callable for pipeline-specific logging

        Returns:
            Trace context manager for the workflow
        """
        # Basic logging
        logger.info(
            f"Running {self.__class__.__name__} with experiment_id: {self.experiment_id}"
        )

        # Pipeline-specific logging
        if additional_logging:
            additional_logging()

        outputs_dir = Path(self.config.pipeline.get("outputs_dir", "outputs"))
        if self.reporter is None:
            self.reporter = RunReporter(
                base_dir=outputs_dir,
                pipeline_slug=self.pipeline_slug,
                workflow_name=self.workflow_name,
                experiment_id=self.experiment_id,
                console=self.console,
            )
        self.reporter.start(self.config)

        # Start printer and update workflow
        self.start_printer()
        if self.printer:
            self.printer.update_item(
                "workflow",
                f"Workflow: {self.workflow_name}",
                is_done=True,
                hide_checkmark=True,
            )

        # Create trace context
        trace_metadata = {
            "experiment_id": self.experiment_id,
            "includes_sensitive_data": "true" if self.trace_sensitive else "false",
        }
        return self.trace_context(self.workflow_name, metadata=trace_metadata)

    def _setup_tracing(self) -> None:
        """Setup tracing configuration with user-friendly output.

        Subclasses can override this method to add pipeline-specific information.
        """
        if self.enable_tracing:
            pipeline_name = self.__class__.__name__.replace("Pipeline", "")
            self.console.print(f"🍌 Starting {pipeline_name} Pipeline with Tracing")
            self.console.print(f"🔧 Provider: {self.config.provider}")
            self.console.print(f"🤖 Model: {self.config.llm.model_name}")
            self.console.print("🔍 Tracing: Enabled")
            self.console.print(
                f"🔒 Sensitive Data in Traces: {'Yes' if self.trace_sensitive else 'No'}"
            )
            self.console.print(f"🏷️ Workflow: {self.workflow_name}")
        else:
            pipeline_name = self.__class__.__name__.replace("Pipeline", "")
            self.console.print(f"🍌 Starting {pipeline_name} Pipeline")
            self.console.print(f"🔧 Provider: {self.config.provider}")
            self.console.print(f"🤖 Model: {self.config.llm.model_name}")

    def trace_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a trace context - delegates to RuntimeTracker."""
        return self.runtime_tracker.trace_context(name, metadata=metadata)

    def span_context(self, span_factory, **kwargs):
        """Create a span context - delegates to RuntimeTracker."""
        return self.runtime_tracker.span_context(span_factory, **kwargs)

    async def agent_step(self, *args, **kwargs) -> Any:
        """Run an agent with span tracking and optional output parsing.

        Delegates to AgentExecutor.agent_step(). See AgentExecutor.agent_step() for full documentation.
        """
        return await self.executor.agent_step(*args, **kwargs)

    def update_printer(self, *args, **kwargs) -> None:
        """Update printer status if printer is active.

        Delegates to RuntimeTracker.update_printer(). See RuntimeTracker.update_printer() for full documentation.
        """
        self.runtime_tracker.update_printer(*args, **kwargs)

    # ============================================
    # Context Managers & Utilities
    # ============================================

    @contextmanager
    def run_context(self, additional_logging: Optional[Callable] = None):
        """Context manager for run lifecycle handling.

        Manages trace context initialization, printer lifecycle, and cleanup.
        Automatically starts the pipeline timer for constraint checking.

        Args:
            additional_logging: Optional callable for pipeline-specific logging

        Yields:
            Trace context for the workflow
        """
        # Start pipeline timer for constraint checking
        self.start_time = time.time()

        trace_ctx = self._initialize_run(additional_logging)
        try:
            with trace_ctx:
                yield trace_ctx
        finally:
            self.stop_printer()

    async def run_span_step(self, *args, **kwargs) -> Any:
        """Execute a step with span context and printer updates.

        Delegates to AgentExecutor.run_span_step(). See AgentExecutor.run_span_step() for full documentation.
        """
        return await self.executor.run_span_step(*args, **kwargs)

    # ============================================
    # Iteration & Group Management
    # ============================================

    def begin_iteration(
        self,
        title: Optional[str] = None,
        border_style: str = "white"
    ) -> Tuple[Any, str]:
        """Begin a new iteration with its associated group.

        Combines context.begin_iteration() + start_group() into a single call.

        Args:
            title: Optional custom title (default: "Iteration {index}")
            border_style: Border style for the group (default: "white")

        Returns:
            Tuple of (iteration_record, group_id)
        """
        iteration, group_id = self.context.begin_iteration()
        self.iteration = iteration.index

        display_title = title or f"Iteration {iteration.index}"
        self.start_group(
            group_id,
            title=display_title,
            border_style=border_style,
            iteration=iteration.index,
        )

        return iteration, group_id

    def end_iteration(self, group_id: str, is_done: bool = True) -> None:
        """End the current iteration and its associated group.

        Combines context.mark_iteration_complete() + end_group() into a single call.

        Args:
            group_id: The group ID to close
            is_done: Whether the iteration completed successfully (default: True)
        """
        self.context.mark_iteration_complete()
        self.end_group(group_id, is_done=is_done)

    def begin_final_report(
        self,
        title: str = "Final Report",
        border_style: str = "white"
    ) -> str:
        """Begin the final report phase with its associated group.

        Combines context.begin_final_report() + start_group() into a single call.

        Args:
            title: Title for the final report group (default: "Final Report")
            border_style: Border style for the group (default: "white")

        Returns:
            The final report group_id
        """
        _, group_id = self.context.begin_final_report()
        self.start_group(group_id, title=title, border_style=border_style)
        return group_id

    def end_final_report(self, group_id: str, is_done: bool = True) -> None:
        """End the final report phase and its associated group.

        Combines context.mark_final_complete() + end_group() into a single call.

        Args:
            group_id: The final report group ID to close
            is_done: Whether the final report completed successfully (default: True)
        """
        self.context.mark_final_complete()
        self.end_group(group_id, is_done=is_done)

    def prepare_query(
        self,
        content: str,
        step_key: str = "prepare_query",
        span_name: str = "prepare_research_query",
        start_msg: str = "Preparing research query...",
        done_msg: str = "Research query prepared"
    ) -> str:
        """Prepare query/content with span context and printer updates.

        Args:
            content: The query/content to prepare
            step_key: Printer status key
            span_name: Name for the span
            start_msg: Start message for printer
            done_msg: Completion message for printer

        Returns:
            The prepared content
        """
        self.update_printer(step_key, start_msg)

        with self.span_context(function_span, name=span_name) as span:
            logger.debug(f"Prepared {span_name}: {content}")

            if span and hasattr(span, "set_output"):
                span.set_output({"output_preview": content[:200]})

        self.update_printer(step_key, done_msg, is_done=True)
        return content

    def _log_message(self, message: str) -> None:
        """Log a message using the configured logger."""
        logger.info(message)

    # ============================================
    # Execution Entry Points
    # ============================================

    def run_sync(self, *args, **kwargs):
        """Synchronous wrapper for the async run method."""
        return asyncio.run(self.run(*args, **kwargs))

    async def run(self, query: Any = None) -> Any:
        """Template method - DO NOT override in subclasses.

        This method provides the fixed lifecycle structure:
        1. Initialize pipeline
        2. Before execution hooks
        3. Main execution (delegated to execute())
        4. After execution hooks
        5. Finalization

        Override execute() instead to implement pipeline logic.

        Args:
            query: Optional query input (can be None for pipelines without input)

        Returns:
            Final result from finalize()
        """
        with self.run_context():
            # Phase 1: Setup
            await self.initialize_pipeline(query)

            # Phase 2: Pre-execution hooks
            await self.before_execution()
            await self._hook_registry.trigger("before_execution", context=self)

            # Phase 3: Main execution (delegated to subclass)
            result = await self.execute()

            # Phase 4: Post-execution hooks
            await self._hook_registry.trigger("after_execution", context=self, result=result)
            await self.after_execution(result)

            # Phase 5: Finalization
            final_result = await self.finalize(result)

            return final_result

    # ============================================
    # Lifecycle Hook Methods (Override in Subclasses)
    # ============================================

    async def initialize_pipeline(self, query: Any) -> None:
        """Initialize pipeline state and format query.

        Default implementation:
        - Formats query via format_query()
        - Sets state query
        - Updates printer status

        Override this for custom initialization logic.

        Args:
            query: Input query (can be None)
        """
        if query is not None:
            formatted_query = self.format_query(query)
            if self.state:
                self.state.set_query(formatted_query)
        self.update_printer("initialization", "Pipeline initialized", is_done=True)

    def format_query(self, query: Any) -> str:
        """Transform input query to formatted string.

        Default behavior (in order of priority):
        1. If query has a format() method, call it
        2. If query is a BaseModel, return model_dump_json()
        3. Otherwise, return str(query)

        Override this to customize query formatting.

        Args:
            query: Input query

        Returns:
            Formatted query string
        """
        if hasattr(query, 'format') and callable(getattr(query, 'format')):
            return query.format()
        if isinstance(query, BaseModel):
            return query.model_dump_json(indent=2)
        return str(query)

    async def before_execution(self) -> None:
        """Hook called before execute().

        Use for:
        - Data loading/validation
        - Resource initialization
        - Pre-flight checks

        Override this for custom pre-execution logic.
        """
        pass

    async def after_execution(self, result: Any) -> None:  # noqa: ARG002
        """Hook called after execute() completes.

        Use for:
        - Result validation
        - Cleanup operations
        - State aggregation

        Override this for custom post-execution logic.

        Args:
            result: The return value from execute()
        """
        pass

    async def finalize(self, result: Any) -> Any:
        """Finalization phase - prepare final return value.

        Default implementation:
        - Returns context.state.final_report if available
        - Otherwise returns result as-is

        Override this for custom finalization logic.

        Args:
            result: The return value from execute()

        Returns:
            Final result to return from run()
        """
        if self.state:
            return self.state.final_report
        return result

    # ============================================
    # Abstract Execute Method (Must Implement in Subclasses)
    # ============================================

    async def execute(self) -> Any:
        """Main execution logic - implement in subclass.

        This is where your pipeline logic goes. You have complete freedom:
        - Iterative loops (use run_iterative_loop helper)
        - Single-shot execution
        - Multi-phase workflows
        - Custom control flow (branching, conditional, parallel)
        - Mix of patterns

        Returns:
            Any result value (passed to after_execution and finalize)

        Examples:
            # Iterative pattern
            async def execute(self):
                return await self.run_iterative_loop(
                    iteration_body=self._do_iteration,
                    final_body=self._write_report
                )

            # Single-shot pattern
            async def execute(self):
                data = await self.load_data()
                analysis = await self.analyze(data)
                return await self.generate_report(analysis)

            # Multi-phase pattern
            async def execute(self):
                exploration = await self._explore_phase()
                if exploration.needs_deep_dive:
                    deep_dive = await self._deep_dive_phase()
                return await self._synthesize(exploration, deep_dive)
        """
        raise NotImplementedError("Subclasses must implement execute()")

    # ============================================
    # Workflow Helper Methods
    # ============================================

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
            should_continue: Optional custom condition (default: checks max iterations/time)

        Returns:
            Result from final_body if provided, else None
        """
        should_continue_fn = should_continue or self._should_continue_iteration

        while should_continue_fn():
            # Begin iteration with its group
            iteration, group_id = self.begin_iteration()

            # Trigger before hooks
            await self._hook_registry.trigger(
                "before_iteration",
                context=self.context,
                iteration=iteration,
                group_id=group_id
            )

            try:
                await iteration_body(iteration, group_id)
            finally:
                # Trigger after hooks
                await self._hook_registry.trigger(
                    "after_iteration",
                    context=self.context,
                    iteration=iteration,
                    group_id=group_id
                )
                # End iteration with its group
                self.end_iteration(group_id)

            # Check if state indicates completion
            if self.state and self.state.complete:
                break

        # Execute final body if provided
        result = None
        if final_body:
            final_group = self.begin_final_report()
            result = await final_body(final_group)
            self.end_final_report(final_group)

        return result

    def _should_continue_iteration(self) -> bool:
        """Check if iteration should continue based on constraints.

        Returns:
            True if should continue, False otherwise
        """
        # Check state completion
        if self.state and self.state.complete:
            return False

        # Check max iterations
        if self.iteration >= self.max_iterations:
            logger.info("\n=== Ending Iteration Loop ===")
            logger.info(f"Reached maximum iterations ({self.max_iterations})")
            return False

        # Check max time
        if self.start_time is not None:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.max_time_minutes:
                logger.info("\n=== Ending Iteration Loop ===")
                logger.info(f"Reached maximum time ({self.max_time_minutes} minutes)")
                return False

        return True

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
        """
        self.start_group(group_id, title=title, border_style=border_style)
        try:
            result = await body()
            return result
        finally:
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
        """
        async def run_step(name: str, fn: Callable):
            key = f"{group_id}:{name}" if group_id else name
            self.update_printer(key, f"Running {name}...", group_id=group_id)
            result = await fn()
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

    # ============================================
    # Integration with Runner Module
    # ============================================

    def _record_structured_payload(self, value: object, context_label: Optional[str] = None) -> None:
        """Record a structured payload to the current iteration state.

        Delegates to runner.utils.record_structured_payload.

        Args:
            value: The payload to record (typically a BaseModel instance)
            context_label: Optional label for debugging purposes
        """
        record_structured_payload(self.state, value, context_label)

    def _serialize_output(self, output: Any) -> str:
        """Serialize agent output to string for storage.

        Delegates to runner.utils.serialize_output.

        Args:
            output: The output to serialize (BaseModel, str, or other)

        Returns:
            String representation of the output
        """
        return serialize_output(output)

    async def execute_tool_plan(
        self,
        plan: Any,
        tool_agents: Dict[str, Any],
        group_id: str,
    ) -> None:
        """Execute a routing plan with tool agents.

        Delegates to runner.patterns.execute_tool_plan.

        Args:
            plan: AgentSelectionPlan with tasks to execute
            tool_agents: Dict mapping agent names to agent instances
            group_id: Group ID for printer updates
        """
        await execute_tool_plan(
            plan=plan,
            tool_agents=tool_agents,
            group_id=group_id,
            context=self.context,
            agent_step_fn=self.agent_step,
            update_printer_fn=self.update_printer,
        )

    async def _execute_tools(
        self,
        route_plan: Any,
        tool_agents: Dict[str, Any],
        group_id: str,
    ) -> None:
        """Execute tool agents based on routing plan.

        Delegates to runner.patterns.execute_tools.

        Args:
            route_plan: The routing plan (can be AgentSelectionPlan or other)
            tool_agents: Dict mapping agent names to agent instances
            group_id: Group ID for printer updates
        """
        await execute_tools(
            route_plan=route_plan,
            tool_agents=tool_agents,
            group_id=group_id,
            context=self.context,
            agent_step_fn=self.agent_step,
            update_printer_fn=self.update_printer,
        )

    # ============================================
    # High-Level Workflow Patterns
    # ============================================

    async def run_manager_tool_loop(
        self,
        manager_agents: Dict[str, Any],
        tool_agents: Dict[str, Any],
        workflow: List[str],
    ) -> Any:
        """Execute standard manager-tool iterative pattern.

        Delegates to runner.patterns.run_manager_tool_loop.

        This pattern implements: observe → evaluate → route → execute tools → repeat.

        Args:
            manager_agents: Dict of manager agents (observe, evaluate, routing, writer)
            tool_agents: Dict of tool agents
            workflow: List of manager agent names to execute in order (e.g., ["observe", "evaluate", "routing"])

        Returns:
            Result from final step
        """
        return await run_manager_tool_loop(
            manager_agents=manager_agents,
            tool_agents=tool_agents,
            workflow=workflow,
            context=self.context,
            agent_step_fn=self.agent_step,
            run_iterative_loop_fn=self.run_iterative_loop,
            update_printer_fn=self.update_printer,
        )

    # ============================================
    # Event Hook System
    # ============================================

    def register_hook(
        self,
        event: str,
        callback: Callable,
        priority: int = 0
    ) -> None:
        """Register a hook callback for an event.

        Delegates to HookRegistry.register.

        Args:
            event: Event name (before_execution, after_execution, before_iteration, after_iteration, etc.)
            callback: Callable or async callable
            priority: Execution priority (higher = earlier)

        Example:
            def log_iteration(pipeline, iteration, group_id):
                logger.info(f"Starting iteration {iteration.index}")

            pipeline.register_hook("before_iteration", log_iteration)
        """
        self._hook_registry.register(event, callback, priority)
