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
)
from agentz.reporting import RunReporter
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

        # Hook registry for event-driven extensibility
        self._hooks: Dict[str, List[Tuple[int, Callable]]] = {
            "before_execution": [],
            "after_execution": [],
            "before_iteration": [],
            "after_iteration": [],
            "before_agent_step": [],
            "after_agent_step": [],
        }

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

    def start_printer(self) -> Printer:
        if self._printer is None:
            self._printer = Printer(self.console)
        return self._printer

    def stop_printer(self) -> None:
        if self._printer is not None:
            self._printer.end()
            self._printer = None

    def _start_printer(self) -> None:
        """Create and attach a live status printer for this run."""
        if self.printer is None:
            self.start_printer()

    def _stop_printer(self) -> None:
        """Stop the live printer if it's currently active."""
        if self.printer is not None:
            self.stop_printer()
        if self.reporter is not None:
            self.reporter.finalize()
            self.reporter.print_terminal_report()

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
        self._start_printer()
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

    async def agent_step(
        self,
        agent,
        instructions: str,
        span_name: str,
        span_type: str = "agent",
        output_model: Optional[type[BaseModel]] = None,
        sync: bool = False,
        printer_key: Optional[str] = None,
        printer_title: Optional[str] = None,
        printer_group_id: Optional[str] = None,
        printer_border_style: Optional[str] = None,
        **span_kwargs
    ) -> Any:
        """Run an agent with span tracking and optional output parsing.

        This method delegates to AgentExecutor from the flow module.

        Args:
            agent: The agent to run
            instructions: Instructions/prompt for the agent
            span_name: Name for the span
            span_type: Type of span - "agent" or "function"
            output_model: Optional pydantic model to parse output
            sync: Whether to run synchronously
            printer_key: Optional key for printer updates (will be prefixed with iter:N:)
            printer_title: Optional title for printer display
            printer_group_id: Optional group to nest this item in
            printer_border_style: Optional border color
            **span_kwargs: Additional kwargs for span (e.g., tools, input)

        Returns:
            Parsed output if output_model provided, otherwise Runner result
        """
        return await self.executor.agent_step(
            agent=agent,
            instructions=instructions,
            span_name=span_name,
            span_type=span_type,
            output_model=output_model,
            sync=sync,
            printer_key=printer_key,
            printer_title=printer_title,
            printer_group_id=printer_group_id,
            printer_border_style=printer_border_style,
            **span_kwargs
        )

    def update_printer(
        self,
        key: str,
        message: str,
        is_done: bool = False,
        hide_checkmark: bool = False,
        title: Optional[str] = None,
        border_style: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """Update printer status if printer is active.

        This method delegates to RuntimeTracker.

        Args:
            key: Status key to update
            message: Status message
            is_done: Whether the task is complete
            hide_checkmark: Whether to hide the checkmark when done
            title: Optional panel title
            border_style: Optional border color
            group_id: Optional group to nest this item in
        """
        self.runtime_tracker.update_printer(
            key,
            message,
            is_done=is_done,
            hide_checkmark=hide_checkmark,
            title=title,
            border_style=border_style,
            group_id=group_id
        )

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
            self._stop_printer()

    async def run_span_step(
        self,
        step_key: str,
        callable_or_coro: Union[Callable, Any],
        span_name: str,
        span_type: str = "function",
        start_message: Optional[str] = None,
        done_message: Optional[str] = None,
        **span_kwargs
    ) -> Any:
        """Execute a step with span context and printer updates.

        This method delegates to AgentExecutor from the flow module.

        Args:
            step_key: Printer status key
            callable_or_coro: Callable or coroutine to execute
            span_name: Name for the span
            span_type: Type of span - "agent" or "function"
            start_message: Optional start message for printer
            done_message: Optional completion message for printer
            **span_kwargs: Additional kwargs for span (e.g., tools, input)

        Returns:
            Result from callable_or_coro
        """
        return await self.executor.run_span_step(
            step_key=step_key,
            callable_or_coro=callable_or_coro,
            span_name=span_name,
            span_type=span_type,
            start_message=start_message,
            done_message=done_message,
            **span_kwargs
        )

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

    def _check_constraints(self) -> bool:
        """Check if we've exceeded our constraints (max iterations or time)."""
        if self.iteration >= self.max_iterations:
            self._log_message("\n=== Ending Research Loop ===")
            self._log_message(f"Reached maximum iterations ({self.max_iterations})")
            return False

        if self.start_time is not None:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.max_time_minutes:
                self._log_message("\n=== Ending Research Loop ===")
                self._log_message(f"Reached maximum time ({self.max_time_minutes} minutes)")
                return False

        return True

    def run_sync(self, *args, **kwargs):
        """Synchronous wrapper for the async run method."""
        return asyncio.run(self.run(*args, **kwargs))

    # ============================================
    # TIER 1: Template Method (Fixed Structure)
    # ============================================

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
            await self._trigger_hooks("before_execution")

            # Phase 3: Main execution (delegated to subclass)
            result = await self.execute()

            # Phase 4: Post-execution hooks
            await self._trigger_hooks("after_execution", result=result)
            await self.after_execution(result)

            # Phase 5: Finalization
            final_result = await self.finalize(result)

            return final_result

    # ============================================
    # TIER 1: Lifecycle Hook Methods
    # ============================================

    async def initialize_pipeline(self, query: Any) -> None:
        """Initialize pipeline state and format query.

        Default implementation:
        - Formats query via prepare_query_hook()
        - Sets state query
        - Updates printer status

        Override this for custom initialization logic.

        Args:
            query: Input query (can be None)
        """
        if query is not None:
            formatted_query = self.prepare_query_hook(query)
            if self.state:
                self.state.set_query(formatted_query)
        self.update_printer("initialization", "Pipeline initialized", is_done=True)

    def prepare_query_hook(self, query: Any) -> str:
        """Transform input query to formatted string.

        Override this to customize query formatting.

        Args:
            query: Input query

        Returns:
            Formatted query string
        """
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

    async def after_execution(self, result: Any) -> None:
        """Hook called after execute() completes.

        Use for:
        - Result validation
        - Cleanup operations
        - State aggregation

        Default implementation:
        - Auto-saves final report to global memory if configured

        Override this for custom post-execution logic.

        Args:
            result: The return value from execute()
        """
        # Auto-save to global memory if enabled
        if self.config.pipeline.get("save_to_memory", False):
            await self._save_report_to_memory()

    async def _save_report_to_memory(self) -> None:
        """Save final report to global memory (if available)."""
        try:
            from agentz.context.global_memory import global_memory

            if self.state:
                final_report = self.state.final_report
                if final_report:
                    timestamped_report = f"Experiment {self.experiment_id}\n\n{final_report.strip()}"
                    global_memory.store(
                        key=f"report_{self.experiment_id}",
                        value=timestamped_report,
                        tags=["research_report"],
                    )
                    self.update_printer("memory", "Report saved to memory", is_done=True)
        except ImportError:
            logger.debug("global_memory not available, skipping report save")
        except Exception as exc:
            logger.debug(f"Failed to save report to memory: {exc}")

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
    # TIER 2: Abstract Execute Method
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
    # TIER 3: Helper Utilities (Opt-in Composition)
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
            should_continue: Optional custom condition (default: _should_continue_iteration)

        Returns:
            Result from final_body if provided, else None

        Example:
            async def execute(self):
                async def my_iteration(iteration, group):
                    observations = await self.observe_agent(...)
                    evaluations = await self.evaluate_agent(...)
                    await self.route_and_execute(evaluations, group)

                async def my_final(group):
                    return await self.writer_agent(...)

                return await self.run_iterative_loop(
                    iteration_body=my_iteration,
                    final_body=my_final
                )
        """
        should_continue_fn = should_continue or self._should_continue_iteration

        while should_continue_fn():
            iteration, group_id = self._begin_iteration()

            await self._trigger_hooks("before_iteration", iteration=iteration, group_id=group_id)

            try:
                await iteration_body(iteration, group_id)
            finally:
                await self._trigger_hooks("after_iteration", iteration=iteration, group_id=group_id)
                self._end_iteration(group_id)

            if self.state and self.state.complete:
                break

        result = None
        if final_body:
            final_group = self._start_final_group()
            result = await final_body(final_group)
            self._end_final_group(final_group)

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
            async def execute(self):
                exploration = await self.run_custom_group(
                    "exploration",
                    "Exploration Phase",
                    self._explore
                )

                analysis = await self.run_custom_group(
                    "analysis",
                    "Deep Analysis",
                    lambda: self._analyze(exploration)
                )

                return analysis
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

        Example:
            async def execute(self):
                results = await self.run_parallel_steps({
                    "data_loading": self.load_data,
                    "validation": self.validate_inputs,
                    "model_init": self.initialize_models,
                })

                data = results["data_loading"]
                return await self.analyze(data)
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

        Example:
            async def execute(self):
                initial = await self.quick_check()

                return await self.run_if(
                    condition=initial.needs_deep_analysis,
                    body=lambda: self.deep_analysis(initial),
                    else_body=lambda: self.simple_report(initial)
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
            async def execute(self):
                results = await self.run_until(
                    condition=lambda: self.context.state.complete,
                    body=self._exploration_step,
                    max_iterations=10
                )
                return self.aggregate(results)
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

    def _begin_iteration(self) -> Tuple[Any, str]:
        """Begin a new iteration with printer group.

        Returns:
            Tuple of (iteration_record, group_id)
        """
        iteration = self.context.begin_iteration()
        group_id = f"{self.ITERATION_GROUP_PREFIX}-{iteration.index}"

        self.iteration = iteration.index
        self.start_group(
            group_id,
            title=f"Iteration {iteration.index}",
            border_style="white",
            iteration=iteration.index,
        )

        return iteration, group_id

    def _end_iteration(self, group_id: str) -> None:
        """End the current iteration and close printer group."""
        self.context.mark_iteration_complete()
        self.end_group(group_id, is_done=True)

    def _start_final_group(self) -> str:
        """Start final group for post-iteration work."""
        self.start_group(self.FINAL_GROUP_ID, title="Final Report", border_style="white")
        return self.FINAL_GROUP_ID

    def _end_final_group(self, group_id: str) -> None:
        """End final group."""
        self.end_group(group_id, is_done=True)

    def _should_continue_iteration(self) -> bool:
        """Check if iteration should continue.

        Checks:
        - State not complete
        - Within max iterations
        - Within max time
        """
        if self.state and self.state.complete:
            return False
        return self._check_constraints()

    # ============================================
    # Generic Utilities
    # ============================================

    def _record_structured_payload(self, value: object, context_label: Optional[str] = None) -> None:
        """Record a structured payload to the current iteration state.

        Args:
            value: The payload to record (typically a BaseModel instance)
            context_label: Optional label for debugging purposes
        """
        if isinstance(value, BaseModel):
            try:
                if self.state:
                    self.state.record_payload(value)
            except Exception as exc:
                if context_label:
                    logger.debug(f"Failed to record payload for {context_label}: {exc}")
                else:
                    logger.debug(f"Failed to record payload: {exc}")

    def _serialize_output(self, output: Any) -> str:
        """Serialize agent output to string for storage.

        Args:
            output: The output to serialize (BaseModel, str, or other)

        Returns:
            String representation of the output
        """
        if isinstance(output, BaseModel):
            return output.model_dump_json(indent=2)
        elif isinstance(output, str):
            return output
        return str(output)

    async def execute_tool_plan(
        self,
        plan: Any,
        tool_agents: Dict[str, Any],
        group_id: str,
    ) -> None:
        """Execute a routing plan with tool agents.

        Args:
            plan: AgentSelectionPlan with tasks to execute
            tool_agents: Dict mapping agent names to agent instances
            group_id: Group ID for printer updates
        """
        # Import here to avoid circular dependency
        from agentz.profiles.manager.routing import AgentSelectionPlan, AgentTask
        from agentz.profiles.base import ToolAgentOutput

        if not isinstance(plan, AgentSelectionPlan) or not plan.tasks:
            return

        state = self.context.state
        state.current_iteration.tools.clear()

        async def run_single(task: AgentTask) -> ToolAgentOutput:
            agent = tool_agents.get(task.agent)
            if agent is None:
                output = ToolAgentOutput(
                    output=f"No implementation found for agent {task.agent}",
                    sources=[],
                )
                self.update_printer(
                    key=f"{group_id}:tool:{task.agent}",
                    message=f"Completed {task.agent}",
                    is_done=True,
                    group_id=group_id,
                )
                return output

            raw_result = await self.agent_step(
                agent=agent,
                instructions=task.model_dump_json(),
                span_name=task.agent,
                span_type="tool",
                output_model=ToolAgentOutput,
                printer_key=f"tool:{task.agent}",
                printer_title=f"Tool: {task.agent}",
                printer_group_id=group_id,
            )

            if isinstance(raw_result, ToolAgentOutput):
                output = raw_result
            elif hasattr(raw_result, "final_output_as"):
                output = raw_result.final_output_as(ToolAgentOutput)
            elif hasattr(raw_result, "final_output"):
                output = ToolAgentOutput(output=str(raw_result.final_output), sources=[])
            else:
                output = ToolAgentOutput(output=str(raw_result), sources=[])

            try:
                state.record_payload(output)
            except Exception as exc:
                logger.debug(f"Failed to record tool payload for {task.agent}: {exc}")

            self.update_printer(
                key=f"{group_id}:tool:{task.agent}",
                message=f"Completed {task.agent}",
                is_done=True,
                group_id=group_id,
            )
            return output

        coroutines = [run_single(task) for task in plan.tasks]
        for coro in asyncio.as_completed(coroutines):
            tool_output = await coro
            state.current_iteration.tools.append(tool_output)

    async def _execute_tools(
        self,
        route_plan: Any,
        tool_agents: Dict[str, Any],
        group_id: str,
    ) -> None:
        """Execute tool agents based on routing plan.

        Args:
            route_plan: The routing plan (can be AgentSelectionPlan or other)
            tool_agents: Dict mapping agent names to agent instances
            group_id: Group ID for printer updates
        """
        from agentz.profiles.manager.routing import AgentSelectionPlan

        # Retrieve route_plan from payloads if needed
        plan = None
        if isinstance(route_plan, AgentSelectionPlan):
            plan = route_plan
        elif route_plan and hasattr(self, 'context'):
            # Try to find AgentSelectionPlan in payloads
            for payload in self.context.state.current_iteration.payloads:
                if isinstance(payload, AgentSelectionPlan):
                    plan = payload
                    break

        if plan and plan.tasks:
            await self.execute_tool_plan(plan, tool_agents, group_id)

    # ============================================
    # Pattern Templates (High-Level Workflows)
    # ============================================

    async def run_manager_tool_loop(
        self,
        manager_agents: Dict[str, Any],
        tool_agents: Dict[str, Any],
        workflow: List[str],
    ) -> Any:
        """Execute standard manager-tool iterative pattern.

        This pattern implements: observe → evaluate → route → execute tools → repeat.

        Args:
            manager_agents: Dict of manager agents (observe, evaluate, routing, writer)
            tool_agents: Dict of tool agents
            workflow: List of manager agent names to execute in order (e.g., ["observe", "evaluate", "routing"])

        Returns:
            Result from final step

        Example:
            async def execute(self):
                return await self.run_manager_tool_loop(
                    manager_agents=self.manager_agents,
                    tool_agents=self.tool_agents,
                    workflow=["observe", "evaluate", "routing"]
                )
        """
        async def iteration_step(iteration, group_id: str):
            """Execute manager workflow + tool execution."""
            previous_output = self.context.state.query

            # Execute manager workflow in sequence
            for agent_name in workflow:
                agent = manager_agents.get(agent_name)
                if agent is None:
                    logger.warning(f"Manager agent '{agent_name}' not found, skipping")
                    continue

                output = await agent(previous_output)

                # Record observation for first step
                if agent_name == workflow[0]:
                    iteration.observation = self._serialize_output(output)

                self._record_structured_payload(output, context_label=agent_name)
                previous_output = output

            # Execute tools if not complete
            if not self.context.state.complete and previous_output:
                await self._execute_tools(previous_output, tool_agents, group_id)

        async def final_step(final_group: str):
            """Generate final report."""
            self.update_printer("research", "Research workflow complete", is_done=True)
            logger.info("Research workflow completed")

            writer = manager_agents.get("writer")
            if writer:
                await writer(self.context.state.findings_text())

        self.update_printer("research", "Executing research workflow...")

        return await self.run_iterative_loop(
            iteration_body=iteration_step,
            final_body=final_step
        )

    # ============================================
    # Hook Registry System (Event-Driven)
    # ============================================

    def register_hook(
        self,
        event: str,
        callback: Callable,
        priority: int = 0
    ) -> None:
        """Register a hook callback for an event.

        Args:
            event: Event name (before_execution, after_execution, before_iteration, after_iteration, etc.)
            callback: Callable or async callable
            priority: Execution priority (higher = earlier)

        Example:
            def log_iteration(pipeline, iteration, group_id):
                logger.info(f"Starting iteration {iteration.index}")

            pipeline.register_hook("before_iteration", log_iteration)
        """
        if event not in self._hooks:
            raise ValueError(f"Unknown hook event: {event}. Valid events: {list(self._hooks.keys())}")

        self._hooks[event].append((priority, callback))
        # Sort by priority (descending)
        self._hooks[event].sort(key=lambda x: -x[0])

    async def _trigger_hooks(self, event: str, **kwargs) -> None:
        """Trigger all registered hooks for an event.

        Args:
            event: Event name
            **kwargs: Arguments to pass to hook callbacks
        """
        for priority, callback in self._hooks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self, **kwargs)
                else:
                    callback(self, **kwargs)
            except Exception as e:
                logger.warning(f"Hook {callback.__name__} for {event} failed: {e}")

