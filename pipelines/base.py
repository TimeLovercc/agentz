import asyncio
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from loguru import logger
from rich.console import Console

from agents.tracing.create import function_span
from agentz.agents.registry import AgentStore, set_current_pipeline_store
from agentz.configuration.base import BaseConfig, resolve_config
from agentz.flow import (
    AgentExecutor,
    ExecutionContext,
)
from agentz.utils import Printer, get_experiment_timestamp
from pydantic import BaseModel


class BasePipeline:
    """Base class for all pipelines with common configuration and setup."""

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

        # Resolve configuration using the new unified API
        self.config = resolve_config(config)

        # Initialize agent store with BaseConfig (not LLM config)
        # Factories now receive BaseConfig and extract specs from agents_index
        self.agents = AgentStore(self.config)

        # Set as current pipeline store for auto-registration
        set_current_pipeline_store(self.agents)

        # Generic pipeline settings
        self.experiment_id = get_experiment_timestamp()

        pipeline_settings = self.config.pipeline
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

        enable_tracing = self.config.pipeline.get("enable_tracing", True)
        trace_sensitive = self.config.pipeline.get("trace_include_sensitive_data", False)
        pipeline_name = self.__class__.__name__
        logger.info(
            f"Initialized {pipeline_name} with experiment_id: {self.experiment_id}, "
            f"tracing: {enable_tracing}, sensitive_data: {trace_sensitive}"
        )

        # Initialize execution context and executor (from flow module)
        self._execution_context: Optional[ExecutionContext] = None
        self._executor: Optional[AgentExecutor] = None

    @property
    def provider_name(self) -> str:
        return self.config.provider

    @property
    def printer(self) -> Optional[Printer]:
        return self._printer

    @property
    def execution_context(self) -> ExecutionContext:
        """Get or create the execution context."""
        if self._execution_context is None:
            enable_tracing = self.config.pipeline.get("enable_tracing", True)
            trace_sensitive = self.config.pipeline.get("trace_include_sensitive_data", False)
            self._execution_context = ExecutionContext(
                printer=self.printer,
                enable_tracing=enable_tracing,
                trace_sensitive=trace_sensitive,
                iteration=self.iteration,
                experiment_id=self.experiment_id
            )
        else:
            # Update iteration in existing context
            self._execution_context.iteration = self.iteration
            self._execution_context.printer = self.printer
        return self._execution_context

    @property
    def executor(self) -> AgentExecutor:
        """Get or create the agent executor."""
        # Refresh execution context so iteration/printer stay in sync across loops
        context = self.execution_context

        if self._executor is None:
            self._executor = AgentExecutor(context)
        else:
            # Executor holds a reference to the context; update it in case it changed
            self._executor.context = context
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

        # Provider and model logging
        provider = self.provider_name
        logger.info(f"Provider: {provider}, Model: {self.config.llm.model_name}")

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
        trace_sensitive = self.config.pipeline.get("trace_include_sensitive_data", False)
        trace_metadata = {
            "experiment_id": self.experiment_id,
            "includes_sensitive_data": "true" if trace_sensitive else "false",
        }
        return self.trace_context(self.workflow_name, metadata=trace_metadata)

    def _setup_tracing(self) -> None:
        """Setup tracing configuration with user-friendly output.

        Subclasses can override this method to add pipeline-specific information.
        """
        enable_tracing = self.config.pipeline.get("enable_tracing", True)
        trace_sensitive = self.config.pipeline.get("trace_include_sensitive_data", False)

        if enable_tracing:
            pipeline_name = self.__class__.__name__.replace("Pipeline", "")
            self.console.print(f"🍌 Starting {pipeline_name} Pipeline with Tracing")
            self.console.print(f"🔧 Provider: {self.provider_name}")
            self.console.print(f"🤖 Model: {self.config.llm.model_name}")
            self.console.print("🔍 Tracing: Enabled")
            self.console.print(
                f"🔒 Sensitive Data in Traces: {'Yes' if trace_sensitive else 'No'}"
            )
            self.console.print(f"🏷️ Workflow: {self.workflow_name}")
        else:
            pipeline_name = self.__class__.__name__.replace("Pipeline", "")
            self.console.print(f"🍌 Starting {pipeline_name} Pipeline")
            self.console.print(f"🔧 Provider: {self.provider_name}")
            self.console.print(f"🤖 Model: {self.config.llm.model_name}")

    def trace_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a trace context - delegates to ExecutionContext."""
        return self.execution_context.trace_context(name, metadata=metadata)

    def span_context(self, span_factory, **kwargs):
        """Create a span context - delegates to ExecutionContext."""
        return self.execution_context.span_context(span_factory, **kwargs)

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

        This method delegates to ExecutionContext.

        Args:
            key: Status key to update
            message: Status message
            is_done: Whether the task is complete
            hide_checkmark: Whether to hide the checkmark when done
            title: Optional panel title
            border_style: Optional border color
            group_id: Optional group to nest this item in
        """
        self.execution_context.update_printer(
            key,
            message,
            is_done=is_done,
            hide_checkmark=hide_checkmark,
            title=title,
            border_style=border_style,
            group_id=group_id
        )

    @contextmanager
    def run_context(self, additional_logging: Optional[Callable] = None):
        """Context manager for run lifecycle handling.

        Manages trace context initialization, printer lifecycle, and cleanup.

        Args:
            additional_logging: Optional callable for pipeline-specific logging

        Yields:
            Trace context for the workflow
        """
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

    def run_sync(self):
        """Synchronous wrapper for the async run method."""
        return asyncio.run(self.run())

    async def run(self):  # pragma: no cover - must be implemented by subclasses
        raise NotImplementedError("Subclasses must implement 'run'")
