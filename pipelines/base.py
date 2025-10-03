import asyncio
import time
from contextlib import contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from loguru import logger
from rich.console import Console

from agents import Runner
from agents.tracing.create import agent_span, function_span, trace
from agentz.configuration.base import BaseConfig, load_pipeline_config
from agentz.utils import Printer, get_experiment_timestamp
from pydantic import BaseModel, Field


def with_run_context(additional_logging: Optional[Callable] = None):
    """Decorator that wraps async run method with automatic context management.

    Handles:
    - Initialization via _initialize_run()
    - Trace context lifecycle
    - Printer cleanup via _stop_printer()
    - Auto-finalization if method returns a research report

    Args:
        additional_logging: Optional callable for pipeline-specific logging
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            trace_ctx = self._initialize_run(additional_logging)
            try:
                with trace_ctx:
                    result = await func(self, *args, **kwargs)
                    # Auto-finalize if result looks like a research report
                    if result and isinstance(result, str) and hasattr(self, '_finalise_research'):
                        await self._finalise_research(result)
                    return result
            finally:
                self._stop_printer()
        return wrapper

    # Support both @with_run_context and @with_run_context()
    if callable(additional_logging):
        func = additional_logging
        additional_logging = None
        return decorator(func)
    return decorator


def with_span_step(
    step_key: str,
    span_name: str,
    span_type: str = "function",
    start_message: Optional[str] = None,
    done_message: Optional[str] = None,
    **span_kwargs
):
    """Decorator that wraps a function/coroutine with span context and printer updates.

    Args:
        step_key: Printer status key
        span_name: Name for the span
        span_type: Type of span - "agent" or "function"
        start_message: Optional start message for printer
        done_message: Optional completion message for printer
        **span_kwargs: Additional kwargs for span (e.g., tools, input)
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            span_factory = agent_span if span_type == "agent" else function_span

            if start_message:
                self.update_printer(step_key, start_message)

            with self.span_context(span_factory, name=span_name, **span_kwargs) as span:
                result = await func(self, *args, **kwargs)

                # Set span output if available
                if span and hasattr(span, "set_output"):
                    if isinstance(result, dict):
                        span.set_output(result)
                    elif isinstance(result, str):
                        span.set_output({"output_preview": result[:200]})
                    elif hasattr(result, "model_dump"):
                        span.set_output(result.model_dump())
                    else:
                        span.set_output({"result": str(result)[:200]})

                if done_message:
                    self.update_printer(step_key, done_message, is_done=True)

                return result

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            span_factory = agent_span if span_type == "agent" else function_span

            if start_message:
                self.update_printer(step_key, start_message)

            with self.span_context(span_factory, name=span_name, **span_kwargs) as span:
                result = func(self, *args, **kwargs)

                # Set span output if available
                if span and hasattr(span, "set_output"):
                    if isinstance(result, dict):
                        span.set_output(result)
                    elif isinstance(result, str):
                        span.set_output({"output_preview": result[:200]})
                    elif hasattr(result, "model_dump"):
                        span.set_output(result.model_dump())
                    else:
                        span.set_output({"result": str(result)[:200]})

                if done_message:
                    self.update_printer(step_key, done_message, is_done=True)

                return result

        # Return appropriate wrapper based on whether func is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class BasePipeline:
    """Base class for all pipelines with common configuration and setup."""

    def __init__(self, config: Union[BaseConfig, Mapping[str, Any], str, Path], manager_factories=None, initial_tool_agents=None):
        """Initialise the pipeline using a single configuration input."""
        self.console = Console()
        self._printer: Optional[Printer] = None

        # Load and process configuration
        self.config = load_pipeline_config(config)

        # Build agents from provided factories
        if manager_factories:
            self.manager_agents = self._build_manager_agents(manager_factories)
        else:
            self.manager_agents = {}

        if initial_tool_agents:
            self.tool_agents = self._build_tool_agents(initial_tool_agents)
        else:
            self.tool_agents = {}

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
        self.conversation = Conversation()
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

    @property
    def provider_name(self) -> str:
        return self.config.provider

    @property
    def printer(self) -> Optional[Printer]:
        return self._printer

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
        enable_tracing = self.config.pipeline.get("enable_tracing", True)
        if enable_tracing:
            return trace(name, metadata=metadata)
        return nullcontext()

    def span_context(self, span_factory, **kwargs):
        enable_tracing = self.config.pipeline.get("enable_tracing", True)
        if enable_tracing:
            return span_factory(**kwargs)
        return nullcontext()

    async def run_agent_with_span(
        self,
        agent,
        instructions: str,
        span_name: str,
        span_type: str = "agent",
        output_model: Optional[type[BaseModel]] = None,
        sync: bool = False,
        **span_kwargs
    ) -> Any:
        """Run an agent with span tracking and optional output parsing.

        Args:
            agent: The agent to run
            instructions: Instructions/prompt for the agent
            span_name: Name for the span
            span_type: Type of span - "agent" or "function"
            output_model: Optional pydantic model to parse output
            sync: Whether to run synchronously
            **span_kwargs: Additional kwargs for span (e.g., tools, input)

        Returns:
            Parsed output if output_model provided, otherwise Runner result
        """
        span_factory = agent_span if span_type == "agent" else function_span

        with self.span_context(span_factory, name=span_name, **span_kwargs) as span:
            if sync:
                result = Runner.run_sync(agent, instructions)
            else:
                result = await Runner.run(agent, instructions)

            if output_model:
                output = output_model.model_validate_json(result.final_output)
                if span and hasattr(span, "set_output"):
                    span.set_output(output.model_dump())
                return output
            else:
                if span and hasattr(span, "set_output"):
                    span.set_output({"output_preview": str(result.final_output)[:200]})
                return result

    def update_printer(
        self,
        key: str,
        message: str,
        is_done: bool = False,
        hide_checkmark: bool = False
    ) -> None:
        """Update printer status if printer is active.

        Args:
            key: Status key to update
            message: Status message
            is_done: Whether the task is complete
            hide_checkmark: Whether to hide the checkmark when done
        """
        if self.printer:
            self.printer.update_item(key, message, is_done=is_done, hide_checkmark=hide_checkmark)

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
        span_factory = agent_span if span_type == "agent" else function_span

        if start_message:
            self.update_printer(step_key, start_message)

        with self.span_context(span_factory, name=span_name, **span_kwargs) as span:
            # Execute callable or await coroutine
            if asyncio.iscoroutine(callable_or_coro):
                result = await callable_or_coro
            elif callable(callable_or_coro):
                result = callable_or_coro()
            else:
                result = callable_or_coro

            # Set span output if available
            if span and hasattr(span, "set_output"):
                if isinstance(result, dict):
                    span.set_output(result)
                elif isinstance(result, str):
                    span.set_output({"output_preview": result[:200]})
                elif hasattr(result, "model_dump"):
                    span.set_output(result.model_dump())
                else:
                    span.set_output({"result": str(result)[:200]})

            if done_message:
                self.update_printer(step_key, done_message, is_done=True)

            return result

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

    def _instantiate_agent_spec(self, spec, config):
        """Instantiate a manager agent from the provided specification."""
        if callable(spec):
            return spec(config)
        if hasattr(spec, "clone"):
            cloned = spec.clone()
            if getattr(cloned, "model", None) is None:
                cloned.model = config.main_model
            return cloned
        return spec

    def _instantiate_tool_agent_spec(self, spec, config):
        """Instantiate a tool agent from the provided specification."""
        if callable(spec):
            return spec(config)
        return spec

    def _build_manager_agents(self, default_factories: Dict[str, Any]) -> Dict[str, Any]:
        """Build manager agents from config and default factories.

        Args:
            default_factories: Dict mapping agent names to factory functions

        Returns:
            Dict of instantiated manager agents
        """
        manager_overrides = self.config.manager_agents
        if manager_overrides and not isinstance(manager_overrides, dict):
            raise TypeError("manager_agents section in config must be a mapping")

        manager_agents = {}
        for name, factory in default_factories.items():
            if manager_overrides and name in manager_overrides:
                manager_agents[name] = self._instantiate_agent_spec(
                    manager_overrides[name], self.config.llm
                )
            else:
                manager_agents[name] = factory(self.config.llm)

        return manager_agents

    def _build_tool_agents(self, initial_tool_agents: Dict[str, Any]) -> Dict[str, Any]:
        """Build tool agents from config and initial agents.

        Args:
            initial_tool_agents: Initial dict of tool agents (e.g., from init_tool_agents)

        Returns:
            Dict of tool agents with config overrides applied
        """
        tool_overrides = self.config.tool_agents
        if tool_overrides and not isinstance(tool_overrides, dict):
            raise TypeError("tool_agents section in config must be a mapping")

        tool_agents = initial_tool_agents.copy()

        if isinstance(tool_overrides, dict):
            for tool_name, spec in tool_overrides.items():
                tool_agents[tool_name] = self._instantiate_tool_agent_spec(
                    spec, self.config.llm
                )

        return tool_agents


class IterationData(BaseModel):
    """Structured information collected during a single research loop iteration."""

    gap: str = Field(description="The gap addressed in the iteration", default="")
    tool_calls: List[str] = Field(description="The tool calls made", default_factory=list)
    findings: List[str] = Field(
        description="The findings collected from tool calls", default_factory=list
    )
    thought: str = Field(
        description="Reflection on iteration success and next steps", default=""
    )


class Conversation(BaseModel):
    """Accumulates thoughts, actions, and findings across research iterations."""

    history: List[IterationData] = Field(
        description="The data for each iteration of the research loop", default_factory=list
    )

    def add_iteration(self, iteration_data: Optional[IterationData] = None) -> None:
        if iteration_data is None:
            iteration_data = IterationData()
        self.history.append(iteration_data)

    def set_latest_gap(self, gap: str) -> None:
        if self.history:
            self.history[-1].gap = gap

    def set_latest_tool_calls(self, tool_calls: List[str]) -> None:
        if self.history:
            self.history[-1].tool_calls = tool_calls

    def set_latest_findings(self, findings: List[str]) -> None:
        if self.history:
            self.history[-1].findings = findings

    def set_latest_thought(self, thought: str) -> None:
        if self.history:
            self.history[-1].thought = thought

    def get_all_findings(self) -> List[str]:
        return [
            finding
            for iteration_data in self.history
            for finding in iteration_data.findings
        ]

    def compile_conversation_history(self) -> str:
        """Compile the conversation history into a string."""

        conversation = ""
        for iteration_num, iteration_data in enumerate(self.history):
            conversation += f"[ITERATION {iteration_num + 1}]\n\n"
            if iteration_data.thought:
                conversation += f"<thought>\n{iteration_data.thought}\n</thought>\n\n"
            if iteration_data.gap:
                conversation += (
                    f"<task>\nAddress this knowledge gap: {iteration_data.gap}\n</task>\n\n"
                )
            if iteration_data.tool_calls:
                joined_calls = "\n".join(iteration_data.tool_calls)
                conversation += (
                    "<action>\nCalling the following tools to address the knowledge gap:\n"
                    f"{joined_calls}\n</action>\n\n"
                )
            if iteration_data.findings:
                joined_findings = "\n\n".join(iteration_data.findings)
                conversation += f"<findings>\n{joined_findings}\n</findings>\n\n"

        return conversation
