"""Core agent execution primitives."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from agentz.agent.agent_base import ResearchRunner as Runner
from agents.tracing.create import agent_span, function_span
from pydantic import BaseModel

from agentz.flow.context import ExecutionContext


@dataclass
class PrinterConfig:
    """Configuration for printer updates during step execution."""
    key: Optional[str] = None
    title: Optional[str] = None
    start_message: Optional[str] = None
    done_message: Optional[str] = None


@dataclass
class AgentStep:
    """Represents a single agent execution step.

    This encapsulates all the information needed to execute an agent:
    - The agent instance
    - Instructions (static or dynamic via callable)
    - Span configuration for tracing
    - Output model for parsing
    - Printer configuration for status updates
    """

    agent: Any
    instructions: Union[str, Callable[[], str]]
    span_name: str
    span_type: str = "agent"
    output_model: Optional[type[BaseModel]] = None
    sync: bool = False
    printer_config: Optional[PrinterConfig] = None
    span_kwargs: Dict[str, Any] = field(default_factory=dict)

    def get_instructions(self) -> str:
        """Get instructions, evaluating callable if needed.

        Returns:
            Instructions string
        """
        if callable(self.instructions):
            return self.instructions()
        return self.instructions

    def get_printer_key(self, iteration: int = 0) -> Optional[str]:
        """Get the printer key, adding iteration prefix if configured.

        Args:
            iteration: Current iteration number

        Returns:
            Printer key with iteration prefix, or None if not configured
        """
        if not self.printer_config or not self.printer_config.key:
            return None
        return f"iter:{iteration}:{self.printer_config.key}"


class AgentExecutor:
    """Core agent execution primitive.

    Handles the low-level mechanics of running agents with:
    - Span tracking and output capture
    - Optional output model parsing
    - Printer status updates
    - Sync/async execution
    """

    def __init__(self, context: ExecutionContext):
        """Initialize executor with execution context.

        Args:
            context: ExecutionContext for tracing, printing, etc.
        """
        self.context = context

    async def execute_step(self, step: AgentStep) -> Any:
        """Execute an AgentStep.

        Args:
            step: AgentStep to execute

        Returns:
            Parsed output if output_model provided, otherwise Runner result
        """
        instructions = step.get_instructions()

        return await self.agent_step(
            agent=step.agent,
            instructions=instructions,
            span_name=step.span_name,
            span_type=step.span_type,
            output_model=step.output_model,
            sync=step.sync,
            printer_key=step.printer_config.key if step.printer_config else None,
            printer_title=step.printer_config.title if step.printer_config else None,
            **step.span_kwargs
        )

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
        span_factory = agent_span if span_type == "agent" else function_span

        reporter = self.context.reporter
        agent_name = getattr(agent, "name", getattr(agent, "__class__", type("obj", (), {})).__name__)
        step_id: Optional[str] = None
        if reporter:
            step_id = f"{self.context.iteration}-{span_name}-{time.time_ns()}"
            reporter.record_agent_step_start(
                step_id=step_id,
                agent_name=str(agent_name),
                span_name=span_name,
                iteration=self.context.iteration,
                group_id=printer_group_id,
                printer_title=printer_title,
            )

        full_printer_key: Optional[str] = None
        if printer_key:
            full_printer_key = f"iter:{self.context.iteration}:{printer_key}"
            self.context.update_printer(
                full_printer_key,
                "Working...",
                title=printer_title or printer_key,
                border_style=printer_border_style,
                group_id=printer_group_id,
            )

        status = "success"
        error_message: Optional[str] = None
        start_time = time.perf_counter()

        try:
            with self.context.span_context(span_factory, name=span_name, **span_kwargs) as span:
                # Activate context so tools can access it
                with self.context.activate():
                    if sync:
                        result = Runner.run_sync(agent, instructions, context=self.context.data_store)
                    else:
                        result = await Runner.run(agent, instructions, context=self.context.data_store)

                raw_output = getattr(result, "final_output", result)

                # Update printer status and emit detailed output as a standalone panel
                if full_printer_key:
                    self.context.update_printer(
                        full_printer_key,
                        "Completed",
                        is_done=True,
                        title=printer_title or printer_key,
                        group_id=printer_group_id,
                        border_style=printer_border_style,
                    )

                    # Extract content from raw_output
                    if hasattr(raw_output, 'output'):
                        panel_content = str(raw_output.output)
                    elif isinstance(raw_output, BaseModel):
                        panel_content = raw_output.model_dump_json(indent=2)
                    elif isinstance(raw_output, dict):
                        panel_content = json.dumps(raw_output, indent=2)
                    else:
                        panel_content = str(raw_output)

                    if panel_content.strip():
                        self.context.log_panel(
                            printer_title or printer_key,
                            panel_content,
                            border_style=printer_border_style,
                            iteration=self.context.iteration,
                            group_id=printer_group_id,
                        )

                if output_model:
                    if isinstance(raw_output, output_model):
                        output = raw_output
                    elif isinstance(raw_output, BaseModel):
                        output = output_model.model_validate(raw_output.model_dump())
                    elif isinstance(raw_output, (dict, list)):
                        output = output_model.model_validate(raw_output)
                    elif isinstance(raw_output, (str, bytes, bytearray)):
                        output = output_model.model_validate_json(raw_output)
                    else:
                        output = output_model.model_validate(raw_output)
                    if span and hasattr(span, "set_output"):
                        span.set_output(output.model_dump())
                    return output
                else:
                    if span and hasattr(span, "set_output"):
                        span.set_output({"output_preview": str(getattr(result, "final_output", result))[:200]})
                    return result
        except Exception as exc:  # noqa: BLE001 - propagate after logging
            status = "error"
            error_message = str(exc)
            raise
        finally:
            if reporter and step_id is not None:
                reporter.record_agent_step_end(
                    step_id=step_id,
                    status=status,
                    duration_seconds=time.perf_counter() - start_time,
                    error=error_message,
                )

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
            self.context.update_printer(step_key, start_message)

        with self.context.span_context(span_factory, name=span_name, **span_kwargs) as span:
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
                self.context.update_printer(step_key, done_message, is_done=True)

            return result
