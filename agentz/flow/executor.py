"""Core agent execution primitives."""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from agents import Runner
from agents.tracing.create import agent_span, function_span
from loguru import logger
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

        with self.context.span_context(span_factory, name=span_name, **span_kwargs) as span:
            if sync:
                result = Runner.run_sync(agent, instructions)
            else:
                result = await Runner.run(agent, instructions)

            raw_output = getattr(result, "final_output", result)

            # Print output preview if printer_key is provided
            if printer_key and self.context.printer:
                full_key = f"iter:{self.context.iteration}:{printer_key}"
                preview = str(raw_output)
                if len(preview) > 600:
                    preview = preview[:600] + "..."

                self.context.printer.update_item(
                    full_key,
                    preview,
                    is_done=True,
                    title=printer_title or printer_key,
                    group_id=printer_group_id,
                    border_style=printer_border_style
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
                    span.set_output({"output_preview": str(result.final_output)[:200]})
                return result

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
