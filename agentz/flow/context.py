"""Execution context management for agent runtime operations."""

import asyncio
from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Dict, Optional

from agents.tracing.create import agent_span, function_span, trace
from agentz.utils import Printer
from agentz.memory.pipeline_context import PipelineDataStore

# Context variable to store the current execution context
# This allows tools to access the context without explicit parameter passing
_current_execution_context: ContextVar[Optional['ExecutionContext']] = ContextVar(
    'current_execution_context',
    default=None
)


class ExecutionContext:
    """Manages execution context including tracing, printing, and span management.

    This class encapsulates the runtime state needed for agent execution including:
    - Tracing configuration and context creation
    - Printer for status updates
    - Iteration tracking
    - Pipeline-scoped data store for sharing objects between agents
    """

    def __init__(
        self,
        printer: Optional[Printer] = None,
        enable_tracing: bool = True,
        trace_sensitive: bool = False,
        iteration: int = 0,
        experiment_id: Optional[str] = None
    ):
        """Initialize execution context.

        Args:
            printer: Optional Printer instance for status updates
            enable_tracing: Whether tracing is enabled
            trace_sensitive: Whether to include sensitive data in traces
            iteration: Current iteration number (for iterative workflows)
            experiment_id: Optional experiment ID for data store tracking
        """
        self.printer = printer
        self.enable_tracing = enable_tracing
        self.trace_sensitive = trace_sensitive
        self.iteration = iteration
        self.data_store = PipelineDataStore(experiment_id=experiment_id)

    def trace_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a trace context manager.

        Args:
            name: Name for the trace
            metadata: Optional metadata to attach to trace

        Returns:
            Trace context manager if tracing enabled, otherwise nullcontext
        """
        if self.enable_tracing:
            return trace(name, metadata=metadata)
        return nullcontext()

    def span_context(self, span_factory, **kwargs):
        """Create a span context manager.

        Args:
            span_factory: Factory function for creating spans (agent_span or function_span)
            **kwargs: Arguments to pass to span factory

        Returns:
            Span context manager if tracing enabled, otherwise nullcontext
        """
        if self.enable_tracing:
            return span_factory(**kwargs)
        return nullcontext()

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

        Args:
            key: Status key to update
            message: Status message
            is_done: Whether the task is complete
            hide_checkmark: Whether to hide the checkmark when done
            title: Optional panel title
            border_style: Optional border color
            group_id: Optional group to nest this item in
        """
        if self.printer:
            self.printer.update_item(
                key,
                message,
                is_done=is_done,
                hide_checkmark=hide_checkmark,
                title=title,
                border_style=border_style,
                group_id=group_id
            )

    def log_panel(
        self,
        title: str,
        content: str,
        *,
        border_style: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """Proxy helper for rendering standalone panels via the printer."""
        if self.printer:
            self.printer.log_panel(
                title,
                content,
                border_style=border_style,
                iteration=iteration,
            )

    @contextmanager
    def activate(self):
        """Context manager to set this context as the current execution context.

        This allows tools to access the context via get_current_context().

        Example:
            with context.activate():
                # Tools can now access this context
                result = await agent.run(...)
        """
        token = _current_execution_context.set(self)
        try:
            yield self
        finally:
            _current_execution_context.reset(token)


def get_current_context() -> Optional[ExecutionContext]:
    """Get the current execution context (if any).

    Returns:
        The current ExecutionContext or None if not in an execution context
    """
    return _current_execution_context.get()


def get_current_data_store() -> Optional[PipelineDataStore]:
    """Get the data store from the current execution context (if any).

    This is a convenience function for tools that need to access the data store.

    Returns:
        The current PipelineDataStore or None if not in an execution context
    """
    context = get_current_context()
    return context.data_store if context else None


def auto_trace(additional_logging: Optional[Callable] = None):
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

    # Support both @auto_trace and @auto_trace()
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
