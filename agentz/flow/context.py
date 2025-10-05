"""Execution context management for agent runtime operations."""

from contextlib import nullcontext
from typing import Any, Dict, Optional

from agents.tracing.create import trace
from agentz.utils import Printer


class ExecutionContext:
    """Manages execution context including tracing, printing, and span management.

    This class encapsulates the runtime state needed for agent execution including:
    - Tracing configuration and context creation
    - Printer for status updates
    - Iteration tracking
    """

    def __init__(
        self,
        printer: Optional[Printer] = None,
        enable_tracing: bool = True,
        trace_sensitive: bool = False,
        iteration: int = 0
    ):
        """Initialize execution context.

        Args:
            printer: Optional Printer instance for status updates
            enable_tracing: Whether tracing is enabled
            trace_sensitive: Whether to include sensitive data in traces
            iteration: Current iteration number (for iterative workflows)
        """
        self.printer = printer
        self.enable_tracing = enable_tracing
        self.trace_sensitive = trace_sensitive
        self.iteration = iteration

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
