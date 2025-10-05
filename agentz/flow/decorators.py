"""Execution decorators for agent runtime operations."""

import asyncio
from functools import wraps
from typing import Callable, Optional

from agents.tracing.create import agent_span, function_span


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
