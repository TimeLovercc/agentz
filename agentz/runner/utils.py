"""Utility functions for pipeline execution."""

from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel


def record_structured_payload(
    state: Any,
    value: object,
    context_label: Optional[str] = None
) -> None:
    """Record a structured payload to the current iteration state.

    Args:
        state: The state object (typically from context.state)
        value: The payload to record (typically a BaseModel instance)
        context_label: Optional label for debugging purposes
    """
    if isinstance(value, BaseModel):
        try:
            if state:
                state.record_payload(value)
        except Exception as exc:
            if context_label:
                logger.debug(f"Failed to record payload for {context_label}: {exc}")
            else:
                logger.debug(f"Failed to record payload: {exc}")


def serialize_output(output: Any) -> str:
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
