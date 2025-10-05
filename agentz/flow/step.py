"""Agent step abstraction for runtime execution."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from pydantic import BaseModel


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
