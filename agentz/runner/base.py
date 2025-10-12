"""Base runner classes for agent execution.

This module provides the foundational runner classes:
- Runner: Base runner from agents library (re-exported for convenience)
- ContextRunner: Context-aware runner that applies output parsing
"""

from typing import Any

from agents import Runner, RunResult

__all__ = ["Runner", "ContextRunner"]


class ContextRunner(Runner):
    """Runner shim that invokes ContextAgent.parse_output after execution."""

    @classmethod
    async def run(cls, *args: Any, **kwargs: Any) -> RunResult:
        result = await Runner.run(*args, **kwargs)
        starting_agent = kwargs.get("starting_agent") or (args[0] if args else None)

        # Import here to avoid circular dependency
        from agentz.agent.base import ContextAgent

        if isinstance(starting_agent, ContextAgent):
            return await starting_agent.parse_output(result)
        return result
