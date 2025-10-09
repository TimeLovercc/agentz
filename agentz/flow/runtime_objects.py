from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel

from agentz.memory.behavior_profiles import runtime_prompts
from agentz.memory.conversation import ConversationState


@dataclass
class AgentCapability:
    """Wrapper around an instantiated agent with IO contract metadata."""

    name: str
    agent: Any

    async def invoke(
        self,
        *,
        pipeline: Any,
        instructions: str,
        span_name: str,
        span_type: str,
        output_model: Optional[type[BaseModel]] = None,
        printer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Any:
        printer_kwargs = printer_kwargs or {}
        return await pipeline.agent_step(
            agent=self.agent,
            instructions=instructions,
            span_name=span_name,
            span_type=span_type,
            output_model=output_model,
            **printer_kwargs,
        )


class PipelineContext:
    """Holds shared conversation state and prompt rendering utilities."""

    def __init__(self, state: ConversationState):
        self.state = state

    def render_prompt(self, profile: str, template: str, payload: Dict[str, Any]) -> str:
        return runtime_prompts.render(profile, template, **payload)

    # Convenience accessors for readability
    @property
    def iteration(self) -> ConversationState:
        return self.state
