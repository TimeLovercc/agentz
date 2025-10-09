from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel

from agentz.context.engine import ContextEngine
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
    """Facade over ContextEngine used by flow executors."""

    def __init__(
        self,
        state: ConversationState,
        *,
        engine: Optional[ContextEngine] = None,
    ):
        self._engine = engine or ContextEngine(state)

    @property
    def engine(self) -> ContextEngine:
        return self._engine

    @property
    def state(self) -> ConversationState:
        return self._engine.state

    def render_prompt(self, profile: str, template: str, payload: Dict[str, Any]) -> str:
        return self._engine.render_prompt(profile, template, payload)

    def build_prompt(
        self,
        agent_key: str,
        profile: str,
        template: str,
        *,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self._engine.build_prompt(
            agent_key,
            profile,
            template,
            overrides=overrides,
        )

    def register_snapshot(self, agent_key: str, builder) -> None:
        self._engine.register_snapshot(agent_key, builder)

    def register_output_handler(self, agent_key: str, handler) -> None:
        self._engine.register_output_handler(agent_key, handler)

    def snapshot(self, agent_key: str) -> Dict[str, Any]:
        return self._engine.snapshot(agent_key)

    def apply_output(self, agent_key: str, output: Any) -> None:
        self._engine.apply_output(agent_key, output)

    # Backwards compatibility for existing usage
    @property
    def iteration(self) -> ConversationState:
        return self.state
