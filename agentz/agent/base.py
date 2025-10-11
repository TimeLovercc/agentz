from __future__ import annotations

import json
from typing import Any, Callable, Optional

from pydantic import BaseModel

from agents import Agent, RunResult, Runner
from agents.run_context import TContext

PromptBuilder = Callable[[Any, Any, "ContextAgent"], str]


class ContextAgent(Agent[TContext]):
    """Capability-centric wrapper that binds LLM + tools + typed IO contract."""

    def __init__(
        self,
        *args: Any,
        input_model: type[BaseModel] | None = None,
        output_model: type[BaseModel] | None = None,
        prompt_builder: PromptBuilder | None = None,
        default_span_type: str = "agent",
        output_parser: Optional[Callable[[str], Any]] = None,
        **kwargs: Any,
    ) -> None:
        if output_model and kwargs.get("output_type"):
            raise ValueError("Use either output_model or output_type, not both.")
        if output_model is not None:
            kwargs["output_type"] = output_model

        super().__init__(*args, **kwargs)

        self.input_model = input_model
        self.output_model = self._coerce_output_model(output_model or getattr(self, "output_type", None))
        self.prompt_builder = prompt_builder
        self.default_span_type = default_span_type
        self.output_parser = output_parser

    @classmethod
    def from_profile(cls, profile: Any, llm: str) -> "ContextAgent":
        """Create a ContextAgent from a Profile instance.

        Automatically derives agent name from the profile's _key attribute.
        Example: profiles["observe"] has _key="observe" → agent name becomes "observe_agent"

        Args:
            profile: Profile instance with instructions, schemas, tools (from profiles dict)
            llm: LLM model name (e.g., "gpt-4", "claude-3-5-sonnet")

        Returns:
            ContextAgent instance configured from the profile

        Example:
            agent = ContextAgent.from_profile(profiles["observe"], "gpt-4")
            # Creates agent with name="observe_agent"
        """
        # Auto-derive name from profile key
        profile_key = getattr(profile, "_key", "agent")
        agent_name = profile_key + "_agent" if profile_key != "agent" else "agent"

        # Get tools, default to empty list if None
        tools = getattr(profile, "tools", None) or []

        return cls(
            name=agent_name,
            instructions=profile.instructions,
            output_model=getattr(profile, "output_schema", None),
            input_model=getattr(profile, "input_schema", None),
            tools=tools,
            model=llm,
        )

    @staticmethod
    def _coerce_output_model(candidate: Any) -> type[BaseModel] | None:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
        return None

    def _coerce_input(self, payload: Any) -> Any:
        if self.input_model is None or payload is None:
            return payload
        if isinstance(payload, self.input_model):
            return payload
        if isinstance(payload, BaseModel):
            return self.input_model.model_validate(payload.model_dump())
        if isinstance(payload, dict):
            return self.input_model.model_validate(payload)
        msg = f"{self.name} expects input compatible with {self.input_model.__name__}"
        raise TypeError(msg)

    @staticmethod
    def _to_prompt_payload(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, BaseModel):
            return payload.model_dump()
        if isinstance(payload, dict):
            return payload
        return {"input": payload}

    def build_prompt(
        self,
        payload: Any = None,
        *,
        context: Any = None,
        template: Optional[str] = None,
    ) -> str:
        validated = self._coerce_input(payload)

        if self.prompt_builder:
            return self.prompt_builder(validated, context, self)

        if context is not None and template:
            builder = getattr(context, "build_prompt", None)
            if builder is None:
                raise AttributeError("Context object must expose build_prompt(...)")
            prompt_data = self._to_prompt_payload(validated)
            return builder(agent=self, template_name=template, data=prompt_data)

        if isinstance(validated, str):
            return validated
        if isinstance(validated, BaseModel):
            return validated.model_dump_json(indent=2)
        if isinstance(validated, dict):
            return json.dumps(validated, indent=2)

        if validated is None and isinstance(self.instructions, str):
            return self.instructions

        return str(validated)

    async def invoke(
        self,
        *,
        pipeline: Any,
        span_name: str,
        payload: Any = None,
        prompt: Optional[str] = None,
        context: Any = None,
        template: Optional[str] = None,
        span_type: Optional[str] = None,
        output_model: Optional[type[BaseModel]] = None,
        printer_key: Optional[str] = None,
        printer_title: Optional[str] = None,
        printer_group_id: Optional[str] = None,
        printer_border_style: Optional[str] = None,
        **span_kwargs: Any,
    ) -> Any:
        instructions = prompt or self.build_prompt(payload, context=context, template=template)
        model = output_model or self.output_model

        return await pipeline.agent_step(
            agent=self,
            instructions=instructions,
            span_name=span_name,
            span_type=span_type or self.default_span_type,
            output_model=model,
            printer_key=printer_key,
            printer_title=printer_title,
            printer_group_id=printer_group_id,
            printer_border_style=printer_border_style,
            **span_kwargs,
        )

    async def parse_output(self, run_result: RunResult) -> RunResult:
        """Apply legacy string parser only when no structured output is configured."""
        if self.output_parser and self.output_model is None:
            run_result.final_output = self.output_parser(run_result.final_output)
        return run_result


class ContextRunner(Runner):
    """Runner shim that invokes ContextAgent.parse_output after execution."""

    @classmethod
    async def run(cls, *args: Any, **kwargs: Any) -> RunResult:
        result = await Runner.run(*args, **kwargs)
        starting_agent = kwargs.get("starting_agent") or (args[0] if args else None)

        if isinstance(starting_agent, ContextAgent):
            return await starting_agent.parse_output(result)
        return result
