from __future__ import annotations

import json
from typing import Any, Callable, Optional

from pydantic import BaseModel

from agents import Agent, RunResult
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
        self._pipeline = None  # Optional pipeline reference for context-aware execution
        self._role = None  # Optional role identifier for automatic iteration tracking

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
        # Note: Tools in profiles are currently string names, not resolved objects
        # This causes errors in the agents library which expects tool objects
        # For now, pass empty list to avoid runtime errors
        # TODO: Implement tool resolution to make tools actually work
        tools = []

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

    def _coerce_input(self, payload: Any, strict: bool = True) -> Any:
        if self.input_model is None or payload is None:
            return payload
        if isinstance(payload, self.input_model):
            return payload
        if isinstance(payload, BaseModel):
            return self.input_model.model_validate(payload.model_dump())
        if isinstance(payload, dict):
            return self.input_model.model_validate(payload)

        # If strict mode is disabled, allow passthrough
        if not strict:
            return payload

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

    async def __call__(self, payload: Any = None, group_id: Optional[str] = None) -> Any:
        """Make ContextAgent callable directly.

        This allows usage like: result = await agent(input_data)

        When called within a pipeline context (self._pipeline is set), uses the
        pipeline's agent_step for full tracking/tracing. Otherwise, uses ContextRunner.

        Note: When calling directly without pipeline context, input validation
        is relaxed to allow string inputs even if agent has a defined input_model.

        Args:
            payload: Input data for the agent
            group_id: Optional group ID for tracking (used with pipeline context)

        Returns:
            Parsed output if in pipeline context, otherwise RunResult
        """
        # Build prompt with non-strict input coercion
        # This allows string inputs to pass through without validation errors
        validated = self._coerce_input(payload, strict=False)

        if isinstance(validated, str):
            instructions = validated
        elif isinstance(validated, BaseModel):
            instructions = validated.model_dump_json(indent=2)
        elif isinstance(validated, dict):
            import json
            instructions = json.dumps(validated, indent=2)
        elif validated is None and isinstance(self.instructions, str):
            instructions = self.instructions
        else:
            instructions = str(validated)

        # If pipeline context is available, use it for full tracking
        if self._pipeline is not None:
            result = await self._pipeline.agent_step(
                agent=self,
                instructions=instructions,
                group_id=group_id,
            )
            # Extract final output for cleaner API
            output = result.final_output if hasattr(result, 'final_output') else result

            # Automatic iteration tracking based on role
            if self._role and hasattr(self._pipeline, 'context'):
                try:
                    iteration = self._pipeline.context.current_iteration

                    # Special handling for "observe" role - set iteration.observation
                    if self._role == "observe":
                        serialized = self._pipeline._serialize_output(output)
                        iteration.observation = serialized

                    # Record structured payload for all roles
                    self._pipeline._record_structured_payload(output, context_label=self._role)
                except Exception:
                    # Silently skip if context/iteration not available
                    pass

            return output

        # Otherwise, use ContextRunner to execute the agent
        from agentz.runner import ContextRunner

        result = await ContextRunner.run(
            starting_agent=self,
            input=instructions,
        )

        return result

    async def parse_output(self, run_result: RunResult) -> RunResult:
        """Apply legacy string parser only when no structured output is configured."""
        if self.output_parser and self.output_model is None:
            run_result.final_output = self.output_parser(run_result.final_output)
        return run_result
