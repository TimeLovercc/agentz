from __future__ import annotations

import json
import inspect
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

from agents import Agent, RunResult, Runner
from agents.run_context import TContext
from agentz.profiles.registry import create_agents
from agentz.context.engine import BehaviorHandle

PromptBuilder = Callable[[Any, Any, "ResearchAgent"], str]


class ResearchAgent(Agent[TContext]):
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


class ResearchRunner(Runner):
    """Runner shim that invokes ResearchAgent.parse_output after execution."""

    @classmethod
    async def run(cls, *args: Any, **kwargs: Any) -> RunResult:
        result = await Runner.run(*args, **kwargs)
        starting_agent = kwargs.get("starting_agent") or (args[0] if args else None)

        if isinstance(starting_agent, ResearchAgent):
            return await starting_agent.parse_output(result)
        return result


class ContextAgent:
    """Minimal agent facade that binds a behavior handle to a concrete runtime agent."""

    def __init__(
        self,
        handle: BehaviorHandle,
        *,
        span_name: str,
        span_type: str,
        printer_key: str,
        printer_title: str,
        output_model: Optional[type[BaseModel]] = None,
    ):
        self.handle = handle
        self.span_name = span_name
        self.span_type = span_type
        self.printer_key = printer_key
        self.printer_title = printer_title
        self.output_model = output_model
        self._agent: Optional[Agent] = None
        self._pipeline: Optional[Any] = None

    def bind(self, pipeline: Any) -> None:
        """Bind the context agent to a pipeline runtime."""
        self._pipeline = pipeline

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            agent_name = self.handle.agent_name or f"{self.handle.key}_agent"
            config = self.handle.config
            if config is None:
                raise ValueError("ContextAgent requires context engine to include configuration.")
            self._agent = create_agents(agent_name, config)
        return self._agent

    async def __call__(self, payload: Optional[Any] = None) -> Any:
        if self._pipeline is None:
            frame = inspect.currentframe()
            try:
                caller = frame.f_back if frame else None
                candidate = caller.f_locals.get("self") if caller else None
                if candidate is None or not hasattr(candidate, "agent_step"):
                    raise RuntimeError("ContextAgent could not infer pipeline binding automatically.")
                self._pipeline = candidate
            finally:
                del frame

        snapshot = self.handle.snapshot()
        render_payload: Dict[str, Any] = {}

        if isinstance(snapshot, Mapping):
            render_payload.update(snapshot)
        elif isinstance(snapshot, BaseModel):
            render_payload.update(snapshot.model_dump())

        if payload is not None:
            if isinstance(payload, Mapping):
                render_payload.update(payload)
            elif isinstance(payload, BaseModel):
                render_payload.update(payload.model_dump())
            else:
                render_payload["input"] = payload

        instructions = self.handle.render(render_payload or None)

        result = await self._pipeline.agent_step(
            agent=self.agent,
            instructions=instructions,
            span_name=self.span_name,
            span_type=self.span_type,
            output_model=self.output_model,
            printer_key=self.printer_key,
            printer_title=self.printer_title,
            printer_group_id=getattr(self._pipeline, "current_printer_group", None),
        )

        self.handle.apply_output(result)
        return result
