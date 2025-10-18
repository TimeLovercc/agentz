from __future__ import annotations

import time
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Set, Tuple, Type
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, create_model
from agentz.profiles.base import Profile, ToolAgentOutput


class BaseIterationRecord(BaseModel):
    """State captured for a single iteration of the research loop."""

    index: int
    observation: Optional[str] = None
    tools: List[ToolAgentOutput] = Field(default_factory=list)
    payloads: List[Any] = Field(default_factory=list)
    status: str = Field(default="pending", description="Iteration status: pending or complete")
    summarized: bool = Field(default=False, description="Whether this iteration has been summarised")
    _output_union: ClassVar[Optional[Type[BaseModel]]] = None  # type: ignore[var-annotated]

    def mark_complete(self) -> None:
        self.status = "complete"

    def is_complete(self) -> bool:
        return self.status == "complete"

    def mark_summarized(self) -> None:
        self.summarized = True

    def history_block(self) -> str:
        """Render this iteration as a formatted history block for prompts."""
        lines: List[str] = [f"[ITERATION {self.index}]"]

        if self.observation:
            lines.append(f"<thought>\n{self.observation}\n</thought>")

        # Render structured payloads generically
        if self.payloads:
            payload_lines = []
            for payload in self.payloads:
                if isinstance(payload, BaseModel):
                    payload_lines.append(payload.model_dump_json(indent=2))
                else:
                    payload_lines.append(str(payload))
            if payload_lines:
                lines.append(f"<payloads>\n{chr(10).join(payload_lines)}\n</payloads>")

        # Render tool execution results
        if self.tools:
            tool_lines = [tool.output for tool in self.tools]
            lines.append(f"<findings>\n{chr(10).join(tool_lines)}\n</findings>")

        return "\n\n".join(lines).strip()

    def add_payload(self, value: Any) -> BaseModel:
        expected_union = getattr(self.__class__, "_output_union", None)
        union_args: Tuple[Type[BaseModel], ...] = ()
        if expected_union is not None:
            union_args = getattr(expected_union, "__args__", ()) or ()

        if isinstance(value, BaseModel):
            payload = value
            if union_args and not isinstance(payload, union_args):
                data = payload.model_dump()
            else:
                self.payloads.append(payload)
                return payload
        elif isinstance(value, dict):
            data = value
        else:
            if union_args:
                raise TypeError(
                    f"Payload type {type(value)!r} is incompatible with expected schemas {union_args}"
                )
            raise TypeError(f"Payload type {type(value)!r} is not supported")

        if not union_args:
            raise TypeError("No output schemas are registered for payload coercion")

        errors: List[ValidationError] = []
        for candidate in union_args:
            try:
                payload = candidate.model_validate(data)
                self.payloads.append(payload)
                return payload
            except ValidationError as exc:
                errors.append(exc)

        raise ValidationError.from_exception_data(
            title="Iteration payload validation failed",
            line_errors=[err for exc in errors for err in exc.errors()],
        ) from (errors[-1] if errors else None)



class ConversationState(BaseModel):
    iterations: List[BaseIterationRecord] = Field(default_factory=list)
    final_report: Optional[str] = None
    started_at: Optional[float] = None
    complete: bool = False
    summary: Optional[str] = None
    query: Optional[Any] = None  # Store original query object
    formatted_query: Optional[str] = None  # Store pre-formatted query string

    _iteration_model: Type[BaseIterationRecord] = PrivateAttr()
    _runtime_context: Dict[str, str] = PrivateAttr(default_factory=dict)

    def start_timer(self) -> None:
        self.started_at = time.time()

    @property
    def iteration(self) -> str:
        """Current iteration index as string."""
        try:
            return str(self.current_iteration.index)
        except (ValueError, AttributeError):
            return '1'

    @property
    def history(self) -> str:
        """Previous iteration history (excluding current)."""
        return self.iteration_history(include_current=False) or 'No previous iterations.'

    @property
    def observation(self) -> str:
        """Current iteration's observation."""
        try:
            obs = self.current_iteration.observation
            return obs if obs else ''
        except (ValueError, AttributeError):
            return ''

    @property
    def last_summary(self) -> str:
        """Last generated summary."""
        return self.summary if self.summary else ''

    @property
    def conversation_history(self) -> str:
        """Full conversation history (alias for iteration_history)."""
        return self.iteration_history(include_current=True)

    @property
    def elapsed_minutes(self) -> str:
        """Elapsed time in minutes as string."""
        if self.started_at is None:
            return '0'
        return str((time.time() - self.started_at) / 60)

    def begin_iteration(self) -> BaseIterationRecord:
        iteration = self._iteration_model(index=len(self.iterations) + 1)
        self.iterations.append(iteration)
        return iteration

    @property
    def current_iteration(self) -> BaseIterationRecord:
        if not self.iterations:
            raise ValueError("No iteration has been started yet.")
        return self.iterations[-1]

    def mark_iteration_complete(self) -> None:
        self.current_iteration.mark_complete()

    def mark_research_complete(self) -> None:
        self.complete = True
        self.current_iteration.mark_complete()

    def get_history_blocks(self, include_current: bool, only_unsummarized: bool = False) -> str:
        relevant = [
            iteration
            for iteration in self.iterations
            if (iteration.is_complete() or include_current and iteration is self.current_iteration)
            and (not only_unsummarized or not iteration.summarized)
        ]
        blocks = [iteration.history_block() for iteration in relevant if iteration.history_block()]
        return "\n\n".join(blocks).strip()

    def iteration_history(self, include_current: bool = False) -> str:
        return self.get_history_blocks(include_current, only_unsummarized=False)

    def unsummarized_history(self, include_current: bool = True) -> str:
        return self.get_history_blocks(include_current, only_unsummarized=True)

    def set_query(self, query: Any) -> None:
        """Set the query - stores the original query object."""
        self.query = query

    def set_runtime_context(self, key: str, value: str) -> None:
        """Set a runtime context value for template placeholders."""
        self._runtime_context[key] = value

    def get_runtime_context(self, key: str, default: str = '') -> str:
        """Get a runtime context value."""
        return self._runtime_context.get(key, default)

    def update_runtime_context(self, context_dict: Dict[str, str]) -> None:
        """Update multiple runtime context values at once."""
        self._runtime_context.update(context_dict)

    def record_payload(self, payload: Any) -> BaseModel:
        """Attach a structured payload to the current iteration."""
        iteration = self.current_iteration if self.iterations else self.begin_iteration()
        return iteration.add_payload(payload)

    def all_findings(self) -> List[str]:
        findings: List[str] = []
        for iteration in self.iterations:
            findings.extend(tool.output for tool in iteration.tools)
        return findings

    def findings_text(self) -> str:
        findings = self.all_findings()
        return "\n\n".join(findings).strip() if findings else ""

    def update_summary(self, summary: str) -> None:
        self.summary = summary
        for iteration in self.iterations:
            iteration.mark_summarized()

    def format_context_prompt(self, current_input: Optional[str] = None) -> str:
        """Format a comprehensive context prompt including query, history, and current input.

        Args:
            current_input: The current input/payload for this agent call

        Returns:
            Formatted context prompt string
        """
        sections = []

        # Add original query if available
        if self.query:
            sections.append(f"[ORIGINAL QUERY]\n{self.query}")

        # Add previous iteration history (excluding current iteration)
        history = self.iteration_history(include_current=False)
        if history:
            sections.append(f"[PREVIOUS ITERATIONS]\n{history}")

        # Add current input if provided
        if current_input:
            sections.append(f"[CURRENT INPUT]\n{current_input}")

        return "\n\n".join(sections).strip()


def create_conversation_state(profiles: Dict[str, Profile]) -> "ConversationState":
    models: List[Type[BaseModel]] = []
    seen: Set[str] = set()

    for profile in profiles.values():
        model = getattr(profile, "output_schema", None)
        if model is not None and isinstance(model, type) and issubclass(model, BaseModel):
            key = f"{model.__module__}.{model.__qualname__}"
            if key not in seen:
                seen.add(key)
                models.append(model)

    if not models:
        models = [ToolAgentOutput]

    iterator = iter(models)
    union_type: Type[BaseModel] = next(iterator)
    for model in iterator:
        union_type = union_type | model  # type: ignore[operator]

    field_definitions = {
        "payloads": (List[union_type], Field(default_factory=list)),
    }

    iteration_model: Type[BaseIterationRecord] = create_model(
        "IterationRecord",
        __base__=BaseIterationRecord,
        __module__=BaseIterationRecord.__module__,
        **field_definitions,
    )
    iteration_model._output_union = union_type  # type: ignore[attr-defined]

    state = ConversationState()
    object.__setattr__(state, "_iteration_model", iteration_model)
    return state
