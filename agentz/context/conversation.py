from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, List, Optional, Sequence, Set, Tuple, Type
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, create_model
from loguru import logger
from agentz.profiles.base import load_all_profiles, ToolAgentOutput

if TYPE_CHECKING:
    from agentz.profiles.manager.routing import AgentSelectionPlan, AgentTask
    from agentz.profiles.manager.evaluate import EvaluateOutput
else:
    AgentSelectionPlan = AgentTask = Any
    EvaluateOutput = Any


def _collect_profile_output_models(extra_models: Optional[Sequence[Type[BaseModel]]]) -> List[Type[BaseModel]]:
    """Gather all declared output schemas from registered profiles."""
    profiles = load_all_profiles()
    models: List[Type[BaseModel]] = []
    seen: Set[str] = set()

    def _maybe_add(model: Optional[Type[BaseModel]]) -> None:
        if model is None:
            return
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            return
        key = f"{model.__module__}.{model.__qualname__}"
        if key in seen:
            return
        seen.add(key)
        models.append(model)

    for profile in profiles.values():
        _maybe_add(getattr(profile, "output_schema", None))

    if extra_models:
        for model in extra_models:
            _maybe_add(model)

    # Always include ToolAgentOutput so tools can be recorded even if no profile declares it explicitly
    _maybe_add(ToolAgentOutput)
    return models or [ToolAgentOutput]


def _build_union_type(models: Iterable[Type[BaseModel]]) -> Type[BaseModel]:
    iterator = iter(models)
    union_type: Type[BaseModel] = next(iterator)
    for model in iterator:
        union_type = union_type | model  # type: ignore[operator]
    return union_type


def _build_iteration_record_model(
    output_models: Sequence[Type[BaseModel]],
    custom_payload_model: Optional[Type[BaseModel]],
) -> Type["IterationRecordBase"]:
    output_union = _build_union_type(output_models)

    custom_type: Any
    if custom_payload_model:
        if not isinstance(custom_payload_model, type) or not issubclass(custom_payload_model, BaseModel):
            raise TypeError("custom_payload_model must be a subclass of BaseModel")
        custom_type = Optional[custom_payload_model]
    else:
        custom_type = Optional[BaseModel]

    field_definitions = {
        "payloads": (List[output_union], Field(default_factory=list)),
        "custom_payload": (custom_type, None),
    }

    iteration_model: Type[IterationRecordBase] = create_model(
        "IterationRecord",
        __base__=IterationRecordBase,
        __module__=IterationRecordBase.__module__,
        **field_definitions,
    )
    iteration_model._output_union = output_union  # type: ignore[attr-defined]
    return iteration_model


class IterationRecordBase(BaseModel):
    """State captured for a single iteration of the research loop."""

    index: int
    observation: Optional[str] = None
    evaluation: Optional[EvaluateOutput] = None
    selected_gap: Optional[str] = None
    route_plan: Optional[AgentSelectionPlan] = None
    tools: List[ToolAgentOutput] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    payloads: List[Any] = Field(default_factory=list)
    custom_payload: Optional[BaseModel] = None
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

        if self.selected_gap:
            lines.append(f"<task>\nAddress this knowledge gap: {self.selected_gap}\n</task>")

        if self.route_plan and self.route_plan.tasks:
            task_lines = []
            for task in self.route_plan.tasks:
                entity = task.entity_website or "null"
                task_lines.append(f"[Agent] {task.agent} [Query] {task.query} [Entity] {entity}")
            lines.append(
                "<action>\nCalling the following tools to address the knowledge gap:\n"
                + "\n".join(task_lines)
                + "\n</action>"
            )

        if self.tools:
            findings_text = "\n\n".join(result.as_history_block() for result in self.tools)
            lines.append(f"<findings>\n{findings_text}\n</findings>")
        elif self.findings:
            findings_text = "\n\n".join(self.findings)
            lines.append(f"<findings>\n{findings_text}\n</findings>")

        return "\n\n".join(lines).strip()

    def add_payload(self, value: Any) -> BaseModel:
        """Validate and record a structured payload for this iteration."""
        payload = value

        expected_union = getattr(self.__class__, "_output_union", None)
        union_args: Tuple[Type[BaseModel], ...] = ()
        if expected_union is not None:
            union_args = getattr(expected_union, "__args__", ()) or ()

        if isinstance(payload, BaseModel):
            if union_args and not isinstance(payload, union_args):
                payload = self._coerce_payload(payload.model_dump(), union_args)
        elif isinstance(payload, dict):
            payload = self._coerce_payload(payload, union_args)
        else:
            if union_args:
                raise TypeError(
                    f"Payload type {type(value)!r} is incompatible with expected schemas {union_args}"
                )
            raise TypeError(f"Payload type {type(value)!r} is not supported")

        if not isinstance(payload, BaseModel):
            raise TypeError("Payload coercion must produce a BaseModel instance")

        self.payloads.append(payload)
        return payload

    @staticmethod
    def _coerce_payload(
        payload_data: dict,
        union_args: Tuple[Type[BaseModel], ...],
    ) -> BaseModel:
        if not union_args:
            raise TypeError("No output schemas are registered for payload coercion")
        errors: List[ValidationError] = []
        for candidate in union_args:
            try:
                return candidate.model_validate(payload_data)
            except ValidationError as exc:
                errors.append(exc)
        raise ValidationError.from_exception_data(
            title="Iteration payload validation failed",
            line_errors=[err for exc in errors for err in exc.errors()],
        ) from (errors[-1] if errors else None)


try:
    _DEFAULT_OUTPUT_MODELS = _collect_profile_output_models(extra_models=None)
except Exception as exc:  # pragma: no cover - defensive fallback
    logger.warning("Failed to auto-discover profile output schemas: {}", exc)
    _DEFAULT_OUTPUT_MODELS = [ToolAgentOutput]
IterationRecord = _build_iteration_record_model(_DEFAULT_OUTPUT_MODELS, custom_payload_model=None)


class ConversationState(BaseModel):
    """Centralised storage for all iteration data shared across agents."""

    iterations: List[IterationRecordBase] = Field(default_factory=list)
    final_report: Optional[str] = None
    started_at: Optional[float] = None
    complete: bool = False
    summary: Optional[str] = None
    query: Optional[str] = None

    _iteration_model: Type[IterationRecordBase] = PrivateAttr(default=IterationRecord)
    _output_models: Tuple[Type[BaseModel], ...] = PrivateAttr(
        default=tuple(_DEFAULT_OUTPUT_MODELS)
    )

    def __init__(
        self,
        *,
        extra_output_models: Optional[Sequence[Type[BaseModel]]] = None,
        custom_iteration_model: Optional[Type[BaseModel]] = None,
        **data: Any,
    ):
        super().__init__(**data)
        output_models = (
            _collect_profile_output_models(extra_output_models)
            if extra_output_models
            else list(_DEFAULT_OUTPUT_MODELS)
        )
        iteration_model = (
            IterationRecord
            if not extra_output_models and custom_iteration_model is None
            else _build_iteration_record_model(output_models, custom_iteration_model)
        )
        object.__setattr__(self, "_output_models", tuple(output_models))
        object.__setattr__(self, "_iteration_model", iteration_model)

    def start_timer(self) -> None:
        self.started_at = time.time()

    def elapsed_minutes(self) -> float:
        if self.started_at is None:
            return 0.0
        return (time.time() - self.started_at) / 60

    def begin_iteration(self) -> IterationRecordBase:
        iteration = self._iteration_model(index=len(self.iterations) + 1)
        self.iterations.append(iteration)
        return iteration

    @property
    def current_iteration(self) -> IterationRecordBase:
        if not self.iterations:
            raise ValueError("No iteration has been started yet.")
        return self.iterations[-1]

    def mark_iteration_complete(self) -> None:
        self.current_iteration.mark_complete()

    def mark_research_complete(self) -> None:
        self.complete = True
        self.current_iteration.mark_complete()

    def iteration_history(self, include_current: bool = False) -> str:
        relevant = [
            iteration
            for iteration in self.iterations
            if iteration.is_complete() or include_current and iteration is self.current_iteration
        ]
        blocks = [iteration.history_block() for iteration in relevant if iteration.history_block()]
        return "\n\n".join(blocks).strip()

    def unsummarized_history(self, include_current: bool = True) -> str:
        relevant = [
            iteration
            for iteration in self.iterations
            if (iteration.is_complete() or include_current and iteration is self.current_iteration)
            and not iteration.summarized
        ]
        blocks = [iteration.history_block() for iteration in relevant if iteration.history_block()]
        return "\n\n".join(blocks).strip()

    def history_with_summary(self) -> str:
        summary_section = ""
        if self.summary:
            summary_section = f"[SUMMARY BEFORE NEW ITERATION]\n\n{self.summary}\n\n"
        return summary_section + self.unsummarized_history()

    @property
    def iteration_payload_union(self) -> Type[BaseModel]:
        return getattr(self._iteration_model, "_output_union", BaseModel)

    def set_query(self, query: str) -> None:
        self.query = query

    def record_payload(self, payload: Any) -> BaseModel:
        """Attach a structured payload to the current iteration."""
        iteration = self.current_iteration if self.iterations else self.begin_iteration()
        return iteration.add_payload(payload)

    def set_custom_iteration_payload(self, payload: BaseModel) -> None:
        """Store an optional, caller-supplied payload for the current iteration."""
        self.current_iteration.custom_payload = payload

    def all_findings(self) -> List[str]:
        findings: List[str] = []
        for iteration in self.iterations:
            findings.extend(iteration.findings)
            findings.extend(result.output.output for result in iteration.tools if hasattr(result.output, "output"))
        return findings

    def findings_text(self) -> str:
        findings = self.all_findings()
        return "\n\n".join(findings).strip() if findings else ""

    def update_summary(self, summary: str) -> None:
        self.summary = summary
        for iteration in self.iterations:
            iteration.mark_summarized()
