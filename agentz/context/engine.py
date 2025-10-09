from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Protocol, Union

from pydantic import BaseModel

from agentz.memory.behavior_profiles import runtime_prompts
from agentz.memory.conversation import ConversationState, IterationRecord, ToolExecutionResult

Payload = Dict[str, Any]


class SnapshotBuilder(Protocol):
    """Callable protocol for building agent-specific snapshot payloads."""

    def __call__(self, state: ConversationState) -> Union[BaseModel, Mapping[str, Any]]:
        ...


class OutputHandler(Protocol):
    """Callable protocol for applying agent outputs back to conversation state."""

    def __call__(self, state: ConversationState, output: Any) -> None:
        ...


def _render_template(template: str, placeholders: Mapping[str, Any]) -> str:
    """Render a template string using [[PLACEHOLDER]] substitutions."""
    text = template
    for key, value in placeholders.items():
        token = f"[[{key}]]"
        text = text.replace(token, str(value))
    return text


def _coerce_payload(data: Union[BaseModel, Mapping[str, Any], None]) -> Payload:
    """Normalise various payload inputs into a mutable dict."""
    if data is None:
        return {}
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, MutableMapping):
        return dict(data)
    if isinstance(data, Mapping):
        return dict(data)
    raise TypeError(f"Unsupported payload type: {type(data)!r}")


class TemplateLibrary:
    """Manages prompt templates with optional runtime overrides."""

    def __init__(self):
        self._overrides: Dict[str, Dict[str, Callable[[Payload], str]]] = {}

    def register_template(self, profile: str, template_name: str, template: str) -> None:
        """Register or override a template string for a profile."""

        def renderer(payload: Payload) -> str:
            return _render_template(template, payload)

        self.register_renderer(profile, template_name, renderer)

    def register_renderer(
        self,
        profile: str,
        template_name: str,
        renderer: Callable[[Payload], str],
    ) -> None:
        """Register a custom renderer callable for a template."""
        self._overrides.setdefault(profile, {})[template_name] = renderer

    def render(self, profile: str, template_name: str, payload: Union[BaseModel, Mapping[str, Any], None] = None) -> str:
        """Render the template using overrides or the runtime prompt store."""
        data = _coerce_payload(payload)

        renderer = self._overrides.get(profile, {}).get(template_name)
        if renderer:
            return renderer(data)

        return runtime_prompts.render(profile, template_name, **data)

    def has_override(self, profile: str, template_name: str) -> bool:
        return template_name in self._overrides.get(profile, {})

    def list_templates(self, profile: Optional[str] = None) -> Dict[str, Dict[str, Callable[[Payload], str]]]:
        """List registered overrides."""
        if profile:
            return {profile: dict(self._overrides.get(profile, {}))}
        return {profile: dict(templates) for profile, templates in self._overrides.items()}


StateHook = Callable[[ConversationState], None]


@dataclass
class ContextRegistry:
    """Holds snapshot builders and output handlers for agents."""

    snapshots: Dict[str, SnapshotBuilder] = field(default_factory=dict)
    outputs: Dict[str, OutputHandler] = field(default_factory=dict)

    def register_snapshot(self, agent_key: str, builder: SnapshotBuilder) -> None:
        self.snapshots[agent_key] = builder

    def register_output(self, agent_key: str, handler: OutputHandler) -> None:
        self.outputs[agent_key] = handler

    def get_snapshot(self, agent_key: str) -> Optional[SnapshotBuilder]:
        return self.snapshots.get(agent_key)

    def get_output(self, agent_key: str) -> Optional[OutputHandler]:
        return self.outputs.get(agent_key)


class ContextEngine:
    """Central coordinator for conversation state and prompt templates."""

    def __init__(
        self,
        state: ConversationState,
        *,
        templates: Optional[TemplateLibrary] = None,
    ) -> None:
        self._state = state
        self.templates = templates or TemplateLibrary()
        self.registry = ContextRegistry()
        self._state_hooks: Dict[str, StateHook] = {}

    @property
    def state(self) -> ConversationState:
        return self._state

    # ------------------------------------------------------------------
    # Template helpers
    # ------------------------------------------------------------------
    def render_prompt(self, profile: str, template_name: str, payload: Union[BaseModel, Mapping[str, Any], None] = None) -> str:
        """Render a prompt template using the supplied payload."""
        return self.templates.render(profile, template_name, payload)

    def build_prompt(
        self,
        agent_key: str,
        profile: str,
        template_name: str,
        *,
        overrides: Optional[Mapping[str, Any]] = None,
        payload: Optional[Union[BaseModel, Mapping[str, Any]]] = None,
    ) -> str:
        """Build a prompt by merging snapshot payloads with optional overrides."""
        snapshot_payload = self.snapshot(agent_key) if payload is None else _coerce_payload(payload)
        if overrides:
            snapshot_payload.update(dict(overrides))
        return self.render_prompt(profile, template_name, snapshot_payload)

    def register_template(
        self,
        profile: str,
        template_name: str,
        template: Union[str, Callable[[Payload], str]],
    ) -> None:
        """Register a template override for a profile."""
        if callable(template):
            self.templates.register_renderer(profile, template_name, template)
        else:
            self.templates.register_template(profile, template_name, template)

    # ------------------------------------------------------------------
    # Snapshot + output registration
    # ------------------------------------------------------------------
    def register_snapshot(self, agent_key: str, builder: SnapshotBuilder) -> None:
        self.registry.register_snapshot(agent_key, builder)

    def register_output_handler(self, agent_key: str, handler: OutputHandler) -> None:
        self.registry.register_output(agent_key, handler)

    def snapshot(self, agent_key: str) -> Payload:
        builder = self.registry.get_snapshot(agent_key)
        if builder is None:
            return {}
        return _coerce_payload(builder(self._state))

    def apply_output(self, agent_key: str, output: Any) -> None:
        handler = self.registry.get_output(agent_key)
        if handler:
            handler(self._state, output)

    # ------------------------------------------------------------------
    # Conversation state helpers
    # ------------------------------------------------------------------
    def begin_iteration(self) -> IterationRecord:
        iteration = self._state.begin_iteration()
        self._notify_hooks("after_iteration_start")
        return iteration

    def mark_iteration_complete(self) -> None:
        self._state.mark_iteration_complete()
        self._notify_hooks("after_iteration_complete")

    def mark_research_complete(self) -> None:
        self._state.mark_research_complete()
        self._notify_hooks("after_research_complete")

    def record_tool_execution(self, result: ToolExecutionResult) -> None:
        """Append a tool execution result to the current iteration."""
        self._state.current_iteration.tools.append(result)
        self._notify_hooks("after_tool_execution")

    def add_finding(self, finding: str) -> None:
        self._state.current_iteration.findings.append(finding)
        self._notify_hooks("after_finding_recorded")

    def set_summary(self, summary: str) -> None:
        self._state.update_summary(summary)
        self._notify_hooks("after_summary_updated")

    def set_final_report(self, report: str) -> None:
        self._state.final_report = report
        self._notify_hooks("after_final_report")

    def register_hook(self, name: str, callback: StateHook) -> None:
        """Register a lifecycle hook executed after specific state transitions."""
        self._state_hooks[name] = callback

    def _notify_hooks(self, name: str) -> None:
        hook = self._state_hooks.get(name)
        if hook:
            hook(self._state)
