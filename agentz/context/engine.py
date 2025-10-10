from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Union

from pydantic import BaseModel

from agentz.context.behavior_profiles import behavior_registry
from agentz.context.conversation import ConversationState, IterationRecord, ToolExecutionResult

Payload = Dict[str, Any]


class SnapshotBuilder(Protocol):
    """Callable protocol for building agent-specific snapshot payloads."""

    def __call__(self, state: ConversationState) -> Union[BaseModel, Mapping[str, Any]]:
        ...


class OutputHandler(Protocol):
    """Callable protocol for applying agent outputs back to conversation state."""

    def __call__(self, state: ConversationState, output: Any) -> None:
        ...


def _coerce_payload(data: Union[BaseModel, Mapping[str, Any], None]) -> Payload:
    """Normalise various payload inputs into a mutable dict."""
    if data is None:
        return {}
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, Mapping):
        return dict(data)
    raise TypeError(f"Unsupported payload type: {type(data)!r}")


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
    """Central coordinator for conversation state and agent I/O."""

    def __init__(
        self,
        state: ConversationState,
        behaviors: Optional[list[str]] = None,
        *,
        config: Optional[Any] = None,
        behavior_agents: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initialize context engine with state and optional behaviors.

        Args:
            state: The conversation state to manage
            behaviors: Optional list of behavior names to auto-register.
                      If provided, will register input builders and output handlers
                      from behavior definitions.
            config: Optional pipeline configuration for downstream agent creation
            behavior_agents: Optional mapping of behavior key -> agent registry name
        """
        self._state = state
        self.registry = ContextRegistry()
        self._config = config
        self._behavior_agents = dict(behavior_agents or {})

        # Auto-register behaviors if provided
        if behaviors:
            for behavior_key in behaviors:
                behavior = behavior_registry.get(behavior_key)
                if behavior.input_builder:
                    self.registry.register_snapshot(behavior_key, behavior.input_builder)
                if behavior.output_handler:
                    self.registry.register_output(behavior_key, behavior.output_handler)

    @property
    def state(self) -> ConversationState:
        return self._state

    @property
    def config(self) -> Optional[Any]:
        return self._config

    @property
    def behavior_agents(self) -> Dict[str, str]:
        return self._behavior_agents

    def register_behavior_agent(self, behavior_key: str, agent_name: str) -> None:
        self._behavior_agents[behavior_key] = agent_name

    # ------------------------------------------------------------------
    # Behavior rendering
    # ------------------------------------------------------------------
    def render_behavior(self, key: str, payload: Union[BaseModel, Mapping[str, Any], None] = None) -> str:
        """Render a behavior template using the supplied payload."""
        behavior = behavior_registry.get(key)
        data = _coerce_payload(payload)
        return behavior.render_template(data)

    def behavior_instructions(self, key: str) -> str:
        """Get the base instructions for a behavior."""
        behavior = behavior_registry.get(key)
        return behavior.instructions

    # ------------------------------------------------------------------
    # Snapshot + output registration
    # ------------------------------------------------------------------
    def register_snapshot(self, agent_key: str, builder: SnapshotBuilder) -> None:
        """Register a snapshot builder for an agent."""
        self.registry.register_snapshot(agent_key, builder)

    def register_output_handler(self, agent_key: str, handler: OutputHandler) -> None:
        """Register an output handler for an agent."""
        self.registry.register_output(agent_key, handler)

    def snapshot(self, agent_key: str) -> Payload:
        """Build a snapshot payload for an agent using its registered builder."""
        builder = self.registry.get_snapshot(agent_key)
        if builder is None:
            return {}
        return _coerce_payload(builder(self._state))

    def apply_output(self, agent_key: str, output: Any) -> None:
        """Apply an agent's output to state using its registered handler."""
        handler = self.registry.get_output(agent_key)
        if handler:
            handler(self._state, output)

    def __getitem__(self, behavior_key: str) -> "BehaviorHandle":
        return BehaviorHandle(engine=self, key=behavior_key)

    # ------------------------------------------------------------------
    # Conversation state helpers
    # ------------------------------------------------------------------
    def begin_iteration(self) -> IterationRecord:
        """Start a new iteration and return its record."""
        return self._state.begin_iteration()

    def mark_iteration_complete(self) -> None:
        """Mark the current iteration as complete."""
        self._state.mark_iteration_complete()

    def mark_research_complete(self) -> None:
        """Mark the entire research process as complete."""
        self._state.mark_research_complete()

    def record_tool_execution(self, result: ToolExecutionResult) -> None:
        """Append a tool execution result to the current iteration."""
        self._state.current_iteration.tools.append(result)

    def add_finding(self, finding: str) -> None:
        """Add a finding to the current iteration."""
        self._state.current_iteration.findings.append(finding)

    def set_summary(self, summary: str) -> None:
        """Update the conversation summary."""
        self._state.update_summary(summary)

    def set_final_report(self, report: str) -> None:
        """Set the final research report."""
        self._state.final_report = report


@dataclass(frozen=True)
class BehaviorHandle:
    """Lightweight wrapper exposing behavior runtime utilities."""

    engine: ContextEngine
    key: str

    @property
    def agent_name(self) -> Optional[str]:
        return self.engine.behavior_agents.get(self.key)

    @property
    def instructions(self) -> str:
        return self.engine.behavior_instructions(self.key)

    def render(self, payload: Union[BaseModel, Mapping[str, Any], None] = None) -> str:
        return self.engine.render_behavior(self.key, payload)

    def snapshot(self) -> Payload:
        return self.engine.snapshot(self.key)

    def apply_output(self, output: Any) -> None:
        self.engine.apply_output(self.key, output)

    @property
    def state(self) -> ConversationState:
        return self.engine.state

    @property
    def config(self) -> Optional[Any]:
        return self.engine.config
