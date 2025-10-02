"""Typed configuration support for the data science pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Dict, Optional, Union

from pydantic import ConfigDict, Field

from .base import (
    PipelineConfigBase,
    PipelineConfigSource,
    ResolvedPipelineConfig,
    resolve_config_source,
)

try:  # Optional runtime imports for typing friendliness
    from agents import Agent  # type: ignore
except Exception:  # pragma: no cover - fallback when SDK not installed
    Agent = Any  # type: ignore

try:  # Avoid eager import when not needed
    from agentz.llm.llm_setup import LLMConfig
except Exception:  # pragma: no cover
    LLMConfig = Any  # type: ignore

AgentFactory = Union[Agent, Callable[[LLMConfig], Agent]]  # type: ignore[misc]
ToolFactory = Union[Any, Callable[[LLMConfig], Any]]  # type: ignore[misc]
ManagerAgentInput = Union[Sequence[AgentFactory], Mapping[str, AgentFactory]]

MANAGER_AGENT_ORDER = (
    "evaluate_agent",
    "routing_agent",
    "observe_agent",
    "writer_agent",
)


def normalise_manager_agent_specs(
    agents: ManagerAgentInput,
) -> Dict[str, AgentFactory]:
    """Convert manager agent overrides into a validated mapping."""

    if isinstance(agents, Mapping):
        invalid = set(agents.keys()) - set(MANAGER_AGENT_ORDER)
        if invalid:
            invalid_list = ", ".join(sorted(invalid))
            raise ValueError(
                "Unknown manager agent names provided: " f"{invalid_list}"
            )
        return {name: agents[name] for name in MANAGER_AGENT_ORDER if name in agents}

    if isinstance(agents, Sequence) and not isinstance(agents, (str, bytes)):
        agent_list = list(agents)
        expected_len = len(MANAGER_AGENT_ORDER)
        if len(agent_list) != expected_len:
            raise ValueError(
                "Expected a sequence of four agents in the order "
                "(evaluate, routing, observe, writer)"
            )
        return dict(zip(MANAGER_AGENT_ORDER, agent_list))

    raise TypeError(
        "manager agent overrides must be provided as a mapping or sequence"
    )


class DataScienceConfig(PipelineConfigBase):
    """Typed configuration for the data science pipeline."""

    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_settings: Optional[Dict[str, Any]] = None
    azure_config: Optional[Dict[str, Any]] = None
    aws_config: Optional[Dict[str, Any]] = None
    pipeline: Dict[str, Any] = Field(default_factory=dict)
    agents: Dict[str, Any] = Field(default_factory=dict)
    tool_agents: Dict[str, Any] = Field(default_factory=dict)
    agent_factories: Dict[str, AgentFactory] = Field(default_factory=dict, exclude=True)
    tool_agent_factories: Dict[str, ToolFactory] = Field(default_factory=dict, exclude=True)
    manager_agent_specs: Optional[ManagerAgentInput] = Field(
        default=None, alias="manager_agents", exclude=True
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
    )

    def resolve(self) -> ResolvedPipelineConfig:
        payload = self.model_dump(
            exclude={"agent_factories", "tool_agent_factories"},
            exclude_none=True,
        )

        attachments: Dict[str, Any] = {}

        manager_overrides: Dict[str, AgentFactory] = {}
        if self.manager_agent_specs is not None:
            manager_overrides.update(
                normalise_manager_agent_specs(self.manager_agent_specs)
            )
        if self.agent_factories:
            manager_overrides.update(self.agent_factories)
        if manager_overrides:
            attachments["manager_agents"] = dict(manager_overrides)
        if self.tool_agent_factories:
            attachments["tool_agents"] = dict(self.tool_agent_factories)

        return ResolvedPipelineConfig(
            config_dict=payload,
            config_object=self,
            attachments=attachments,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataScienceConfig":
        return cls.from_mapping(data)

    @classmethod
    def from_path(cls, path: str) -> "DataScienceConfig":
        return cls.from_file(path)

    def with_manager_agents(
        self,
        agents: Optional[ManagerAgentInput] = None,
        **named_agents: AgentFactory,
    ) -> "DataScienceConfig":
        new_config = self.model_copy(deep=True)
        overrides: Dict[str, AgentFactory] = {}
        if agents is not None:
            overrides.update(normalise_manager_agent_specs(agents))
        if named_agents:
            overrides.update(named_agents)
        new_config.agent_factories.update(overrides)
        return new_config

    def with_tool_agents(self, **agents: ToolFactory) -> "DataScienceConfig":
        new_config = self.model_copy(deep=True)
        new_config.tool_agent_factories.update(agents)
        return new_config


def resolve_data_science_config(source: PipelineConfigSource) -> ResolvedPipelineConfig:
    """Resolve any accepted input into a data science pipeline configuration."""

    return resolve_config_source(source, config_cls=DataScienceConfig)


def _clone_with_model(agent: Agent, config: LLMConfig) -> Agent:  # type: ignore[override]
    """Return a copy of the agent bound to the current pipeline model."""

    cloned = agent.clone() if hasattr(agent, "clone") else agent
    if getattr(cloned, "model", None) is None:
        cloned.model = config.main_model  # type: ignore[attr-defined]
    return cloned


def instantiate_agent_spec(spec: AgentFactory, config: LLMConfig) -> Agent:
    """Instantiate a manager agent from the provided specification."""

    if callable(spec):  # type: ignore[arg-type]
        return spec(config)  # type: ignore[misc]
    if isinstance(spec, Agent):  # type: ignore[arg-type]
        return _clone_with_model(spec, config)
    return spec  # type: ignore[return-value]


def instantiate_tool_agent_spec(spec: ToolFactory, config: LLMConfig) -> Any:
    """Instantiate a tool agent from the provided specification."""

    if callable(spec):  # type: ignore[arg-type]
        return spec(config)  # type: ignore[misc]
    return spec


__all__ = [
    "AgentFactory",
    "DataScienceConfig",
    "ManagerAgentInput",
    "MANAGER_AGENT_ORDER",
    "instantiate_agent_spec",
    "instantiate_tool_agent_spec",
    "normalise_manager_agent_specs",
    "resolve_data_science_config",
]
