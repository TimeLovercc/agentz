"""Typed configuration support for the data science pipeline."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Union

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
        if self.agent_factories:
            attachments["manager_agents"] = dict(self.agent_factories)
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

    def with_manager_agents(self, **agents: AgentFactory) -> "DataScienceConfig":
        new_config = self.model_copy(deep=True)
        new_config.agent_factories.update(agents)
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
    "instantiate_agent_spec",
    "instantiate_tool_agent_spec",
    "resolve_data_science_config",
]
