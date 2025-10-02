"""Shared infrastructure for pipeline configuration objects."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from agentz.utils import load_config

PipelineConfigSource = Union["BaseConfig", Mapping[str, Any], str, Path]
PipelineConfigInput = Optional[Union[PipelineConfigSource, "ResolvedPipelineConfig"]]
ConfigResolver = Callable[[PipelineConfigSource], "ResolvedPipelineConfig"]


@dataclass
class ResolvedPipelineConfig:
    """Concrete configuration payload handed to pipelines."""

    config_dict: Dict[str, Any]
    config_object: Optional["BaseConfig"] = None
    attachments: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    def copy(self) -> "ResolvedPipelineConfig":
        return ResolvedPipelineConfig(
            config_dict=deepcopy(self.config_dict),
            config_object=self.config_object,
            attachments=deepcopy(self.attachments),
            source_path=self.source_path,
        )


@dataclass
class PreparedPipelineConfig:
    """Resolved pipeline configuration and metadata ready for instantiation."""

    resolved: ResolvedPipelineConfig
    full_config: Dict[str, Any]
    config_file: str


@dataclass
class PipelineRuntimeState:
    """Runtime artefacts produced from a prepared configuration."""

    resolved: ResolvedPipelineConfig
    full_config: Dict[str, Any]
    config_dict: Dict[str, Any]
    llm_config: Any
    data_path: Optional[str]
    user_prompt: Optional[str]
    enable_tracing: bool
    trace_include_sensitive_data: bool


def _deep_merge_dicts(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two mappings without mutating the originals."""

    merged: Dict[str, Any] = {key: deepcopy(value) for key, value in base.items()}
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_path(
    config_path: Union[str, Path],
    *,
    base_source: Optional[str] = None,
) -> Path:
    """Resolve ``config_path`` against an optional base configuration file."""

    candidate = Path(config_path)
    if not candidate.is_absolute() and base_source:
        try:
            base_dir = Path(base_source).expanduser().parent
        except Exception:
            base_dir = None
        if base_dir is not None:
            candidate = base_dir / candidate
    candidate = candidate.expanduser()
    return candidate


class BaseConfig(BaseModel):
    """Base class for strongly-typed pipeline configuration objects."""

    config_path: Optional[Path] = Field(default=None, exclude=True)
    source_path: Optional[str] = Field(default=None, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
    )

    def _resolve_external_source(self) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        if self.config_path is None:
            return None, self.source_path

        candidate = _resolve_config_path(self.config_path, base_source=self.source_path)

        data = load_mapping_from_path(candidate)
        resolved_path = str(candidate)
        # Persist resolved metadata for downstream access
        self.config_path = candidate
        self.source_path = resolved_path
        return data, resolved_path

    def _merge_with_external_payload(
        self, payload: Mapping[str, Any]
    ) -> tuple[Dict[str, Any], Optional[str]]:
        inline_payload: Dict[str, Any] = {
            key: deepcopy(value) for key, value in payload.items()
        }
        external_data, external_source = self._resolve_external_source()
        if external_data is not None:
            merged_payload = _deep_merge_dicts(external_data, inline_payload)
        else:
            merged_payload = inline_payload

        source = external_source or self.source_path
        if source is not None:
            self.source_path = source
        return merged_payload, source

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain serialisable dictionary of the configuration."""

        return self.model_dump(exclude_none=True)

    def resolve(self) -> ResolvedPipelineConfig:
        """Produce the resolved configuration payload for pipeline consumption."""

        payload, source = self._merge_with_external_payload(self.model_dump(exclude_none=True))
        return ResolvedPipelineConfig(
            config_dict=payload,
            config_object=self,
            source_path=source,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BaseConfig":
        """Instantiate the config object from a mapping."""

        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BaseConfig":
        """Instantiate the config object from a YAML or JSON file."""

        data = load_mapping_from_path(path)
        config = cls.from_mapping(data)
        config.source_path = str(Path(path))
        return config


def load_mapping_from_path(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a mapping from YAML or JSON file, supporting env substitution."""

    data = load_config(Path(path))
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Configuration file must define a mapping, got {type(data)!r}")
    return dict(data)


def resolve_config_source(
    source: PipelineConfigSource,
    *,
    config_cls: Optional[type[BaseConfig]] = None,
) -> ResolvedPipelineConfig:
    """Normalise multiple configuration entry-points into a single payload."""

    if isinstance(source, BaseConfig):
        if config_cls and not isinstance(source, config_cls):
            source = config_cls.model_validate(source.model_dump())
        resolved = source.resolve()
        if resolved.source_path is None and hasattr(source, "source_path"):
            resolved.source_path = getattr(source, "source_path")
        return resolved

    if isinstance(source, Mapping):
        if config_cls is None:
            raw_mapping = dict(source)
            config_path_value = raw_mapping.pop("config_path", None)
            source_path_value = raw_mapping.pop("source_path", None)

            if config_path_value is not None:
                resolved_path = _resolve_config_path(
                    config_path_value, base_source=source_path_value
                )
                external_data = load_mapping_from_path(resolved_path)
                config_dict = _deep_merge_dicts(external_data, raw_mapping)
                source_path_value = source_path_value or str(resolved_path)
            else:
                config_dict = raw_mapping

            return ResolvedPipelineConfig(
                config_dict=config_dict,
                source_path=source_path_value,
            )
        config_obj = config_cls.from_mapping(source)
        resolved = config_obj.resolve()
        resolved.source_path = getattr(config_obj, "source_path", None)
        return resolved

    if isinstance(source, (str, Path)):
        path = Path(source)
        if config_cls is None:
            data = load_mapping_from_path(path)
            return ResolvedPipelineConfig(config_dict=data, source_path=str(path))
        config_obj = config_cls.from_file(path)
        resolved = config_obj.resolve()
        resolved.source_path = str(path)
        return resolved

    raise TypeError(
        "Unsupported configuration source type. Expected BaseConfig, mapping, or path string."
    )


def normalize_pipeline_config(
    source: PipelineConfigInput,
    *,
    config_cls: Optional[type[BaseConfig]] = None,
    default_path: Optional[Union[str, Path]] = None,
) -> ResolvedPipelineConfig:
    """Normalise unstructured pipeline inputs into a resolved configuration.

    Args:
        source: User-supplied configuration input. Accepts the same variants as
            :data:`PipelineConfigSource`, an already resolved payload, or ``None``.
        config_cls: Optional Pydantic model used to coerce mappings into strongly
            typed objects.
        default_path: Optional fallback invoked when ``source`` is ``None``.

    Returns:
        A :class:`ResolvedPipelineConfig` instance that encapsulates the merged
        configuration dictionary, attachments, and provenance metadata.
    """

    if source is None:
        if default_path is None:
            raise ValueError(
                "A configuration source must be provided or a default_path must be set."
            )
        source = default_path

    if isinstance(source, ResolvedPipelineConfig):
        resolved = source.copy()
        if config_cls and resolved.config_object is not None:
            if not isinstance(resolved.config_object, config_cls):
                coerced = config_cls.from_mapping(resolved.config_dict)
                refreshed = coerced.resolve()
                refreshed.attachments.update(resolved.attachments)
                if resolved.source_path and refreshed.source_path is None:
                    refreshed.source_path = resolved.source_path
                return refreshed
        return resolved

    resolved = resolve_config_source(source, config_cls=config_cls)
    return resolved.copy()


def prepare_pipeline_config(
    config: PipelineConfigInput,
    *,
    config_cls: Optional[type[BaseConfig]] = None,
    default_path: Optional[Union[str, Path]] = None,
) -> PreparedPipelineConfig:
    """Normalise user input and capture metadata for pipeline instantiation."""

    resolved = normalize_pipeline_config(
        config,
        config_cls=config_cls,
        default_path=default_path,
    )

    resolved_clone = resolved.copy()
    full_config = deepcopy(resolved_clone.config_dict)
    resolved_clone.config_dict = full_config

    if resolved_clone.source_path is not None:
        config_file = resolved_clone.source_path
    elif isinstance(config, (str, Path)):
        config_file = str(config)
    elif default_path is not None:
        config_file = str(default_path)
    else:
        config_file = "<inline>"

    resolved_clone.source_path = config_file

    return PreparedPipelineConfig(
        resolved=resolved_clone,
        full_config=full_config,
        config_file=config_file,
    )


def instantiate_pipeline_runtime(
    prepared: PreparedPipelineConfig,
    *,
    llm_config_factory: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    api_key_resolver: Optional[Callable[[str], Optional[str]]] = None,
    require_data_path: bool = False,
    require_user_prompt: bool = False,
) -> PipelineRuntimeState:
    """Instantiate runtime artefacts from a prepared configuration."""

    resolved = prepared.resolved
    full_config = prepared.full_config

    pipeline_section = full_config.get("pipeline")
    if not isinstance(pipeline_section, Mapping):
        pipeline_section = {}

    enable_tracing = bool(pipeline_section.get("enable_tracing", True))
    trace_include_sensitive_data = bool(
        pipeline_section.get("trace_include_sensitive_data", False)
    )

    provider = full_config.get("provider")
    if not provider:
        raise ValueError("Configuration missing required field 'provider'")

    api_key = full_config.get("api_key")
    if not api_key and api_key_resolver is not None:
        api_key = api_key_resolver(provider)

    if not api_key:
        raise ValueError("Unable to determine API key from config or environment")

    config_dict: Dict[str, Any] = {
        "provider": provider,
        "api_key": api_key,
    }
    for optional_key in (
        "model",
        "base_url",
        "model_settings",
        "azure_config",
        "aws_config",
    ):
        if optional_key in full_config:
            config_dict[optional_key] = full_config[optional_key]

    if llm_config_factory is None or not callable(llm_config_factory):
        raise TypeError(
            "A callable llm_config_factory must be provided to instantiate pipeline runtime"
        )

    llm_config = llm_config_factory(config_dict, full_config)

    data_section = full_config.get("data")
    if not isinstance(data_section, MutableMapping):
        data_section = {}
        full_config["data"] = data_section

    data_path = data_section.get("path")
    user_prompt = full_config.get("user_prompt") or data_section.get("prompt")

    if data_path is not None:
        data_section["path"] = data_path
    if user_prompt:
        full_config["user_prompt"] = user_prompt

    if require_data_path and not data_path:
        raise ValueError(
            "This pipeline requires 'data_path' in the configuration or as an argument"
        )
    if require_user_prompt and not user_prompt:
        raise ValueError(
            "This pipeline requires 'user_prompt' in the configuration or as an argument"
        )

    return PipelineRuntimeState(
        resolved=resolved,
        full_config=full_config,
        config_dict=config_dict,
        llm_config=llm_config,
        data_path=data_path,
        user_prompt=user_prompt,
        enable_tracing=enable_tracing,
        trace_include_sensitive_data=trace_include_sensitive_data,
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


class DataScienceConfig(BaseConfig):
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

        payload, source_path = self._merge_with_external_payload(payload)

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
            source_path=source_path,
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
    "ConfigResolver",
    "BaseConfig",
    "PipelineConfigSource",
    "PipelineConfigInput",
    "ResolvedPipelineConfig",
    "PreparedPipelineConfig",
    "PipelineRuntimeState",
    "load_mapping_from_path",
    "normalize_pipeline_config",
    "prepare_pipeline_config",
    "instantiate_pipeline_runtime",
    "resolve_config_source",
    "AgentFactory",
    "DataScienceConfig",
    "MANAGER_AGENT_ORDER",
    "ManagerAgentInput",
    "ToolFactory",
    "instantiate_agent_spec",
    "instantiate_tool_agent_spec",
    "normalise_manager_agent_specs",
    "resolve_data_science_config",
]
