"""Shared infrastructure for pipeline configuration objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union, List

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from agentz.utils import load_config


class BaseConfig(BaseModel):
    """Base class for strongly-typed pipeline configuration objects."""

    # Provider/LLM configuration
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_settings: Optional[Dict[str, Any]] = None
    azure_config: Optional[Dict[str, Any]] = None
    aws_config: Optional[Dict[str, Any]] = None

    # Data configuration
    data: Dict[str, Any] = Field(default_factory=dict)
    user_prompt: Optional[str] = None

    # Pipeline settings
    pipeline: Dict[str, Any] = Field(default_factory=dict)

    # Agents: may be flat or single-level groups; leaves will be runtime Agent instances
    agents: Dict[str, Any] = Field(default_factory=dict)

    # Runtime/computed fields (excluded from serialization)
    llm: Any = Field(default=None, exclude=True)
    config_file: Optional[str] = Field(default=None, exclude=True)
    # runtime conveniences for agents
    agents_flat: Dict[str, Any] = Field(default_factory=dict, exclude=True)   # name -> Agent
    agent_groups: Dict[str, List[str]] = Field(default_factory=dict, exclude=True)
    agents_index: Dict[str, Dict[str, Any]] = Field(default_factory=dict, exclude=True)  # name -> {instructions, params}

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        extra="allow",
    )

    @property
    def data_path(self) -> Optional[str]:
        """Get data path from data section."""
        return self.data.get("path")

    @property
    def prompt(self) -> Optional[str]:
        """Get prompt from user_prompt or data section."""
        return self.user_prompt or self.data.get("prompt")

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain serialisable dictionary of the configuration."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BaseConfig":
        """Instantiate the config object from a mapping."""
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "BaseConfig":
        """Instantiate the config object from a YAML or JSON file."""
        data = load_mapping_from_path(path)
        config = cls.from_dict(data)
        config.config_file = str(path)
        return config


def load_mapping_from_path(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a mapping from YAML or JSON file, supporting env substitution."""
    data = load_config(Path(path))
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Configuration file must define a mapping, got {type(data)!r}")
    return dict(data)


def get_api_key_from_env(provider: str) -> str:
    """Auto-load API key from environment based on provider."""
    env_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
    }

    env_var = env_map.get(provider)
    if not env_var:
        raise ValueError(f"Unknown provider: {provider}. Cannot auto-load API key.")

    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"API key not found. Set {env_var} in environment or .env file."
        )

    return api_key


############################
# New helpers (inline)    #
############################

# We will not create new files; helpers live here.
try:
    from agents import Agent  # runtime Agent class must exist
except Exception as e:
    raise ImportError("Expected `from agents import Agent` to be importable") from e

def _is_agent_instance(obj: Any) -> bool:
    return hasattr(obj, "name") and hasattr(obj, "instructions")

def _make_agent_from_mapping(m: Mapping[str, Any]) -> Agent:
    if not isinstance(m, Mapping):
        raise TypeError(f"Agent spec must be a mapping, got {type(m)}")
    name = m.get("name")
    instr = m.get("instructions")
    if not name or not instr:
        raise ValueError("Agent spec must include 'name' and 'instructions'")
    kwargs: Dict[str, Any] = {k: v for k, v in m.items() if k not in {"name", "instructions"}}
    return Agent(name=name, instructions=instr, **kwargs)

def _coerce_agent_like(value: Any) -> Agent:
    if _is_agent_instance(value):
        return value
    if isinstance(value, Mapping):
        return _make_agent_from_mapping(value)
    raise TypeError("Agent must be an Agent instance or mapping with 'name' and 'instructions'")

def _normalize_top_level_keys(d: Mapping[str, Any]) -> Dict[str, Any]:
    """
    - Accept synonyms: data_path -> data.path; prompt -> user_prompt
    - Migrate legacy top-level 'manager_agents' into agents['manager'] unless collision.
    """
    out: Dict[str, Any] = dict(d)
    if "data_path" in out:
        out.setdefault("data", {})
        out["data"] = dict(out["data"], path=out.pop("data_path"))
    if "prompt" in out and "user_prompt" not in out:
        out["user_prompt"] = out.pop("prompt")
    if "manager_agents" in out:
        mgr = out.pop("manager_agents")
        out.setdefault("agents", {})
        if "manager" in out["agents"] and isinstance(out["agents"]["manager"], Mapping):
            out["agents"]["manager"] = {**mgr, **out["agents"]["manager"]}
        elif "manager" not in out["agents"]:
            out["agents"]["manager"] = mgr
        else:
            out["agents"]["manager_migrated"] = mgr
    return out

def _normalize_agents_tree(agents_node: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[str]]]:
    """
    First-level values can be:
      - Agent-like leaf (Agent or mapping with name/instructions)
      - Group dict: { agent_name: Agent-like }
      - Legacy config dict (pass through as-is for registry system)
    Returns:
      agents_tree: same shape, but leaves are runtime Agent instances (or legacy dicts)
      agents_flat: { agent_name: Agent }
      agent_groups: { group_name: [agent_name, ...] }, root-level leaves under "_root"
    Only one level of grouping is allowed; deeper nesting raises ValueError.
    """
    agents_tree: Dict[str, Any] = {}
    agents_flat: Dict[str, Any] = {}
    agent_groups: Dict[str, List[str]] = {}

    for k, v in (agents_node or {}).items():
        # root-level leaf (runtime Agent)
        if _is_agent_instance(v) or (isinstance(v, Mapping) and "name" in v and "instructions" in v):
            agent = _coerce_agent_like(v)
            agents_tree[k] = agent
            if k in agents_flat and agents_flat[k] is not agent:
                raise ValueError(f"Duplicate agent name detected: {k}")
            agents_flat[k] = agent
            agent_groups.setdefault("_root", []).append(k)
            continue

        # group (check if it's a runtime agent group or legacy config)
        if isinstance(v, Mapping):
            group_name = k
            # Check if this is a runtime agent group (all values are Agent-like)
            is_runtime_group = any(_is_agent_instance(av) or (isinstance(av, Mapping) and "name" in av and "instructions" in av) for av in v.values())

            if is_runtime_group:
                group_dict: Dict[str, Any] = {}
                names: List[str] = []
                for ak, av in v.items():
                    # disallow deeper nesting (i.e., group inside group)
                    if isinstance(av, Mapping) and not ("name" in av and "instructions" in av) and any(isinstance(x, Mapping) for x in av.values()):
                        raise ValueError(f"Agents group '{group_name}' contains nested groups; only one level is allowed.")
                    agent = _coerce_agent_like(av)
                    if ak in agents_flat and agents_flat[ak] is not agent:
                        raise ValueError(f"Duplicate agent name across groups: {ak}")
                    group_dict[ak] = agent
                    agents_flat[ak] = agent
                    names.append(ak)
                agents_tree[group_name] = group_dict
                agent_groups[group_name] = names
            else:
                # Legacy config dict - pass through as-is
                agents_tree[group_name] = dict(v)
            continue

        # Pass through other types as-is (for backward compatibility)
        agents_tree[k] = v

    return agents_tree, agents_flat, agent_groups

def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = v
    return out


def normalize_agents(agents_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Normalize agent config into flat index: {name: {instructions, params}}.

    Args:
        agents_config: Raw agents section from config

    Returns:
        Dict mapping agent name to {instructions: str, params: dict}
    """
    index: Dict[str, Dict[str, Any]] = {}

    def _walk(node: Dict[str, Any], parent_path: str = "") -> None:
        for key, value in node.items():
            # Check if this is an agent spec
            if isinstance(value, str):
                # String -> instructions
                index[key] = {"instructions": value, "params": {}}
            elif isinstance(value, dict):
                # Check if it has instructions (leaf spec)
                if "instructions" in value:
                    index[key] = {
                        "instructions": value["instructions"],
                        "params": value.get("params", {})
                    }
                elif _is_agent_instance(value):
                    # Runtime Agent instance - skip (handled elsewhere)
                    pass
                else:
                    # It's a group - recurse
                    _walk(value, key)

    _walk(agents_config)
    return index


def get_agent_spec(cfg: BaseConfig, name: str, required: bool = True) -> Optional[Dict[str, Any]]:
    """Get normalized agent spec from config.

    Args:
        cfg: BaseConfig instance
        name: Agent name
        required: If True, raise error if not found; if False, return None

    Returns:
        Dict with {instructions: str, params: dict} or None if not found and not required

    Raises:
        ValueError: If agent not found or missing instructions and required=True
    """
    idx = getattr(cfg, "agents_index", None)
    if not idx or name not in idx:
        if required:
            available = list(idx.keys()) if idx else []
            raise ValueError(
                f"Agent '{name}' not found in config.agents. "
                f"Available agents: {available}. "
                f"Please add agent definition to your config YAML file."
            )
        return None

    spec = idx[name]
    if not spec.get("instructions"):
        if required:
            raise ValueError(f"Missing instructions for agent '{name}' in config.agents")
        return None

    return spec


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in strings (${VAR} format)."""
    if isinstance(obj, str):
        # Expand ${VAR} patterns
        import re
        def replace_env(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        return re.sub(r'\$\{([^}]+)\}', replace_env, obj)
    elif isinstance(obj, Mapping):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_expand_env_vars(item) for item in obj)
    return obj


def _resolve_relative_paths(data: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Resolve relative paths in data.path relative to base_dir."""
    out = dict(data)
    if "data" in out and isinstance(out["data"], Mapping):
        data_section = dict(out["data"])
        if "path" in data_section and isinstance(data_section["path"], str):
            path_str = data_section["path"]
            # Only resolve if it looks like a relative path
            if not Path(path_str).is_absolute() and not path_str.startswith("http"):
                resolved = (base_dir / path_str).resolve()
                data_section["path"] = str(resolved)
        out["data"] = data_section
    return out


def resolve_config(spec: Union[str, Path, Mapping[str, Any], BaseConfig]) -> BaseConfig:
    """Resolve configuration from various input formats.

    Args:
        spec: Configuration specification:
            - str/Path: Load YAML/JSON file
            - dict with 'config_path': Load file, then deep-merge dict on top
            - dict without 'config_path': Use as-is
            - BaseConfig: Use as-is

    Returns:
        BaseConfig instance with all fields resolved and validated.

    Examples:
        # Load from file
        config = resolve_config("config.yaml")

        # Dict without config_path
        config = resolve_config({"provider": "openai", "data": {"path": "data.csv"}})

        # Dict with config_path (patches file)
        config = resolve_config({
            "config_path": "base.yaml",
            "data": {"path": "override.csv"},
            "agents": {"manager_agents": {"custom_agent": Agent(...)}}
        })

        # BaseConfig instance
        config = resolve_config(BaseConfig(provider="openai", ...))
    """
    # Load environment variables
    load_dotenv()

    base_dir: Optional[Path] = None
    raw: Dict[str, Any] = {}

    # 1. Handle different spec types
    if isinstance(spec, BaseConfig):
        # Already a BaseConfig, convert to dict
        raw = spec.to_dict()
        if spec.config_file:
            base_dir = Path(spec.config_file).parent

    elif isinstance(spec, (str, Path)):
        # Simple file path
        path = Path(spec)
        raw = load_mapping_from_path(path)
        base_dir = path.parent

    elif isinstance(spec, Mapping):
        # Dictionary input
        spec_dict = dict(spec)

        # Check for config_path key
        config_path = spec_dict.pop("config_path", None)

        if config_path:
            # Load base file first
            path = Path(config_path)
            base_data = load_mapping_from_path(path)
            base_dir = path.parent
            # Deep merge spec on top of base
            raw = _deep_merge(base_data, spec_dict)
        else:
            # No config_path, use dict as-is
            raw = spec_dict

    else:
        raise TypeError(
            f"spec must be str, Path, Mapping, or BaseConfig; got {type(spec).__name__}"
        )

    # 2. Expand environment variables in all string fields
    raw = _expand_env_vars(raw)

    # 3. Resolve relative paths if we have a base directory
    if base_dir:
        raw = _resolve_relative_paths(raw, base_dir)

    # 4. Normalize synonyms & migrate legacy keys
    raw = _normalize_top_level_keys(raw)

    # 5. Ensure required sections exist
    raw.setdefault("data", {})
    raw.setdefault("pipeline", {})
    raw.setdefault("agents", {})

    # 6. Build BaseConfig instance
    try:
        config = BaseConfig.from_dict(raw)
    except Exception as e:
        raise ValueError(f"Failed to validate config: {e}") from e

    # 7. Normalize agents tree
    try:
        agents_tree, agents_flat, agent_groups = _normalize_agents_tree(config.agents or {})
        config.agents = agents_tree
        config.agents_flat = agents_flat
        config.agent_groups = agent_groups

        # Build agents index for easy spec lookup
        config.agents_index = normalize_agents(config.agents or {})
    except Exception as e:
        raise ValueError(f"Failed to process agents configuration: {e}") from e

    # 8. Resolve API key from environment if not provided
    if not config.api_key and hasattr(config, 'provider') and config.provider:
        try:
            config.api_key = get_api_key_from_env(config.provider)
        except ValueError:
            pass  # API key will be validated later if needed

    # 9. Build LLM config
    from agentz.llm.llm_setup import LLMConfig

    llm_config_dict: Dict[str, Any] = {
        "provider": config.provider,
        "api_key": config.api_key,
    }
    for optional_key in (
        "model",
        "base_url",
        "model_settings",
        "azure_config",
        "aws_config",
    ):
        value = getattr(config, optional_key, None)
        if value is not None:
            llm_config_dict[optional_key] = value

    config.llm = LLMConfig(llm_config_dict, config.to_dict())

    # Store config file path if we have it
    if base_dir and not config.config_file:
        config.config_file = str(base_dir / "config")

    return config


def load_pipeline_config(
    source: Union[BaseConfig, Mapping[str, Any], str, Path],
    *,
    overrides: Optional[Mapping[str, Any]] = None,
) -> BaseConfig:
    """Load and process pipeline configuration (legacy API).

    Args:
        source: Config input (BaseConfig, dict, or file path).
        overrides: Optional dict to deep merge into config.

    Returns:
        BaseConfig instance with llm field populated.

    Note:
        This is the legacy API. Prefer resolve_config() for new code.
        For the new API, use config_path in the dict instead of overrides.
    """
    # Load environment variables
    load_dotenv()

    # Import here to avoid circular dependency
    from agentz.llm.llm_setup import LLMConfig

    # Ingest anything → mapping
    if isinstance(source, BaseConfig):
        raw: Dict[str, Any] = source.to_dict()
    elif isinstance(source, Mapping):
        raw = dict(source)
    elif isinstance(source, (str, Path)):
        raw = load_mapping_from_path(source)
    else:
        raise TypeError(f"Unsupported config type: {type(source)}.")

    # Normalize synonyms & migrate legacy keys
    raw = _normalize_top_level_keys(raw)

    # Ensure sections
    raw.setdefault("data", {})
    raw.setdefault("pipeline", {})
    raw.setdefault("agents", {})

    # Apply overrides last
    if overrides:
        raw = _deep_merge(raw, overrides)

    # Build initial BaseConfig
    config = BaseConfig.from_dict(raw)

    # Normalize agents: coerce to runtime Agent, support one-level groups
    agents_tree, agents_flat, agent_groups = _normalize_agents_tree(config.agents or {})
    config.agents = agents_tree
    config.agents_flat = agents_flat
    config.agent_groups = agent_groups

    # Resolve API key from environment if not provided
    if not config.api_key:
        config.api_key = get_api_key_from_env(config.provider)

    # Build LLM config dict
    llm_config_dict: Dict[str, Any] = {
        "provider": config.provider,
        "api_key": config.api_key,
    }
    for optional_key in (
        "model",
        "base_url",
        "model_settings",
        "azure_config",
        "aws_config",
    ):
        value = getattr(config, optional_key, None)
        if value is not None:
            llm_config_dict[optional_key] = value

    # Create LLM config instance
    config.llm = LLMConfig(llm_config_dict, config.to_dict())

    return config


__all__ = [
    "BaseConfig",
    "load_mapping_from_path",
    "get_api_key_from_env",
    "resolve_config",
    "load_pipeline_config",
]
