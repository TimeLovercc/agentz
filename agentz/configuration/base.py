"""Shared infrastructure for pipeline configuration objects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

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

    # Agent configurations (optional, for data scientist pipeline)
    agents: Dict[str, Any] = Field(default_factory=dict)
    tool_agents: Dict[str, Any] = Field(default_factory=dict)
    manager_agents: Dict[str, Any] = Field(default_factory=dict)

    # Runtime/computed fields (excluded from serialization)
    llm: Any = Field(default=None, exclude=True)
    config_file: Optional[str] = Field(default=None, exclude=True)

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


def load_pipeline_config(
    source: Union[BaseConfig, Mapping[str, Any], str, Path]
) -> BaseConfig:
    """Load and process pipeline configuration.

    Args:
        source: Config input (BaseConfig, dict, or file path).

    Returns:
        BaseConfig instance with llm field populated.
    """
    # Load environment variables
    load_dotenv()

    # Import here to avoid circular dependency
    from agentz.llm.llm_setup import LLMConfig

    # Load config based on type
    if isinstance(source, BaseConfig):
        config = source
    elif isinstance(source, Mapping):
        config = BaseConfig.from_dict(source)
    elif isinstance(source, (str, Path)):
        config = BaseConfig.from_file(source)
    else:
        raise TypeError(
            f"Unsupported config type: {type(source)}. Expected BaseConfig, dict, or file path."
        )

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
    "load_pipeline_config",
]
