import asyncio
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Union

from dotenv import load_dotenv
from rich.console import Console

from agents.tracing.create import trace
from agentz.configuration.base import BaseConfig, normalize_pipeline_config
from agentz.llm.llm_setup import LLMConfig
from agentz.utils import Printer
from pydantic import BaseModel, Field


class BasePipeline:
    """Base class for all pipelines with common configuration and setup."""

    def __init__(
        self, config: Optional[Union[BaseConfig, Mapping[str, Any], str, Path]] = None
    ):
        """Initialise the pipeline using a single configuration input."""
        load_dotenv()

        self.console = Console()
        self._printer: Optional[Printer] = None

        config_schema = getattr(self, "CONFIG_SCHEMA", None)
        default_config_path = getattr(self, "DEFAULT_CONFIG_PATH", None)

        # Normalize configuration
        self.full_config, source_path = normalize_pipeline_config(
            config,
            config_cls=config_schema,
            default_path=default_config_path,
        )

        # Determine config file path
        if source_path is not None:
            self.config_file = source_path
        elif isinstance(config, (str, Path)):
            self.config_file = str(config)
        elif default_config_path is not None:
            self.config_file = str(default_config_path)
        else:
            self.config_file = "<inline>"

        # Extract pipeline settings
        pipeline_section = self.full_config.get("pipeline")
        if not isinstance(pipeline_section, Mapping):
            pipeline_section = {}

        self.enable_tracing = bool(pipeline_section.get("enable_tracing", True))
        self.trace_include_sensitive_data = bool(
            pipeline_section.get("trace_include_sensitive_data", False)
        )

        # Extract provider and API key
        provider = self.full_config.get("provider")
        if not provider:
            raise ValueError("Configuration missing required field 'provider'")

        api_key = self.full_config.get("api_key")
        if not api_key:
            api_key = self._get_api_key_from_env(provider)

        if not api_key:
            raise ValueError("Unable to determine API key from config or environment")

        # Build LLM config dict
        self.config_dict: Dict[str, Any] = {
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
            if optional_key in self.full_config:
                self.config_dict[optional_key] = self.full_config[optional_key]

        # Create LLM config
        self.config = LLMConfig(self.config_dict, self.full_config)

        # Extract data path and user prompt
        data_section = self.full_config.get("data")
        if not isinstance(data_section, MutableMapping):
            data_section = {}
            self.full_config["data"] = data_section

        self.data_path = data_section.get("path")
        self.user_prompt = self.full_config.get("user_prompt") or data_section.get("prompt")

        # Validate required fields
        require_data_path = bool(getattr(self, "REQUIRE_DATA_PATH", False))
        require_user_prompt = bool(getattr(self, "REQUIRE_USER_PROMPT", False))

        if require_data_path and not self.data_path:
            raise ValueError(
                "This pipeline requires 'data_path' in the configuration or as an argument"
            )
        if require_user_prompt and not self.user_prompt:
            raise ValueError(
                "This pipeline requires 'user_prompt' in the configuration or as an argument"
            )

    @property
    def provider_name(self) -> str:
        return (self.config_dict or {}).get("provider", "unknown provider")

    @property
    def printer(self) -> Optional[Printer]:
        return self._printer

    @property
    def pipeline_settings(self) -> Dict[str, Any]:
        settings = self.full_config.get("pipeline")
        return settings if isinstance(settings, dict) else {}

    def start_printer(self) -> Printer:
        if self._printer is None:
            self._printer = Printer(self.console)
        return self._printer

    def stop_printer(self) -> None:
        if self._printer is not None:
            self._printer.end()
            self._printer = None

    def trace_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        if self.enable_tracing:
            return trace(name, metadata=metadata)
        return nullcontext()

    def span_context(self, span_factory, **kwargs):
        if self.enable_tracing:
            return span_factory(**kwargs)
        return nullcontext()

    def _get_api_key_from_env(self, provider: str) -> str:
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

    def run_sync(self):
        """Synchronous wrapper for the async run method."""
        return asyncio.run(self.run())

    async def run(self):  # pragma: no cover - must be implemented by subclasses
        raise NotImplementedError("Subclasses must implement 'run'")


class IterationData(BaseModel):
    """Structured information collected during a single research loop iteration."""

    gap: str = Field(description="The gap addressed in the iteration", default="")
    tool_calls: List[str] = Field(description="The tool calls made", default_factory=list)
    findings: List[str] = Field(
        description="The findings collected from tool calls", default_factory=list
    )
    thought: str = Field(
        description="Reflection on iteration success and next steps", default=""
    )


class Conversation(BaseModel):
    """Accumulates thoughts, actions, and findings across research iterations."""

    history: List[IterationData] = Field(
        description="The data for each iteration of the research loop", default_factory=list
    )

    def add_iteration(self, iteration_data: Optional[IterationData] = None) -> None:
        if iteration_data is None:
            iteration_data = IterationData()
        self.history.append(iteration_data)

    def set_latest_gap(self, gap: str) -> None:
        if self.history:
            self.history[-1].gap = gap

    def set_latest_tool_calls(self, tool_calls: List[str]) -> None:
        if self.history:
            self.history[-1].tool_calls = tool_calls

    def set_latest_findings(self, findings: List[str]) -> None:
        if self.history:
            self.history[-1].findings = findings

    def set_latest_thought(self, thought: str) -> None:
        if self.history:
            self.history[-1].thought = thought

    def get_all_findings(self) -> List[str]:
        return [
            finding
            for iteration_data in self.history
            for finding in iteration_data.findings
        ]

    def compile_conversation_history(self) -> str:
        """Compile the conversation history into a string."""

        conversation = ""
        for iteration_num, iteration_data in enumerate(self.history):
            conversation += f"[ITERATION {iteration_num + 1}]\n\n"
            if iteration_data.thought:
                conversation += f"<thought>\n{iteration_data.thought}\n</thought>\n\n"
            if iteration_data.gap:
                conversation += (
                    f"<task>\nAddress this knowledge gap: {iteration_data.gap}\n</task>\n\n"
                )
            if iteration_data.tool_calls:
                joined_calls = "\n".join(iteration_data.tool_calls)
                conversation += (
                    "<action>\nCalling the following tools to address the knowledge gap:\n"
                    f"{joined_calls}\n</action>\n\n"
                )
            if iteration_data.findings:
                joined_findings = "\n\n".join(iteration_data.findings)
                conversation += f"<findings>\n{joined_findings}\n</findings>\n\n"

        return conversation
