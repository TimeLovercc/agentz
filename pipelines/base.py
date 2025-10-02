import asyncio
import os
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from rich.console import Console

from agents.tracing.create import trace
from agentz.llm.llm_setup import LLMConfig
from agentz.utils import Printer, load_config
from pydantic import BaseModel, Field


class BasePipeline:
    """Base class for all pipelines with common configuration and setup."""

    def __init__(
        self,
        *,
        config_file: str,
        data_path: Optional[str] = None,
        user_prompt: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        enable_tracing: bool = True,
        trace_include_sensitive_data: bool = False,
    ):
        """
        Initialize base pipeline with automatic environment loading.

        Args:
            config_file: Path to the configuration file (YAML/JSON).
            data_path: Optional dataset path for this run; overrides config value.
            user_prompt: Optional user prompt; overrides config value.
            overrides: Optional dictionary merged into the loaded config for
                ad-hoc tweaks (e.g., adjusting provider/model at runtime).
            enable_tracing: Whether to enable tracing support.
            trace_include_sensitive_data: Whether to include sensitive data in traces.
        """
        # Load environment variables
        load_dotenv()

        self.enable_tracing = enable_tracing
        self.trace_include_sensitive_data = trace_include_sensitive_data
        self.console = Console()
        self._printer: Optional[Printer] = None

        # Load full configuration and apply overrides
        if not config_file:
            raise ValueError("'config_file' is required to initialize a pipeline")

        raw_config = load_config(config_file)
        config_data: Dict[str, Any] = deepcopy(raw_config)
        if overrides:
            config_data = _deep_merge_dicts(config_data, overrides)

        self.config_file = config_file
        self.full_config = config_data

        # Build LLM configuration from merged settings
        provider = self.full_config.get('provider')
        if not provider:
            raise ValueError("Configuration missing required field 'provider'")

        api_key = self.full_config.get('api_key')
        if not api_key:
            api_key = self._get_api_key_from_env(provider)

        if not api_key:
            raise ValueError("Unable to determine API key from config or environment")

        config_dict: Dict[str, Any] = {
            "provider": provider,
            "api_key": api_key,
        }
        for optional_key in ("model", "base_url", "model_settings", "azure_config", "aws_config"):
            if optional_key in self.full_config:
                config_dict[optional_key] = self.full_config[optional_key]

        self.config_dict = config_dict
        self.config = LLMConfig(self.config_dict, self.full_config)

        # Persist frequently used fields
        existing_data_section = self.full_config.get('data')
        data_section = existing_data_section if isinstance(existing_data_section, dict) else {}
        self.data_path = data_path or data_section.get('path')
        self.user_prompt = user_prompt or self.full_config.get('user_prompt') or data_section.get('prompt')

        if self.data_path:
            if not isinstance(self.full_config.get('data'), dict):
                self.full_config['data'] = {}
            self.full_config['data']['path'] = self.data_path
        if self.user_prompt:
            self.full_config['user_prompt'] = self.user_prompt

    @property
    def provider_name(self) -> str:
        return (self.config_dict or {}).get('provider', 'unknown provider')

    @property
    def printer(self) -> Optional[Printer]:
        return self._printer

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
            raise ValueError(f"API key not found. Set {env_var} in environment or .env file.")

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
        return [finding for iteration_data in self.history for finding in iteration_data.findings]

    def compile_conversation_history(self) -> str:
        """Compile the conversation history into a string."""

        conversation = ""
        for iteration_num, iteration_data in enumerate(self.history):
            conversation += f"[ITERATION {iteration_num + 1}]\n\n"
            if iteration_data.thought:
                conversation += f"<thought>\n{iteration_data.thought}\n</thought>\n\n"
            if iteration_data.gap:
                conversation += f"<task>\nAddress this knowledge gap: {iteration_data.gap}\n</task>\n\n"
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


def _deep_merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating the originals."""

    merged = deepcopy(base)
    for key, value in updates.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
