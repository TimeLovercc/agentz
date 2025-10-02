import asyncio
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from rich.console import Console

from agents.tracing.create import trace
from agentz.configuration.base import (
    BaseConfig,
    PipelineConfigInput,
    instantiate_pipeline_runtime,
    prepare_pipeline_config,
)
from agentz.llm.llm_setup import LLMConfig
from agentz.utils import Printer
from pydantic import BaseModel, Field


class BasePipeline:
    """Base class for all pipelines with common configuration and setup."""

    CONFIG_SCHEMA: Optional[type[BaseConfig]] = None
    DEFAULT_CONFIG_PATH: Optional[Union[str, Path]] = None
    REQUIRE_DATA_PATH: bool = False
    REQUIRE_USER_PROMPT: bool = False

    def __init__(self, config: PipelineConfigInput = None):
        """Initialise the pipeline using a single configuration input."""
        load_dotenv()

        self.console = Console()
        self._printer: Optional[Printer] = None

        prepared = prepare_pipeline_config(
            config,
            config_cls=self.CONFIG_SCHEMA,
            default_path=self.DEFAULT_CONFIG_PATH,
        )

        runtime = instantiate_pipeline_runtime(
            prepared,
            llm_config_factory=LLMConfig,
            api_key_resolver=self._get_api_key_from_env,
            require_data_path=self.REQUIRE_DATA_PATH,
            require_user_prompt=self.REQUIRE_USER_PROMPT,
        )

        self._resolved_config = runtime.resolved
        self.full_config = runtime.full_config
        self._resolved_config.config_dict = self.full_config

        self.config_file = prepared.config_file
        self._resolved_config.source_path = prepared.config_file

        self.enable_tracing = runtime.enable_tracing
        self.trace_include_sensitive_data = runtime.trace_include_sensitive_data

        self.config_dict = runtime.config_dict
        self.config = runtime.llm_config

        self.data_path = runtime.data_path
        self.user_prompt = runtime.user_prompt

    @property
    def provider_name(self) -> str:
        return (self.config_dict or {}).get("provider", "unknown provider")

    @property
    def printer(self) -> Optional[Printer]:
        return self._printer

    @property
    def config_object(self):
        return getattr(self._resolved_config, "config_object", None)

    @property
    def config_attachments(self) -> Dict[str, Any]:
        attachments = getattr(self._resolved_config, "attachments", None)
        return attachments if attachments is not None else {}

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
