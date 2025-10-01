from typing import Dict, Any
import os

from agents import (
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    ModelSettings,
)
from agents.extensions.models.litellm_model import LitellmModel
from openai import AsyncAzureOpenAI, AsyncOpenAI

# Provider configurations - use OpenAIResponsesModel for most providers
PROVIDER_CONFIGS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model_class": OpenAIResponsesModel,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model_class": OpenAIResponsesModel,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model_class": OpenAIResponsesModel,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_class": OpenAIChatCompletionsModel,
        "use_litellm": False,
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "model_class": OpenAIResponsesModel,
    },
    "perplexity": {
        "base_url": "https://api.perplexity.ai/chat/completions",
        "model_class": OpenAIResponsesModel,
    },
    "huggingface": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_class": OpenAIResponsesModel,
    },
    "local": {
        "base_url": None,  # Will be provided in config
        "model_class": OpenAIChatCompletionsModel,
        "default_api_key": "ollama",
    },
    "azureopenai": {
        "model_class": OpenAIChatCompletionsModel,
        "requires_azure": True,
    },
    "bedrock": {
        "model_class": LitellmModel,
        "use_litellm": True,
    },
}


class LLMConfig:
    """Direct configuration system - no environment variables."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM configuration from direct config.

        Args:
            config: Dictionary containing:
                - provider: str (e.g., "openai", "gemini", "deepseek")
                - api_key: str
                - model: str (optional, will use defaults)
                - base_url: str (optional for custom endpoints)
                - azure_config: dict (for Azure OpenAI)
                - aws_config: dict (for Bedrock)
                - model_settings: dict (optional, for temperature etc.)
        """
        self.provider = config["provider"]
        self.api_key = config["api_key"]
        self.model_name = config.get("model", self._get_default_model())
        self.config = config

        # Validate provider
        if self.provider not in PROVIDER_CONFIGS:
            valid = list(PROVIDER_CONFIGS.keys())
            raise ValueError(f"Invalid provider: {self.provider}. Available: {valid}")

        # Create main model (used for all purposes - reasoning, main, fast)
        self.main_model = self._create_model()
        self.reasoning_model = self.main_model
        self.fast_model = self.main_model

        # Model settings from config or defaults
        model_settings_config = self.config.get("model_settings", {})
        self.default_model_settings = ModelSettings(
            temperature=model_settings_config.get("temperature", 0.1)
        )

        # Set tracing if OpenAI key provided
        if self.provider == "openai" and self.api_key:
            from agents import set_tracing_export_api_key
            set_tracing_export_api_key(self.api_key)

    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4.1",
            "gemini": "gemini-2.5-flash",
            "deepseek": "deepseek-chat",
            "anthropic": "claude-3-5-sonnet-20241022",
            "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "perplexity": "llama-3.1-sonar-large-128k-online",
            "openrouter": "meta-llama/llama-3.2-3b-instruct:free",
        }
        return defaults.get(self.provider, "gpt-4.1")

    def _create_model(self):
        """Create model instance using direct configuration."""
        provider_config = PROVIDER_CONFIGS[self.provider]
        model_class = provider_config["model_class"]

        if provider_config.get("use_litellm"):
            return model_class(model=self.model_name, api_key=self.api_key, base_url=provider_config.get("base_url"))

        elif self.provider == "azureopenai":
            azure_config = self.config.get("azure_config", {})
            client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=azure_config.get("endpoint"),
                azure_deployment=azure_config.get("deployment"),
                api_version=azure_config.get("api_version", "2023-12-01-preview"),
            )
            return model_class(model=self.model_name, openai_client=client)

        else:
            # Standard OpenAI-compatible providers
            base_url = self.config.get("base_url", provider_config["base_url"])
            api_key = self.api_key or provider_config.get("default_api_key", "key")

            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            return model_class(model=self.model_name, openai_client=client)
