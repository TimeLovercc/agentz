import os
from typing import Optional
from dotenv import load_dotenv
from ds1.src.llm.llm_setup import LLMConfig


class BasePipeline:
    """Base class for all pipelines with common configuration and setup."""

    def __init__(
        self,
        data_path: str,
        user_prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_tracing: bool = True,
        trace_include_sensitive_data: bool = False,
        config_dict: Optional[dict] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Initialize base pipeline with automatic environment loading.

        Args:
            data_path: Path to the dataset file
            user_prompt: User's description of the task
            provider: LLM provider name (openai, gemini, deepseek, etc.)
            model: Model name (optional, uses provider defaults)
            api_key: API key (optional, auto-loads from env)
            base_url: Custom base URL (optional)
            enable_tracing: Whether to enable tracing
            trace_include_sensitive_data: Whether to include sensitive data in traces
            config_dict: Pre-built config dictionary (alternative to individual params)
            llm_config: Pre-created LLM configuration (alternative to config_dict)
        """
        # Load environment variables
        load_dotenv()

        self.data_path = data_path
        self.user_prompt = user_prompt
        self.enable_tracing = enable_tracing
        self.trace_include_sensitive_data = trace_include_sensitive_data

        # Build config from parameters or use provided config_dict
        if llm_config:
            self.config = llm_config
            self.config_dict = llm_config.config
        elif config_dict:
            self.config_dict = config_dict
            self.config = LLMConfig(config_dict)
        else:
            # Auto-build config from parameters
            if not provider:
                raise ValueError("Either provide 'provider' or 'config_dict'")

            # Auto-load API key from environment if not provided
            if not api_key:
                api_key = self._get_api_key_from_env(provider)

            self.config_dict = {
                "provider": provider,
                "api_key": api_key,
            }
            if model:
                self.config_dict["model"] = model
            if base_url:
                self.config_dict["base_url"] = base_url

            self.config = LLMConfig(self.config_dict)

        # Tracing is automatically set up in LLMConfig for OpenAI provider

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
