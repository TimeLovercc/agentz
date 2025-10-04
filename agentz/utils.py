"""
Utility helpers for the multi-agent data science system.
"""

import os
import json
import logging
import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.spinner import Spinner


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("ds_multiagent")
    logger.setLevel(getattr(logging, log_level.upper()))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def load_json_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_json_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def get_experiment_timestamp() -> str:
    """Get timestamp for experiment naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def validate_experiment_config(config: Dict[str, Any]) -> bool:
    """Validate experiment configuration."""
    required_fields = ['name', 'description', 'agents']
    return all(field in config for field in required_fields)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged

def get_env_with_prefix(base_name: str, prefix: str = "DR_", default: Optional[str] = None) -> Optional[str]:
    """
    Retrieves an environment variable, checking for a prefixed version first.

    Args:
        base_name: The base name of the environment variable (e.g., "OPENAI_API_KEY").
        prefix: The prefix to check for (e.g., "DR_"). Defaults to "DR_".
        default: The default value to return if neither the prefixed nor the
                 base variable is found.

    Returns:
        The value of the environment variable, or the default value, or None.
    """
    prefixed_name = f"{prefix}{base_name}"
    value = os.getenv(prefixed_name)
    if value is not None:
        return value
    return os.getenv(base_name, default)


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file with env variable substitution.

    Args:
        config_file: Path to config file (YAML or JSON)

    Returns:
        Dictionary containing the configuration

    Example:
        config = load_config("configs/data_science_gemini.yaml")
    """
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Load file based on extension
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    # Substitute environment variables
    config = _substitute_env_vars(config)

    return config


def _substitute_env_vars(obj: Any) -> Any:
    """
    Recursively substitute ${VAR_NAME} with environment variable values.

    Args:
        obj: Object to process (dict, list, str, etc.)

    Returns:
        Object with environment variables substituted
    """
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, obj)

        result = obj
        for var_name in matches:
            env_value = os.getenv(var_name, '')
            result = result.replace(f'${{{var_name}}}', env_value)

        return result
    else:
        return obj


def get_agent_instructions(config: Dict[str, Any], agent_name: str) -> str:
    """
    Extract agent instructions from config.

    Args:
        config: Configuration dictionary
        agent_name: Name of the agent (e.g., 'evaluate_agent')

    Returns:
        Instructions string for the agent
    """
    return config.get('agents', {}).get(agent_name, {}).get('instructions', '')


def get_pipeline_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract pipeline settings from config.

    Args:
        config: Configuration dictionary

    Returns:
        Pipeline settings dictionary
    """
    return config.get('pipeline', {})


class Printer:
    """Rich-powered status printer for streaming pipeline progress updates."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.live = Live(console=console)
        self.items: Dict[str, Tuple[str, bool]] = {}
        self.hide_done_ids: Set[str] = set()
        self.live.start()

    def end(self) -> None:
        """Stop the live rendering session."""
        self.live.stop()

    def hide_done_checkmark(self, item_id: str) -> None:
        """Hide the completion checkmark for the given item id."""
        self.hide_done_ids.add(item_id)

    def update_item(
        self,
        item_id: str,
        content: str,
        *,
        is_done: bool = False,
        hide_checkmark: bool = False,
    ) -> None:
        """Insert or update a status line and refresh the live console."""
        self.items[item_id] = (content, is_done)
        if hide_checkmark:
            self.hide_done_ids.add(item_id)
        self._flush()

    def mark_item_done(self, item_id: str) -> None:
        """Mark an existing status line as completed."""
        if item_id in self.items:
            content, _ = self.items[item_id]
            self.items[item_id] = (content, True)
            self._flush()

    def _flush(self) -> None:
        """Re-render the live view with the latest status items."""
        renderables: List[Any] = []
        for item_id, (content, is_done) in self.items.items():
            if is_done:
                prefix = "✅ " if item_id not in self.hide_done_ids else ""
                renderables.append(prefix + content)
            else:
                renderables.append(Spinner("dots", text=content))
        if renderables:
            self.live.update(Group(*renderables))
        else:
            self.live.update(Group())
