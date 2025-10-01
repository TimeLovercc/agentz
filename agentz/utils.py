"""
Utility functions for the multi-agent data science system.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


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