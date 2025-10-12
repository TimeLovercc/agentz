"""
Miscellaneous helper utilities.
"""

import datetime
from pathlib import Path
from typing import Any, Dict


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
