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
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

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
    """Rich-powered status printer for streaming pipeline progress updates.

    - Each item is displayed as a Panel (box) with a title.
    - In-progress items show a spinner above the panel.
    - Completed items show a checkmark in the panel title (unless hidden).
    - Supports nested layout: groups (iterations) contain section panels.
    - Per-section border colors with sensible defaults.
    - Backward compatible with previous `update_item` signature.
    """

    # Default border colors by section name
    DEFAULT_BORDER_COLORS = {
        "observations": "yellow",
        "observation": "yellow",
        "observe": "yellow",
        "evaluation": "magenta",
        "evaluate": "magenta",
        "routing": "blue",
        "route": "blue",
        "tools": "cyan",
        "tool": "cyan",
        "writer": "green",
        "write": "green",
    }

    def __init__(self, console: Console) -> None:
        self.console = console
        # You can tweak screen=True to prevent scrollback; kept False to preserve logs
        self.live = Live(
            console=console,
            refresh_per_second=12,
            vertical_overflow="visible",   # <-- key change: no height cropping
            screen=False,                  # keep using normal screen buffer
            transient=False,               # keep contents after stopping
        )

        # items: id -> (content, is_done, title, border_style, group_id)
        self.items: Dict[str, Tuple[str, bool, Optional[str], Optional[str], Optional[str]]] = {}
        # Track which items should hide the done checkmark
        self.hide_done_ids: Set[str] = set()

        # Group management
        self.group_order: List[str] = []  # Order of groups
        self.groups: Dict[str, Dict[str, Any]] = {}  # group_id -> {title, is_done, border_style, order}
        self.item_order: List[str] = []  # Order of top-level items (no group_id)

        self.live.start()

    def end(self) -> None:
        """Stop the live rendering session."""
        self.live.stop()

    def hide_done_checkmark(self, item_id: str) -> None:
        """Hide the completion checkmark for the given item id."""
        self.hide_done_ids.add(item_id)

    def start_group(
        self,
        group_id: str,
        *,
        title: Optional[str] = None,
        border_style: Optional[str] = None
    ) -> None:
        """Start a new group (e.g., an iteration panel).

        Args:
            group_id: Unique identifier for the group
            title: Optional title for the group panel
            border_style: Optional border color (defaults to white)
        """
        if group_id not in self.groups:
            self.group_order.append(group_id)
        self.groups[group_id] = {
            "title": title or group_id,
            "is_done": False,
            "border_style": border_style or "white",
            "order": []  # Track order of items in this group
        }
        self._flush()

    def end_group(
        self,
        group_id: str,
        *,
        is_done: bool = True,
        title: Optional[str] = None
    ) -> None:
        """Mark a group as complete.

        Args:
            group_id: Unique identifier for the group
            is_done: Whether the group is complete (default: True)
            title: Optional updated title for the group
        """
        if group_id in self.groups:
            self.groups[group_id]["is_done"] = is_done
            if title:
                self.groups[group_id]["title"] = title
            # Update border style to bright_white when done
            if is_done and self.groups[group_id]["border_style"] == "white":
                self.groups[group_id]["border_style"] = "bright_white"
            self._flush()

    def update_item(
        self,
        item_id: str,
        content: str,
        *,
        is_done: bool = False,
        hide_checkmark: bool = False,
        title: Optional[str] = None,
        border_style: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """Insert or update a status line and refresh the live console.

        Args:
            item_id: Unique identifier for the item
            content: Content to display
            is_done: Whether the task is complete
            hide_checkmark: Hide completion checkmark
            title: Optional panel title
            border_style: Optional border color (auto-detected from title if not provided)
            group_id: Optional group to nest this item in
        """
        # Auto-detect border style from title if not provided
        if border_style is None and title:
            title_lower = title.lower().strip()
            # Check for exact match first
            if title_lower in self.DEFAULT_BORDER_COLORS:
                border_style = self.DEFAULT_BORDER_COLORS[title_lower]
            else:
                # Check if any key is a substring of the title
                for key, color in self.DEFAULT_BORDER_COLORS.items():
                    if key in title_lower:
                        border_style = color
                        break

        # Track item in appropriate order list
        if item_id not in self.items:
            if group_id and group_id in self.groups:
                if item_id not in self.groups[group_id]["order"]:
                    self.groups[group_id]["order"].append(item_id)
            elif item_id not in self.item_order:
                self.item_order.append(item_id)

        self.items[item_id] = (content, is_done, title, border_style, group_id)
        if hide_checkmark:
            self.hide_done_ids.add(item_id)
        self._flush()

    def mark_item_done(
        self,
        item_id: str,
        *,
        title: Optional[str] = None,
        border_style: Optional[str] = None
    ) -> None:
        """Mark an existing status line as completed (optionally update title/border).

        Args:
            item_id: Unique identifier for the item
            title: Optional updated title
            border_style: Optional updated border color
        """
        if item_id in self.items:
            content, _, old_title, old_border, group_id = self.items[item_id]
            self.items[item_id] = (
                content,
                True,
                title or old_title,
                border_style or old_border,
                group_id
            )
            self._flush()

    # ------------ internals ------------

    def _render_title(self, item_id: str, is_done: bool, title: Optional[str]) -> Text:
        """Render panel title with optional checkmark."""
        base = title or item_id
        if is_done and item_id not in self.hide_done_ids:
            # checkmark in the panel title
            return Text(f"✅ {base}")
        elif not is_done:
            # no checkmark while running; spinner is rendered above the panel
            return Text(base)
        return Text(base)

    def _detect_and_render_body(self, content: str) -> Any:
        """Auto-detect content type and render with appropriate Rich object.

        Detection order:
        1. ANSI escape codes → Text.from_ansi
        2. Rich markup (e.g., [bold cyan]...[/]) → Text.from_markup
        3. Code fences (```) → Markdown
        4. Plain text → Text
        """
        # Check for ANSI escape codes
        ansi_pattern = r'\x1b\[[0-9;]*m'
        if re.search(ansi_pattern, content):
            return Text.from_ansi(content)

        # Check for Rich markup patterns
        rich_markup_pattern = r'\[/?[a-z_]+(?:\s+[a-z_]+)*\]'
        if re.search(rich_markup_pattern, content, re.IGNORECASE):
            return Text.from_markup(content, emoji=True)

        # Check for code fences
        if '```' in content:
            return Markdown(content, code_theme="monokai")

        # Default to plain text
        return Text(content)

    def _get_border_color(self, is_done: bool, border_style: Optional[str]) -> str:
        """Determine border color for an item.

        Args:
            is_done: Whether the item is complete
            border_style: Explicitly provided border style

        Returns:
            Border style string
        """
        if border_style:
            return border_style
        # Default colors if no style provided
        return "cyan" if not is_done else "green"

    def _render_item(
        self,
        item_id: str,
        content: str,
        is_done: bool,
        title: Optional[str],
        border_style: Optional[str]
    ) -> Any:
        """Render a single item as a Panel with optional spinner."""
        panel = Panel(
            self._detect_and_render_body(content),
            title=self._render_title(item_id, is_done, title),
            border_style=self._get_border_color(is_done, border_style),
            padding=(1, 2),
            expand=True,
        )
        if is_done:
            return panel
        # For in-progress, show a spinner stacked above the panel
        spin = Spinner("dots", text=Text("Working…"))
        return Group(spin, panel)

    def _render_group(self, group_id: str) -> Any:
        """Render a group panel containing its child section panels.

        Args:
            group_id: The group to render

        Returns:
            A Panel containing Group of child panels
        """
        group = self.groups[group_id]
        group_items = []

        # Render items in the order they were added to this group
        for item_id in group["order"]:
            if item_id in self.items:
                content, is_done, title, border_style, _ = self.items[item_id]
                group_items.append(
                    self._render_item(item_id, content, is_done, title, border_style)
                )

        # Create group title with checkmark if done
        group_title = group["title"]
        if group["is_done"]:
            group_title = f"✅ {group_title}"

        # Create panel containing all group items
        return Panel(
            Group(*group_items) if group_items else Text(""),
            title=Text(group_title),
            border_style=group["border_style"],
            padding=(1, 2),
            expand=True,
        )

    def _flush(self) -> None:
        """Re-render the live view with the latest status items."""
        renderables: List[Any] = []

        # Render top-level items (those without group_id)
        for item_id in self.item_order:
            if item_id in self.items:
                content, is_done, title, border_style, group_id = self.items[item_id]
                if group_id is None:
                    renderables.append(
                        self._render_item(item_id, content, is_done, title, border_style)
                    )

        # Render groups in order
        for group_id in self.group_order:
            if group_id in self.groups:
                renderables.append(self._render_group(group_id))

        self.live.update(Group(*renderables) if renderables else Group())