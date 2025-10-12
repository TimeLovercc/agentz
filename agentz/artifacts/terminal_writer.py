"""TerminalWriter handles real-time console output and panel display."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

if TYPE_CHECKING:
    from agentz.artifacts.reporter import PanelRecord


class TerminalWriter:
    """Handles real-time terminal display of panels and run information."""

    def __init__(
        self,
        *,
        run_dir: Path,
        console: Optional[Console] = None,
    ) -> None:
        self.run_dir = run_dir
        self.console = console
        self._panels: List[PanelRecord] = []

    def record_panel(self, record: PanelRecord) -> None:
        """Store panel for later terminal display."""
        self._panels.append(record)

    def print_terminal_report(self) -> None:
        """Stream captured panel content back to the console."""
        if not self.console or not self._panels:
            return

        panels = self._select_terminal_panels()
        if not panels:
            return

        self.console.print(
            Text(f"Run artefacts saved to {self.run_dir}", style="bold cyan")
        )
        for record in panels:
            renderable = self._panel_renderable(record.content)
            panel = Panel(
                renderable,
                title=record.title,
                border_style=record.border_style or "cyan",
                padding=(1, 2),
            )
            self.console.print(panel)

    # ----------------------------------------------------------------- helpers

    def _select_terminal_panels(self) -> List[PanelRecord]:
        """Return only final panels for terminal replay."""
        final_panels = [
            record for record in self._panels if self._is_final_panel(record)
        ]
        if final_panels:
            return final_panels
        # Fallback: display only the most recent panel
        return self._panels[-1:]

    @staticmethod
    def _is_final_panel(record: PanelRecord) -> bool:
        """Heuristic for identifying final report panels."""
        if record.group_id and "final" in record.group_id.lower():
            return True
        if record.title:
            title = record.title.lower()
            if "final" in title or "writer" in title:
                return True
        return False

    def _panel_renderable(self, content: str):
        """Render Markdown panels using rich, otherwise plain text."""
        if self._looks_like_markdown(content):
            return Markdown(content)
        return Text(content)

    @staticmethod
    def _looks_like_markdown(content: str) -> bool:
        """Rudimentary detection of Markdown content."""
        if not content:
            return False
        markdown_patterns = (
            r"^#{1,6}\s",           # headings
            r"^\s*[-*+]\s+\S",      # bullet lists
            r"^\s*\d+\.\s+\S",      # numbered lists
            r"`{1,3}.+?`{1,3}",     # inline or fenced code
            r"\*\*.+\*\*",          # bold text
            r"_{1,2}.+_{1,2}",      # italic/underline emphasis
        )
        return any(re.search(pattern, content, re.MULTILINE) for pattern in markdown_patterns)

