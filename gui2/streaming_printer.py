"""Streaming printer dedicated to gui2.

This mirrors the behaviour required for server-sent streaming without relying
on the original gui package.
"""

from __future__ import annotations

import queue
from typing import Any, Dict, Optional

from rich.console import Console

from agentz.utils.printer import Printer


class StreamingPrinter(Printer):
    """Printer that captures updates for streaming to the frontend."""

    def __init__(self, console: Console):
        super().__init__(console)
        self.update_queue: queue.Queue = queue.Queue()
        self.is_streaming = True

    def _emit_update(self, event_type: str, data: Dict[str, Any]) -> None:
        if not self.is_streaming:
            return
        try:
            self.update_queue.put({"type": event_type, "data": data}, block=False)
        except queue.Full:
            pass

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
        super().update_item(
            item_id,
            content,
            is_done=is_done,
            hide_checkmark=hide_checkmark,
            title=title,
            border_style=border_style,
            group_id=group_id,
        )
        self._emit_update(
            "status_update",
            {
                "item_id": item_id,
                "content": content,
                "is_done": is_done,
                "title": title,
                "border_style": border_style,
                "group_id": group_id,
            },
        )

    def start_group(
        self,
        group_id: str,
        *,
        title: Optional[str] = None,
        border_style: Optional[str] = None,
    ) -> None:
        super().start_group(group_id, title=title, border_style=border_style)
        self._emit_update(
            "group_start",
            {
                "group_id": group_id,
                "title": title or group_id,
                "border_style": border_style or "white",
            },
        )

    def end_group(
        self,
        group_id: str,
        *,
        is_done: bool = True,
        title: Optional[str] = None,
    ) -> None:
        super().end_group(group_id, is_done=is_done, title=title)
        self._emit_update(
            "group_end",
            {"group_id": group_id, "is_done": is_done, "title": title},
        )

    def log_panel(
        self,
        title: str,
        content: str,
        *,
        border_style: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> None:
        super().log_panel(
            title,
            content,
            border_style=border_style,
            iteration=iteration,
        )
        self._emit_update(
            "log_panel",
            {
                "title": title,
                "content": content,
                "border_style": border_style,
                "iteration": iteration,
            },
        )

    def stop_streaming(self) -> None:
        self._emit_update("stream_end", {})
        self.is_streaming = False

    def get_updates(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        try:
            return self.update_queue.get(timeout=timeout)
        except queue.Empty:
            return None
