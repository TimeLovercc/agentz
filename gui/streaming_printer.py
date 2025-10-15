"""Streaming Printer for GUI Integration.

This module provides a custom Printer implementation that captures
pipeline updates and makes them available for streaming to GUI clients.
"""

import json
import queue
from typing import Any, Dict, Optional, Set, Tuple
from rich.console import Console

from agentz.utils.printer import Printer


class StreamingPrinter(Printer):
    """Extended Printer that captures updates for streaming to GUI.

    Inherits from Printer and adds a queue for capturing all updates
    that can be consumed by GUI SSE endpoints.
    """

    def __init__(self, console: Console):
        super().__init__(console)
        # Queue for streaming updates to GUI
        self.update_queue: queue.Queue = queue.Queue()
        self.is_streaming = True

    def _emit_update(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an update event to the queue for GUI streaming.

        Args:
            event_type: Type of event (status, group_start, group_end, log, etc.)
            data: Event data dictionary
        """
        if self.is_streaming:
            try:
                event = {
                    "type": event_type,
                    "data": data
                }
                self.update_queue.put(event, block=False)
            except queue.Full:
                # Queue full, skip this update
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
        """Override to capture item updates for streaming."""
        super().update_item(
            item_id,
            content,
            is_done=is_done,
            hide_checkmark=hide_checkmark,
            title=title,
            border_style=border_style,
            group_id=group_id,
        )

        # Emit update event
        self._emit_update("status_update", {
            "item_id": item_id,
            "content": content,
            "is_done": is_done,
            "title": title,
            "border_style": border_style,
            "group_id": group_id,
        })

    def start_group(
        self,
        group_id: str,
        *,
        title: Optional[str] = None,
        border_style: Optional[str] = None
    ) -> None:
        """Override to capture group start for streaming."""
        super().start_group(group_id, title=title, border_style=border_style)

        # Emit group start event
        self._emit_update("group_start", {
            "group_id": group_id,
            "title": title or group_id,
            "border_style": border_style or "white",
        })

    def end_group(
        self,
        group_id: str,
        *,
        is_done: bool = True,
        title: Optional[str] = None
    ) -> None:
        """Override to capture group end for streaming."""
        super().end_group(group_id, is_done=is_done, title=title)

        # Emit group end event
        self._emit_update("group_end", {
            "group_id": group_id,
            "is_done": is_done,
            "title": title,
        })

    def log_panel(
        self,
        title: str,
        content: str,
        *,
        border_style: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """Override to capture panel logs for streaming."""
        super().log_panel(
            title,
            content,
            border_style=border_style,
            iteration=iteration,
        )

        # Emit log panel event
        self._emit_update("log_panel", {
            "title": title,
            "content": content,
            "border_style": border_style,
            "iteration": iteration,
        })

    def stop_streaming(self) -> None:
        """Stop capturing updates and signal end of stream."""
        print("[DEBUG] StreamingPrinter.stop_streaming() called")
        # Emit stream_end BEFORE setting is_streaming to False
        self._emit_update("stream_end", {})
        print("[DEBUG] stream_end event emitted")
        self.is_streaming = False

    def get_updates(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get next update from queue (non-blocking with timeout).

        Args:
            timeout: Timeout in seconds

        Returns:
            Update event dictionary or None if queue is empty
        """
        try:
            return self.update_queue.get(timeout=timeout)
        except queue.Empty:
            return None
