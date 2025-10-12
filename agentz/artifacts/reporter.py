"""Shared data models and RunReporter facade for pipeline runs."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rich.console import Console

from agentz.artifacts.artifact_writer import ArtifactWriter
from agentz.artifacts.terminal_writer import TerminalWriter


@dataclass
class PanelRecord:
    """Representation of a panel rendered during the run."""

    title: Optional[str]
    content: str
    border_style: Optional[str]
    iteration: Optional[int]
    group_id: Optional[str]
    recorded_at: str


@dataclass
class AgentStepRecord:
    """Runtime information captured per agent execution."""

    agent_name: str
    span_name: str
    iteration: Optional[int]
    group_id: Optional[str]
    started_at: str
    finished_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str = "running"
    error: Optional[str] = None


class RunReporter:
    """Facade combining terminal display and artifact persistence."""

    def __init__(
        self,
        *,
        base_dir: Path,
        pipeline_slug: str,
        workflow_name: str,
        experiment_id: str,
        console: Optional[Console] = None,
    ) -> None:
        self.base_dir = base_dir
        self.pipeline_slug = pipeline_slug
        self.workflow_name = workflow_name
        self.experiment_id = experiment_id
        self.console = console

        self.run_dir = base_dir / pipeline_slug / experiment_id
        self.terminal_md_path = self.run_dir / "terminal_log.md"
        self.terminal_html_path = self.run_dir / "terminal_log.html"
        self.final_report_md_path = self.run_dir / "final_report.md"
        self.final_report_html_path = self.run_dir / "final_report.html"

        self._lock = threading.RLock()

        # Delegate to specialized reporters
        self._artifact_writer = ArtifactWriter(
            base_dir=base_dir,
            pipeline_slug=pipeline_slug,
            workflow_name=workflow_name,
            experiment_id=experiment_id,
        )
        self._terminal_writer = TerminalWriter(
            run_dir=self.run_dir,
            console=console,
        )

    # ------------------------------------------------------------------ basics

    def start(self, config: Any) -> None:
        """Prepare filesystem layout and capture start metadata."""
        with self._lock:
            self._artifact_writer.start(config)

    def set_final_result(self, result: Any) -> None:
        """Store pipeline result for later persistence."""
        with self._lock:
            self._artifact_writer.set_final_result(result)

    # ----------------------------------------------------------------- logging

    def record_status_update(
        self,
        *,
        item_id: str,
        content: str,
        is_done: bool,
        title: Optional[str],
        border_style: Optional[str],
        group_id: Optional[str],
    ) -> None:
        """Currently unused; maintained for interface compatibility."""
        with self._lock:
            self._artifact_writer.record_status_update(
                item_id=item_id,
                content=content,
                is_done=is_done,
                title=title,
                border_style=border_style,
                group_id=group_id,
            )

    def record_group_start(
        self,
        *,
        group_id: str,
        title: Optional[str],
        border_style: Optional[str],
        iteration: Optional[int] = None,
    ) -> None:
        """Record the start of an iteration/group."""
        with self._lock:
            self._artifact_writer.record_group_start(
                group_id=group_id,
                title=title,
                border_style=border_style,
                iteration=iteration,
            )

    def record_group_end(
        self,
        *,
        group_id: str,
        is_done: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Record the end of an iteration/group."""
        with self._lock:
            self._artifact_writer.record_group_end(
                group_id=group_id,
                is_done=is_done,
                title=title,
            )

    def record_agent_step_start(
        self,
        *,
        step_id: str,
        agent_name: str,
        span_name: str,
        iteration: Optional[int],
        group_id: Optional[str],
        printer_title: Optional[str],
    ) -> None:
        """Capture metadata when an agent step begins."""
        with self._lock:
            self._artifact_writer.record_agent_step_start(
                step_id=step_id,
                agent_name=agent_name,
                span_name=span_name,
                iteration=iteration,
                group_id=group_id,
                printer_title=printer_title,
            )

    def record_agent_step_end(
        self,
        *,
        step_id: str,
        status: str,
        duration_seconds: float,
        error: Optional[str] = None,
    ) -> None:
        """Update agent step telemetry on completion."""
        with self._lock:
            self._artifact_writer.record_agent_step_end(
                step_id=step_id,
                status=status,
                duration_seconds=duration_seconds,
                error=error,
            )

    def record_panel(
        self,
        *,
        title: str,
        content: str,
        border_style: Optional[str],
        iteration: Optional[int],
        group_id: Optional[str],
    ) -> None:
        """Persist panel meta for terminal & HTML artefacts."""
        with self._lock:
            # Create panel record
            from agentz.artifacts.artifact_writer import _utc_timestamp
            record = PanelRecord(
                title=title,
                content=content,
                border_style=border_style,
                iteration=iteration,
                group_id=group_id,
                recorded_at=_utc_timestamp(),
            )
            # Record in both reporters
            self._artifact_writer.record_panel(
                title=title,
                content=content,
                border_style=border_style,
                iteration=iteration,
                group_id=group_id,
            )
            self._terminal_writer.record_panel(record)

    # ------------------------------------------------------------- finalisation

    def finalize(self) -> None:
        """Persist markdown + HTML artefacts."""
        with self._lock:
            self._artifact_writer.finalize()

    # ---------------------------------------------------------- terminal flush

    def print_terminal_report(self) -> None:
        """Stream captured panel content back to the console."""
        self._terminal_writer.print_terminal_report()

    # ----------------------------------------------------------------- helpers

    def ensure_started(self) -> None:
        """Raise if reporter not initialised."""
        self._artifact_writer.ensure_started()

