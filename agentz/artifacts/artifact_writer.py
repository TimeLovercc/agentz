"""ArtifactWriter persists run data to markdown and HTML files."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from agentz.artifacts.reporter import AgentStepRecord, PanelRecord


def _utc_timestamp() -> str:
    """Return current UTC timestamp with second precision."""
    return datetime.utcnow().replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def _json_default(obj: Any) -> Any:
    """Fallback JSON serialiser for arbitrary objects."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


class ArtifactWriter:
    """Collects run data and persists it as markdown and HTML artifacts."""

    def __init__(
        self,
        *,
        base_dir: Path,
        pipeline_slug: str,
        workflow_name: str,
        experiment_id: str,
    ) -> None:
        self.base_dir = base_dir
        self.pipeline_slug = pipeline_slug
        self.workflow_name = workflow_name
        self.experiment_id = experiment_id

        self.run_dir = base_dir / pipeline_slug / experiment_id
        self.terminal_md_path = self.run_dir / "terminal_log.md"
        self.terminal_html_path = self.run_dir / "terminal_log.html"
        self.final_report_md_path = self.run_dir / "final_report.md"
        self.final_report_html_path = self.run_dir / "final_report.html"

        self._panels: List[PanelRecord] = []
        self._agent_steps: Dict[str, AgentStepRecord] = {}
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._iterations: Dict[str, Dict[str, Any]] = {}
        self._final_result: Optional[Any] = None

        self._start_time: Optional[float] = None
        self._started_at_iso: Optional[str] = None
        self._finished_at_iso: Optional[str] = None

    # ------------------------------------------------------------------ basics

    def start(self, config: Any) -> None:  # noqa: ARG002 - config reserved for future use
        """Prepare filesystem layout and capture start metadata."""
        if self._start_time is not None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()
        self._started_at_iso = _utc_timestamp()

    def set_final_result(self, result: Any) -> None:
        """Store pipeline result for later persistence."""
        self._final_result = result

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
    ) -> None:  # noqa: D401 - keeps signature compatibility
        """Currently unused; maintained for interface compatibility."""
        # Intentionally no-op for the simplified reporter.
        return None

    def record_group_start(
        self,
        *,
        group_id: str,
        title: Optional[str],
        border_style: Optional[str],
        iteration: Optional[int] = None,
    ) -> None:
        """Record the start of an iteration/group."""
        timestamp = _utc_timestamp()
        payload = {
            "group_id": group_id,
            "title": title,
            "border_style": border_style,
            "iteration": iteration,
            "started_at": timestamp,
        }
        self._groups[group_id] = payload
        if iteration is not None:
            iter_key = f"iter-{iteration}"
            self._iterations.setdefault(
                iter_key,
                {
                    "iteration": iteration,
                    "title": title or f"Iteration {iteration}",
                    "started_at": timestamp,
                    "finished_at": None,
                    "panels": [],
                    "agent_steps": [],
                },
            )

    def record_group_end(
        self,
        *,
        group_id: str,
        is_done: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Record the end of an iteration/group."""
        timestamp = _utc_timestamp()
        group_meta = self._groups.get(group_id)
        if not group_meta:
            return
        group_meta.update(
            {
                "title": title or group_meta.get("title"),
                "is_done": is_done,
                "finished_at": timestamp,
            }
        )
        iteration = group_meta.get("iteration")
        if iteration is not None:
            iter_key = f"iter-{iteration}"
            iteration_meta = self._iterations.setdefault(
                iter_key,
                {
                    "iteration": iteration,
                    "title": title or f"Iteration {iteration}",
                    "panels": [],
                    "agent_steps": [],
                },
            )
            iteration_meta["finished_at"] = timestamp

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
        from agentz.artifacts.reporter import AgentStepRecord
        
        record = AgentStepRecord(
            agent_name=agent_name,
            span_name=span_name,
            iteration=iteration,
            group_id=group_id,
            started_at=_utc_timestamp(),
        )
        self._agent_steps[step_id] = record
        if iteration is not None:
            iter_key = f"iter-{iteration}"
            iteration_meta = self._iterations.setdefault(
                iter_key,
                {
                    "iteration": iteration,
                    "title": printer_title or f"Iteration {iteration}",
                    "panels": [],
                    "agent_steps": [],
                },
            )
            iteration_meta["agent_steps"].append(record)

    def record_agent_step_end(
        self,
        *,
        step_id: str,
        status: str,
        duration_seconds: float,
        error: Optional[str] = None,
    ) -> None:
        """Update agent step telemetry on completion."""
        timestamp = _utc_timestamp()
        record = self._agent_steps.get(step_id)
        if record:
            record.finished_at = timestamp
            record.duration_seconds = round(duration_seconds, 3)
            record.status = status
            record.error = error

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
        from agentz.artifacts.reporter import PanelRecord
        
        record = PanelRecord(
            title=title,
            content=content,
            border_style=border_style,
            iteration=iteration,
            group_id=group_id,
            recorded_at=_utc_timestamp(),
        )
        self._panels.append(record)
        if iteration is not None:
            iter_key = f"iter-{iteration}"
            iteration_meta = self._iterations.setdefault(
                iter_key,
                {
                    "iteration": iteration,
                    "title": f"Iteration {iteration}",
                    "panels": [],
                    "agent_steps": [],
                },
            )
            iteration_meta["panels"].append(record)

    # ------------------------------------------------------------- finalisation

    def finalize(self) -> None:
        """Persist markdown + HTML artefacts."""
        if self._start_time is None or self._finished_at_iso is not None:
            return
        self._finished_at_iso = _utc_timestamp()
        duration = round(time.time() - self._start_time, 3)

        terminal_sections = self._build_terminal_sections()
        terminal_md = self._render_terminal_markdown(duration, terminal_sections)
        terminal_html = self._render_terminal_html(duration, terminal_sections)

        self.terminal_md_path.write_text(terminal_md, encoding="utf-8")
        self.terminal_html_path.write_text(terminal_html, encoding="utf-8")

        final_md, final_html = self._render_final_report()
        self.final_report_md_path.write_text(final_md, encoding="utf-8")
        self.final_report_html_path.write_text(final_html, encoding="utf-8")

    def _build_terminal_sections(self) -> List[Dict[str, Any]]:
        """Collect ordered sections for terminal artefacts."""
        sections: List[Dict[str, Any]] = []

        # Iteration/scoped panels
        for iter_key, meta in sorted(
            self._iterations.items(),
            key=lambda item: item[1].get("iteration", 0),
        ):
            sections.append(
                {
                    "title": meta.get("title") or iter_key,
                    "started_at": meta.get("started_at"),
                    "finished_at": meta.get("finished_at"),
                    "panels": meta.get("panels", []),
                    "agent_steps": meta.get("agent_steps", []),
                }
            )

        # Global panels (no iteration)
        global_panels = [
            record
            for record in self._panels
            if record.iteration is None
        ]
        if global_panels:
            sections.append(
                {
                    "title": "General",
                    "started_at": None,
                    "finished_at": None,
                    "panels": global_panels,
                    "agent_steps": [],
                }
            )

        return sections

    def _render_terminal_markdown(
        self,
        duration: float,
        sections: List[Dict[str, Any]],
    ) -> str:
        """Render the terminal log as Markdown."""
        lines: List[str] = []
        lines.append(f"# Terminal Log · {self.workflow_name}")
        lines.append("")
        lines.append(f"- **Experiment ID:** `{self.experiment_id}`")
        lines.append(f"- **Started:** {self._started_at_iso or '–'}")
        lines.append(f"- **Finished:** {self._finished_at_iso or '–'}")
        lines.append(f"- **Duration:** {duration} seconds")
        lines.append("")

        if not sections:
            lines.append("_No panels recorded during this run._")
            lines.append("")
            return "\n".join(lines)

        for section in sections:
            lines.append(f"## {section['title']}")
            span = ""
            if section.get("started_at") or section.get("finished_at"):
                span = f"{section.get('started_at', '–')} → {section.get('finished_at', '–')}"
            if span:
                lines.append(f"*Time:* {span}")
            lines.append("")

            agent_steps: List[AgentStepRecord] = section.get("agent_steps", [])
            if agent_steps:
                lines.append("### Agent Steps")
                for step in agent_steps:
                    duration_txt = (
                        f"{step.duration_seconds}s"
                        if step.duration_seconds is not None
                        else "pending"
                    )
                    status = step.status
                    error = f" · Error: {step.error}" if step.error else ""
                    lines.append(
                        f"- **{step.agent_name}** · {step.span_name} "
                        f"({duration_txt}) · {status}{error}"
                    )
                lines.append("")

            panels: List[PanelRecord] = section.get("panels", [])
            for panel in panels:
                panel_title = panel.title or "Panel"
                lines.append(f"### {panel_title}")
                lines.append("")
                lines.append("```")
                lines.append(panel.content.rstrip())
                lines.append("```")
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _render_terminal_html(
        self,
        duration: float,
        sections: List[Dict[str, Any]],
    ) -> str:
        """Render the terminal log as standalone HTML."""
        body_sections: List[str] = []

        for section in sections:
            panels_html: List[str] = []
            for panel in section.get("panels", []):
                panel_html = f"""
        <article class="panel">
          <h3>{panel.title or "Panel"}</h3>
          <pre>{panel.content}</pre>
        </article>
        """.strip()
                panels_html.append(panel_html)

            agent_html: List[str] = []
            for step in section.get("agent_steps", []):
                info = json.dumps(
                    {
                        "agent": step.agent_name,
                        "span": step.span_name,
                        "status": step.status,
                        "duration_seconds": step.duration_seconds,
                        "error": step.error,
                    },
                    default=_json_default,
                )
                agent_html.append(f'<li><code>{info}</code></li>')

            timeframe = ""
            if section.get("started_at") or section.get("finished_at"):
                timeframe = (
                    f"<p class=\"time\">{section.get('started_at', '–')} → "
                    f"{section.get('finished_at', '–')}</p>"
                )

            agents_block = ""
            if agent_html:
                agents_block = '<ul class="agents">' + "".join(agent_html) + "</ul>"

            panels_block = "".join(panels_html)

            block = (
                f"\n      <section class=\"section\">\n"
                f"        <h2>{section['title']}</h2>\n"
                f"        {timeframe}\n"
                f"        {agents_block}\n"
                f"        {panels_block}\n"
                "      </section>\n      "
            ).strip()
            body_sections.append(block)

        sections_html = "\n".join(body_sections) if body_sections else "<p>No panels recorded.</p>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Terminal Log · {self.workflow_name}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 24px;
      background: #0f172a;
      color: #e2e8f0;
    }}
    h1 {{
      margin-top: 0;
    }}
    .meta {{
      margin-bottom: 24px;
      line-height: 1.6;
    }}
    .section {{
      border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 12px;
      padding: 16px 20px;
      margin-bottom: 18px;
      background: rgba(15, 23, 42, 0.6);
    }}
    .section h2 {{
      margin-top: 0;
    }}
    .section .time {{
      color: #60a5fa;
      font-size: 0.9rem;
      margin-top: -8px;
    }}
    pre {{
      background: rgba(15, 23, 42, 0.85);
      border-radius: 10px;
      padding: 12px;
      overflow-x: auto;
      border: 1px solid rgba(148, 163, 184, 0.2);
      white-space: pre-wrap;
      word-wrap: break-word;
    }}
    ul.agents {{
      list-style: none;
      padding-left: 0;
      margin: 0 0 16px 0;
    }}
    ul.agents li {{
      margin-bottom: 6px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Terminal Log · {self.workflow_name}</h1>
    <div class="meta">
      <div><strong>Experiment ID:</strong> {self.experiment_id}</div>
      <div><strong>Started:</strong> {self._started_at_iso or "–"}</div>
      <div><strong>Finished:</strong> {self._finished_at_iso or "–"}</div>
      <div><strong>Duration:</strong> {duration} seconds</div>
    </div>
  </header>
  <main>
    {sections_html}
  </main>
</body>
</html>
"""

    def _render_final_report(self) -> tuple[str, str]:
        """Render final report markdown + HTML."""
        if isinstance(self._final_result, str):
            body_md = self._final_result.rstrip()
        elif self._final_result is not None:
            body_md = json.dumps(self._final_result, indent=2, default=_json_default)
        else:
            body_md = "No final report generated."

        markdown_content = f"# Final Report · {self.workflow_name}\n\n{body_md}\n"

        body_pre = body_md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Final Report · {self.workflow_name}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 24px;
      background: #111827;
      color: #f9fafb;
    }}
    h1 {{
      margin-top: 0;
    }}
    pre {{
      background: #1f2937;
      border-radius: 10px;
      padding: 16px;
      overflow-x: auto;
      white-space: pre-wrap;
      word-wrap: break-word;
      border: 1px solid rgba(148, 163, 184, 0.3);
    }}
  </style>
</head>
<body>
  <h1>Final Report · {self.workflow_name}</h1>
  <pre>{body_pre}</pre>
</body>
</html>
"""
        return markdown_content, html_content

    # ------------------------------------------------------------------ helpers

    def ensure_started(self) -> None:
        """Raise if reporter not initialised."""
        if self._start_time is None:
            raise RuntimeError("ArtifactWriter.start must be called before logging events.")
