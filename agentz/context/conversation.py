from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, List, Optional

from pydantic import BaseModel, Field

from agentz.profiles.registry import ToolAgentOutput

if TYPE_CHECKING:
    from agentz.profiles.manager.routing import AgentSelectionPlan, AgentTask
else:
    AgentSelectionPlan = AgentTask = Any


class KnowledgeGapOutput(BaseModel):
    """Evaluation result capturing research completion status and gaps."""

    research_complete: bool = Field(description="Whether the research is complete")
    outstanding_gaps: List[str] = Field(
        description="List of outstanding knowledge gaps", default_factory=list
    )
    reasoning: str = Field(description="Reasoning behind the evaluation", default="")


class ToolExecutionResult(BaseModel):
    """Structured record of a tool execution within an iteration."""

    task: AgentTask
    output: ToolAgentOutput

    def as_history_block(self) -> str:
        parts = [
            f"[Tool] {self.task.agent}",
            f"[Query] {self.task.query}",
        ]
        if self.task.entity_website:
            parts.append(f"[Entity] {self.task.entity_website}")
        detail = "\n".join(parts)
        output = self.output.output if hasattr(self.output, "output") else ""
        return f"{detail}\n\n{output}".strip()


class IterationRecord(BaseModel):
    """State captured for a single iteration of the research loop."""

    index: int
    observation: Optional[str] = None
    evaluation: Optional[KnowledgeGapOutput] = None
    selected_gap: Optional[str] = None
    route_plan: Optional[AgentSelectionPlan] = None
    tools: List[ToolExecutionResult] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    status: str = Field(default="pending", description="Iteration status: pending or complete")
    summarized: bool = Field(default=False, description="Whether this iteration has been summarised")

    def mark_complete(self) -> None:
        self.status = "complete"

    def is_complete(self) -> bool:
        return self.status == "complete"

    def mark_summarized(self) -> None:
        self.summarized = True

    def history_block(self) -> str:
        """Render this iteration as a formatted history block for prompts."""
        lines: List[str] = [f"[ITERATION {self.index}]"]

        if self.observation:
            lines.append(f"<thought>\n{self.observation}\n</thought>")

        if self.selected_gap:
            lines.append(f"<task>\nAddress this knowledge gap: {self.selected_gap}\n</task>")

        if self.route_plan and self.route_plan.tasks:
            task_lines = []
            for task in self.route_plan.tasks:
                entity = task.entity_website or "null"
                task_lines.append(f"[Agent] {task.agent} [Query] {task.query} [Entity] {entity}")
            lines.append(
                "<action>\nCalling the following tools to address the knowledge gap:\n"
                + "\n".join(task_lines)
                + "\n</action>"
            )

        if self.tools:
            findings_text = "\n\n".join(result.as_history_block() for result in self.tools)
            lines.append(f"<findings>\n{findings_text}\n</findings>")
        elif self.findings:
            findings_text = "\n\n".join(self.findings)
            lines.append(f"<findings>\n{findings_text}\n</findings>")

        return "\n\n".join(lines).strip()


class ConversationState(BaseModel):
    """Centralised storage for all iteration data shared across agents."""

    query: str = ""
    data_path: Optional[str] = None
    max_iterations: int = 5
    max_minutes: float = 10.0

    iterations: List[IterationRecord] = Field(default_factory=list)
    final_report: Optional[str] = None
    started_at: Optional[float] = None
    complete: bool = False
    summary: Optional[str] = None

    def set_query(self, query: str) -> None:
        self.query = query

    def start_timer(self) -> None:
        self.started_at = time.time()

    def elapsed_minutes(self) -> float:
        if self.started_at is None:
            return 0.0
        return (time.time() - self.started_at) / 60

    def begin_iteration(self) -> IterationRecord:
        iteration = IterationRecord(index=len(self.iterations) + 1)
        self.iterations.append(iteration)
        return iteration

    @property
    def current_iteration(self) -> IterationRecord:
        if not self.iterations:
            raise ValueError("No iteration has been started yet.")
        return self.iterations[-1]

    def mark_iteration_complete(self) -> None:
        self.current_iteration.mark_complete()

    def mark_research_complete(self) -> None:
        self.complete = True
        self.current_iteration.mark_complete()

    def iteration_history(self, include_current: bool = False) -> str:
        relevant = [
            iteration
            for iteration in self.iterations
            if iteration.is_complete() or include_current and iteration is self.current_iteration
        ]
        blocks = [iteration.history_block() for iteration in relevant if iteration.history_block()]
        return "\n\n".join(blocks).strip()

    def unsummarized_history(self, include_current: bool = True) -> str:
        relevant = [
            iteration
            for iteration in self.iterations
            if (iteration.is_complete() or include_current and iteration is self.current_iteration)
            and not iteration.summarized
        ]
        blocks = [iteration.history_block() for iteration in relevant if iteration.history_block()]
        return "\n\n".join(blocks).strip()

    def history_with_summary(self) -> str:
        summary_section = ""
        if self.summary:
            summary_section = f"[SUMMARY BEFORE NEW ITERATION]\n\n{self.summary}\n\n"
        return summary_section + self.unsummarized_history()

    def all_findings(self) -> List[str]:
        findings: List[str] = []
        for iteration in self.iterations:
            findings.extend(iteration.findings)
            findings.extend(result.output.output for result in iteration.tools if hasattr(result.output, "output"))
        return findings

    def findings_text(self) -> str:
        findings = self.all_findings()
        return "\n\n".join(findings).strip() if findings else ""

    def update_summary(self, summary: str) -> None:
        self.summary = summary
        for iteration in self.iterations:
            iteration.mark_summarized()
