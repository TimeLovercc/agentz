from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentz.agent.base import ContextAgent
from agentz.context.context import Context
from pipelines.base import BasePipeline


class WebSearchQuery(BaseModel):
    """Query model for data science tasks."""
    prompt: str

    def format(self) -> str:
        """Format web search query."""
        return (
            f"Web search query: {self.prompt}\n"
            "Provide a comprehensive web search workflow"
        )


class WebSearcherPipeline(BasePipeline):
    """Web search pipeline using manager-tool pattern with multi-task planning.

    This pipeline demonstrates web search with parallel query execution:
    - __init__: Setup agents (observe, evaluate, planning, writer) and tool agents (web_searcher)
    - execute: Implement the workflow logic (observe → evaluate → plan → execute multiple searches → write)
    - WebSearchQuery.format(): Format query (handled automatically by BasePipeline)

    The planning agent generates multiple web search tasks that are executed in parallel.
    All other logic (iteration, tool execution, memory save) is handled by BasePipeline.
    """

    def __init__(self, config):
        """Initialize pipeline with explicit manager agents and tool agent dictionary."""
        super().__init__(config)

        # Initialize context and profiles
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Create manager agents - automatically bound to pipeline with role
        self.observe_agent = ContextAgent.from_profile(self, "observe", llm)
        self.evaluate_agent = ContextAgent.from_profile(self, "evaluate", llm)
        self.planning_agent = ContextAgent.from_profile(self, "web_planning", llm)
        self.writer_agent = ContextAgent.from_profile(self, "writer", llm)

        # Create tool agents as dictionary - automatically bound to pipeline
        tool_agents = [
            "web_searcher",
        ]
        self.tool_agents = {
            f"{name}": ContextAgent.from_profile(self, name, llm)
            for name in tool_agents
        }

    async def execute(self) -> Any:
        """Execute web search workflow - full implementation in one function."""
        self.update_printer("research", "Executing web search workflow...")

        # Iterative loop: observe → evaluate → plan → execute multiple searches
        while self.iteration < self.max_iterations and not self.context.state.complete:
            # Begin iteration with its group
            _, group_id = self.begin_iteration()

            # Get pre-formatted query from state
            query = self.context.state.formatted_query or ""

            # Observe → Evaluate → Plan → Execute Multiple Tools
            observe_output = await self.observe_agent(query, group_id=group_id)
            evaluate_output = await self.evaluate_agent(observe_output, group_id=group_id)

            if not self.context.state.complete:
                planning_output = await self.planning_agent(self._serialize_output(evaluate_output), group_id=group_id)
                await self._execute_tools(planning_output, self.tool_agents, group_id)

            # End iteration with its group
            self.end_iteration(group_id)

            if self.context.state.complete:
                break

        # Final report
        final_group = self.begin_final_report()
        self.update_printer("research", "Web search workflow complete", is_done=True)

        findings = self.context.state.findings_text()
        await self.writer_agent(findings, group_id=final_group)

        self.end_final_report(final_group)
