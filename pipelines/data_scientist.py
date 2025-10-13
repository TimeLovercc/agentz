from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentz.agent.base import ContextAgent
from agentz.context.context import Context
from pipelines.base import BasePipeline


class DataScienceQuery(BaseModel):
    """Query model for data science tasks."""
    prompt: str
    data_path: str

    def format(self) -> str:
        """Format data science query."""
        return (
            f"Task: {self.prompt}\n"
            f"Dataset path: {self.data_path}\n"
            "Provide a comprehensive data science workflow"
        )


class DataScientistPipeline(BasePipeline):
    """Data science pipeline using manager-tool pattern.

    This pipeline demonstrates the minimal implementation needed:
    - __init__: Setup agents
    - execute: Implement the workflow logic
    - DataScienceQuery.format(): Format query (handled automatically by BasePipeline)

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
        self.routing_agent = ContextAgent.from_profile(self, "routing", llm)
        self.writer_agent = ContextAgent.from_profile(self, "writer", llm)

        # Create tool agents as dictionary - automatically bound to pipeline
        tool_agents = [
            "data_loader_agent",
            "data_analysis_agent",
            "preprocessing_agent",
            "model_training_agent",
            "evaluation_agent",
            "visualization_agent",
        ]
        self.tool_agents = {
            f"{name}": ContextAgent.from_profile(self, name, llm)
            for name in tool_agents
        }

    async def execute(self) -> Any:
        """Execute data science workflow - full implementation in one function."""
        self.update_printer("research", "Executing research workflow...")

        # Iterative loop: observe → evaluate → route → tools
        while self.iteration < self.max_iterations and not self.context.state.complete:
            # Begin iteration with its group
            _, group_id = self.begin_iteration()

            query = self.context.state.query

            # Observe → Evaluate → Route → Tools
            observe_output = await self.observe_agent(query, group_id=group_id)
            evaluate_output = await self.evaluate_agent(observe_output, group_id=group_id)

            if not self.context.state.complete:
                routing_output = await self.routing_agent(self._serialize_output(evaluate_output), group_id=group_id)
                await self._execute_tools(routing_output, self.tool_agents, group_id)

            # End iteration with its group
            self.end_iteration(group_id)

            if self.context.state.complete:
                break

        # Final report
        final_group = self.begin_final_report()
        self.update_printer("research", "Research workflow complete", is_done=True)

        findings = self.context.state.findings_text()
        await self.writer_agent(findings, group_id=final_group)

        self.end_final_report(final_group)
