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


class DataScientistPipeline(BasePipeline):
    """Data science pipeline using manager-tool pattern.

    This pipeline demonstrates the minimal implementation needed:
    - __init__: Setup agents
    - execute: Use run_manager_tool_loop pattern
    - prepare_query_hook: Format query (optional)

    All other logic (iteration, tool execution, memory save) is handled by BasePipeline.
    """

    def __init__(self, config):
        """Initialize pipeline with explicit manager agents and tool agent dictionary."""
        super().__init__(config)

        # Initialize context and profiles
        self.context = Context(["profiles", "states"])
        profiles = self.context.profiles
        llm = self.config.llm.main_model

        # Create manager agents as explicit attributes
        self.observe_agent = ContextAgent.from_profile(profiles["observe"], llm)
        self.evaluate_agent = ContextAgent.from_profile(profiles["evaluate"], llm)
        self.routing_agent = ContextAgent.from_profile(profiles["routing"], llm)
        self.writer_agent = ContextAgent.from_profile(profiles["writer"], llm)

        # Create tool agents as dictionary
        tool_names = [
            "data_loader",
            "data_analysis",
            "preprocessing",
            "model_training",
            "evaluation",
            "visualization",
        ]
        self.tool_agents = {
            f"{name}_agent": ContextAgent.from_profile(profiles[name], llm)
            for name in tool_names
        }

        # Bind all agents to this pipeline for context-aware execution
        self.observe_agent._pipeline = self
        self.observe_agent._role = "observe"

        self.evaluate_agent._pipeline = self
        self.evaluate_agent._role = "evaluate"

        self.routing_agent._pipeline = self
        self.routing_agent._role = "routing"

        self.writer_agent._pipeline = self
        self.writer_agent._role = "writer"

        for agent in self.tool_agents.values():
            agent._pipeline = self

    def prepare_query_hook(self, query: DataScienceQuery) -> str:
        """Format data science query."""
        return (
            f"Task: {query.prompt}\n"
            f"Dataset path: {query.data_path}\n"
            "Provide a comprehensive data science workflow"
        )

    async def execute(self) -> Any:
        """Execute data science workflow - full implementation in one function."""
        self.update_printer("research", "Executing research workflow...")

        # Iterative loop: observe → evaluate → route → tools
        while self.iteration < self.max_iterations and not self.context.state.complete:
            # Begin iteration
            iteration = self.context.begin_iteration()
            group_id = f"iter-{iteration.index}"
            self.iteration = iteration.index
            self.start_group(group_id, title=f"Iteration {iteration.index}", border_style="white", iteration=iteration.index)

            query = self.context.state.query

            # Observe → Evaluate → Route → Tools
            observe_output = await self.observe_agent(query, group_id=group_id)
            evaluate_output = await self.evaluate_agent(observe_output, group_id=group_id)

            if not self.context.state.complete:
                routing_output = await self.routing_agent(self._serialize_output(evaluate_output), group_id=group_id)
                await self._execute_tools(routing_output, self.tool_agents, group_id)

            # End iteration
            self.context.mark_iteration_complete()
            self.end_group(group_id, is_done=True)

            if self.context.state.complete:
                break

        # Final report
        final_group = "iter-final"
        self.start_group(final_group, title="Final Report", border_style="white")
        self.update_printer("research", "Research workflow complete", is_done=True)

        findings = self.context.state.findings_text()
        await self.writer_agent(findings, group_id=final_group)

        self.end_group(final_group, is_done=True)
