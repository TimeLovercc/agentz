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
        """Initialize pipeline with manager and tool agents."""
        super().__init__(config)

        # Initialize context and profiles
        self.context = Context(["profiles", "states"])
        profiles = self.context.profiles
        llm = self.config.llm.main_model

        # Create manager agents
        self.manager_agents = {
            "observe": ContextAgent.from_profile(profiles["observe"], llm),
            "evaluate": ContextAgent.from_profile(profiles["evaluate"], llm),
            "routing": ContextAgent.from_profile(profiles["routing"], llm),
            "writer": ContextAgent.from_profile(profiles["writer"], llm),
        }

        # Create tool agents
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

    def prepare_query_hook(self, query: DataScienceQuery) -> str:
        """Format data science query."""
        return (
            f"Task: {query.prompt}\n"
            f"Dataset path: {query.data_path}\n"
            "Provide a comprehensive data science workflow"
        )

    async def execute(self) -> Any:
        """Execute data science workflow using manager-tool pattern.

        Uses the base class run_manager_tool_loop which handles:
        - Iteration management
        - Observe → Evaluate → Route workflow
        - Tool execution
        - Final report generation
        """
        return await self.run_manager_tool_loop(
            manager_agents=self.manager_agents,
            tool_agents=self.tool_agents,
            workflow=["observe", "evaluate", "routing"]
        )
