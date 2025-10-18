"""Minimal pipeline for exercising the web_searcher agent directly.

This pipeline mirrors the structure from the data_scientist pipeline but keeps
only the pieces required to invoke the web search tool. It is intended for
debugging and testing the web_searcher profile without the full manager/tool
loop.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from agentz.agent.base import ContextAgent
from agentz.context.context import Context
from agentz.profiles.base import ToolAgentOutput
from pipelines.base import BasePipeline


class WebSearchDebugQuery(BaseModel):
    """Lightweight query wrapper for web search debugging."""

    prompt: str = Field(description="The web search prompt to execute.")

    def format(self) -> str:
        """Render the prompt into a consistent formatted string."""
        return f"Web search task: {self.prompt}"


class SimpleWebPipeline(BasePipeline):
    """Single-agent pipeline that directly runs the web_searcher profile."""

    def __init__(self, config):
        super().__init__(config)

        # Initialize shared context (profiles + conversation state)
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Bind web_searcher agent from loaded profiles
        self.web_search_agent = ContextAgent.from_profile(self, "web_searcher", llm)

    async def execute(self) -> ToolAgentOutput:
        """Run a single iteration that calls the web_searcher agent."""
        self.update_printer("research", "Executing web search debug pipeline...")

        iteration, group_id = self.begin_iteration(title="Web Search Debug")
        logger.debug(f"Starting web search iteration {iteration.index}")

        try:
            query = self.context.state.query

            result = await self.web_search_agent(query, group_id=group_id)

            if self.state:
                self.state.final_report = result.output
                self.state.mark_research_complete()

            self.update_printer("research", "Web search completed", is_done=True)
            logger.debug("Web search agent execution finished successfully.")
            return result
        finally:
            self.end_iteration(group_id)

    async def finalize(self, result: ToolAgentOutput) -> ToolAgentOutput:
        """Return the agent output directly for debugging convenience."""
        return result


__all__ = ["WebSearchDebugQuery", "SimpleWebPipeline"]
