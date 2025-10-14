from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel

from agentz.agent.base import ContextAgent
from agentz.context.context import Context
from agentz.profiles.base import ToolAgentOutput
from agentz.profiles.manager.routing import AgentSelectionPlan, AgentTask, RoutingInput
from pipelines.base import BasePipeline


class SimpleQuery(BaseModel):
    """Lightweight query model for the simple pipeline."""
    prompt: str

    def format(self) -> str:
        """Render the query into the routing-friendly prompt."""
        return f"Task: {self.prompt}"


class SimplePipeline(BasePipeline):
    """Single-pass pipeline that routes directly to the web searcher tool."""

    def __init__(self, config):
        super().__init__(config)

        # Initialize shared context (profiles + conversation state)
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Bind agents from registered profiles
        self.routing_agent = ContextAgent.from_profile(self, "routing", llm)
        self.tool_agent = ContextAgent.from_profile(self, "web_searcher", llm)

    async def initialize_pipeline(self, query: Any) -> None:
        """Ensure a SimpleQuery is available when the pipeline starts."""
        if query is None:
            prompt = self.config.prompt or "Analyze the dataset and provide insights."
            query = SimpleQuery(prompt=prompt)
        elif not isinstance(query, SimpleQuery):
            # Coerce arbitrary input into SimpleQuery for consistent formatting
            prompt = getattr(query, "prompt", None) or str(query)
            query = SimpleQuery(prompt=prompt)

        await super().initialize_pipeline(query)

    async def execute(self) -> ToolAgentOutput:
        """Route the query once and execute the web searcher agent."""
        logger.info(f"User prompt: {self.config.prompt}")

        # Start single iteration for structured logging
        _, group_id = self.begin_iteration(title="Single Pass")
        try:
            routing_input = RoutingInput(
                query=self.context.state.query or "",
                gap="Route the query to the web_searcher_agent",
                history=self.context.state.iteration_history(include_current=False) or "",
            )

            routing_plan = await self.routing_agent(routing_input, group_id=group_id)
            task = self._select_task(routing_plan)

            tool_payload = (
                self.tool_agent.input_model(task=task.query)
                if self.tool_agent.input_model
                else task.query
            )

            result = await self.tool_agent(tool_payload, group_id=group_id)

            if self.state:
                self.state.final_report = result.output
                self.state.mark_research_complete()

            logger.info("Simple pipeline completed")
            return result
        finally:
            self.end_iteration(group_id)

    async def finalize(self, result: Any) -> Any:
        """Return the tool output directly."""
        return result

    @staticmethod
    def _select_task(plan: AgentSelectionPlan) -> AgentTask:
        """Pick the first web searcher task from the routing plan."""
        if not plan or not plan.tasks:
            raise ValueError("Routing agent did not return any tasks.")

        for task in plan.tasks:
            if task.agent == "web_searcher_agent":
                return task

        # Fallback: take the first task when a specific agent isn't assigned
        return plan.tasks[0]
