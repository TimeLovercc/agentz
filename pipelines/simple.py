from __future__ import annotations

from loguru import logger

from agentz.agents.registry import create_agents
from agentz.flow import auto_trace
from pipelines.base import BasePipeline


class SimplePipeline(BasePipeline):
    """Simple two-agent pipeline: routing agent + single tool agent."""

    def __init__(self, config):
        super().__init__(config)

        # Setup routing agent
        self.routing_agent = create_agents("routing_agent", config)

        # Setup single tool agent
        self.tool_agent = create_agents("data_analysis_agent", config)

    @auto_trace
    async def run(self):
        """Run the simple pipeline with single-pass execution."""
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"User prompt: {self.config.prompt}")

        # Prepare query
        query = self.prepare_query(
            content=f"Task: {self.config.prompt}\n"
                f"Dataset path: {self.config.data_path}\n"
                "Analyze the data and provide insights"
        )

        # Route the task
        # self.update_printer("route", "Routing task to agent...")
        selection_plan = await self.agent_step(
            agent=self.routing_agent,
            instructions=f"""
            QUERY: {query}

            Available agent: data_analysis_agent

            Create a routing plan with a task for the data_analysis_agent.
            """,
            span_name="route_task",
            span_type="agent",
            output_model=self.routing_agent.output_type,
            printer_key="route",
            printer_title="Routing",
        )
        # self.update_printer("route", "Task routed", is_done=True)

        # Execute the tool agent
        task = selection_plan.tasks[0]
        # self.update_printer("tool", f"Executing {task.agent}...")
        
        # import ipdb
        # ipdb.set_trace()

        result = await self.agent_step(
            agent=self.tool_agent,
            instructions=task.model_dump_json(),
            span_name=task.agent,
            span_type="tool",
            printer_key="tool",
            printer_title=f"Tool: {task.agent}",
        )
        # import ipdb
        # ipdb.set_trace()

        # self.update_printer("tool", f"Completed {task.agent}", is_done=True)

        logger.info("Simple pipeline completed")
        return result
    