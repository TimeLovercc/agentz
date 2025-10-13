from __future__ import annotations

from loguru import logger

from agentz.agent.base import ContextAgent
from agentz.profiles.base import load_all_profiles
from pipelines.base import BasePipeline


class SimplePipeline(BasePipeline):
    """Simple two-agent pipeline: routing agent + single tool agent."""

    def __init__(self, config):
        super().__init__(config)

        # Load profiles for template rendering
        self.profiles = load_all_profiles()
        llm = self.config.llm.main_model

        # Setup routing agent
        self.routing_agent = ContextAgent.from_profile(self.profiles["routing"], llm)

        # Setup single tool agent
        self.tool_agent = ContextAgent.from_profile(self.profiles["data_loader"], llm)

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
        routing_instructions = self.profiles["routing"].render(
            QUERY=query,
            GAP="Route the query to the data_analysis_agent",
            HISTORY=""
        )
        selection_plan = await self.agent_step(
            agent=self.routing_agent,
            instructions=routing_instructions,
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
    
