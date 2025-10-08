from __future__ import annotations

from loguru import logger

from agentz.agents.registry import create_agents
from agentz.flow import auto_trace
from pipelines.base import BasePipeline
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser
from agentz.agents.base import ResearchAgent as Agent
from agents.mcp import MCPServer, MCPServerStdio, MCPServerSse


async def get_browser_server():
    async with MCPServerStdio(
        name = "Browser",
        params = {
            "command": "npx",
            "args": ["-y", "@browsermcp/mcp@latest"]
        }
    ) as server:
        return server

class SimpleBrowserPipeline(BasePipeline):
    """Simple two-agent pipeline: routing agent + single tool agent."""

    def __init__(self, config):
        super().__init__(config)

        # Setup routing agent
        self.routing_agent = create_agents("routing_agent", config)

        # Setup single tool agent
        self.tool_agent = None

    @auto_trace
    async def run(self):
        """Run the simple pipeline with single-pass execution to validate the browser agent."""
        logger.info(f"User prompt: {self.config.prompt}")

        async with MCPServerStdio(
            name = "Browser",
            params = {
                "command": "npx",
                "args": ["@browsermcp/mcp@latest"]
            }
        ) as server:
        # await server.connect()
            self.tool_agent = Agent(
                name="Browser",
                instructions=f"""
                    You are a browser agent. Your task is to interact with the browser following the instructions provided.
                    """,
                mcp_servers=[server],
                model=self.config.llm.main_model,
                # output_type=ToolAgentOutput if model_supports_json_and_tool_calls(self.config.llm.main_model) else None,
                # output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(self.config.llm.main_model) else None
            )

        

            # Prepare query
            query = self.prepare_query(
                content=f"Task: {self.config.prompt}\n"
            )

            # Route the task
            # self.update_printer("route", "Routing task to agent...")
            selection_plan = await self.agent_step(
                agent=self.routing_agent,
                instructions=f"""
                QUERY: {query}

                Available agent: browser_agent

                Create a routing plan with a task for the browser_agent.
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
            print(task)
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

            logger.info("Simple browser pipeline completed")
            return result
        