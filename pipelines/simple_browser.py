from __future__ import annotations

from loguru import logger

from agentz.profiles.registry import create_agents
from agentz.flow import auto_trace
from agentz.context.behavior_profiles import runtime_prompts
from pipelines.base import BasePipeline
from agentz.profiles.agent_base import ResearchAgent as Agent
from agentz.mcp.manager import MCPManager, MCPServerSpec
from agentz.mcp.patches import apply_browsermcp_close_patch


class NullReporter:
    """No-op reporter to disable report generation for this pipeline."""

    def start(self, *_args, **_kwargs):  # noqa: D401 - intentionally blank
        return None

    def set_final_result(self, _result):  # noqa: D401 - intentionally blank
        return None

    def finalize(self):  # noqa: D401 - intentionally blank
        return None

    def print_terminal_report(self):  # noqa: D401 - intentionally blank
        return None

    def record_status_update(self, **_kwargs):  # noqa: D401 - intentionally blank
        return None

    def record_panel(self, **_kwargs):  # noqa: D401 - intentionally blank
        return None

    def record_group_start(self, **_kwargs):  # noqa: D401 - intentionally blank
        return None

    def record_group_end(self, **_kwargs):  # noqa: D401 - intentionally blank
        return None

    def record_agent_step_start(self, **_kwargs):  # noqa: D401 - intentionally blank
        return None

    def record_agent_step_end(self, **_kwargs):  # noqa: D401 - intentionally blank
        return None


class SimpleBrowserPipeline(BasePipeline):
    """Simple two-agent pipeline: routing agent + single tool agent."""

    def __init__(self, config):
        super().__init__(config)

        # Ensure Browser MCP stdio server has patched close handler to avoid recursion errors.
        apply_browsermcp_close_patch()

        # Disable report generation for this pipeline
        self.reporter = NullReporter()

        # Setup routing agent
        self.routing_agent = create_agents("routing_agent", config)

        # Setup single tool agent
        self.tool_agent = None
        mcp_config = getattr(self.config, "mcp", None)
        self.mcp_manager = MCPManager.from_config(mcp_config)
        self.mcp_manager.ensure_server(
            "browser",
            MCPServerSpec(
                type="stdio",
                options={
                    "name": "Browser",
                    "params": {"command": "npx", "args": ["-y", "@browsermcp/mcp@latest"]},
                },
            ),
        )

    @auto_trace
    async def run(self):
        """Run the simple pipeline with single-pass execution to validate the browser agent."""
        logger.info(f"User prompt: {self.config.prompt}")

        async with self.mcp_manager.session() as mcp_session:
            server = await mcp_session.get_server("browser")
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
                instructions=runtime_prompts.render(
                    "routing_agent",
                    "single_agent_routing",
                    QUERY=query,
                    AVAILABLE_AGENT="browser_agent",
                ),
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
        
