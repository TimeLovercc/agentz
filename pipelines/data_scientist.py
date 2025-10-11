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

    def prepare_query_hook(self, query: DataScienceQuery) -> str:
        """Format data science query."""
        return (
            f"Task: {query.prompt}\n"
            f"Dataset path: {query.data_path}\n"
            "Provide a comprehensive data science workflow"
        )

    async def execute(self) -> Any:
        """Execute data science workflow with explicit iteration and tool execution.

        This shows the full workflow:
        1. Iterative loop with observe → evaluate → route → execute tools
        2. Final report generation with writer agent
        """
        self.update_printer("research", "Executing research workflow...")

        return await self.run_iterative_loop(
            iteration_body=self._iteration_step,
            final_body=self._final_step
        )

    async def _iteration_step(self, iteration, group_id: str):
        """Execute one iteration: observe → evaluate → route → tools."""
        query = self.context.state.query

        # Step 1: Observe current state and gather information
        observe_result = await self.agent_step(
            agent=self.observe_agent,
            instructions=query,
            span_name="observe",
            span_type="agent",
            printer_key="observe",
            printer_title="Observing",
            printer_group_id=group_id,
        )
        observe_output = observe_result.final_output if hasattr(observe_result, 'final_output') else observe_result
        iteration.observation = self._serialize_output(observe_output)
        self._record_structured_payload(observe_output, context_label="observe")

        # Step 2: Evaluate progress and determine next actions
        # Pass serialized observe output to evaluate agent
        evaluate_result = await self.agent_step(
            agent=self.evaluate_agent,
            instructions=self._serialize_output(observe_output),
            span_name="evaluate",
            span_type="agent",
            printer_key="evaluate",
            printer_title="Evaluating",
            printer_group_id=group_id,
        )
        evaluate_output = evaluate_result.final_output if hasattr(evaluate_result, 'final_output') else evaluate_result
        self._record_structured_payload(evaluate_output, context_label="evaluate")

        # Step 3: Route to appropriate tools if not complete
        if not self.context.state.complete:
            # Pass serialized evaluate output to routing agent
            routing_result = await self.agent_step(
                agent=self.routing_agent,
                instructions=self._serialize_output(evaluate_output),
                span_name="routing",
                span_type="agent",
                printer_key="routing",
                printer_title="Routing",
                printer_group_id=group_id,
            )
            routing_output = routing_result.final_output if hasattr(routing_result, 'final_output') else routing_result
            self._record_structured_payload(routing_output, context_label="routing")

            # Step 4: Execute selected tools in parallel
            await self._execute_tools(routing_output, self.tool_agents, group_id)

    async def _final_step(self, final_group: str):
        """Generate final report using writer agent."""
        self.update_printer("research", "Research workflow complete", is_done=True)

        findings = self.context.state.findings_text()
        await self.agent_step(
            agent=self.writer_agent,
            instructions=findings,
            span_name="writer",
            span_type="agent",
            printer_key="writer",
            printer_title="Writing Report",
            printer_group_id=final_group,
        )
