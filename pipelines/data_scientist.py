from __future__ import annotations



import asyncio
from typing import Dict

from loguru import logger
from pydantic import BaseModel

from agentz.agent.base import ContextAgent
from agentz.profiles.manager.routing import AgentSelectionPlan, AgentTask
from agentz.profiles.base import ToolAgentOutput
from agentz.context.context import Context
from agentz.context.global_memory import global_memory
from agentz.flow import auto_trace
from pipelines.base import BasePipeline


class DataScienceQuery(BaseModel):
    """Query model for data science tasks."""
    prompt: str
    data_path: str



async def execute_tool_plan(
    *,
    pipeline: BasePipeline,
    context: Context,
    tool_agents: Dict[str, object],
    iteration_group: str,
) -> None:
    state = context.state
    # Retrieve route_plan from payloads
    plan = None
    for payload in state.current_iteration.payloads:
        if isinstance(payload, AgentSelectionPlan):
            plan = payload
            break
    if not plan or not plan.tasks:
        return

    state.current_iteration.tools.clear()

    async def run_single(task: AgentTask) -> ToolAgentOutput:
        agent = tool_agents.get(task.agent)
        if agent is None:
            output = ToolAgentOutput(
                output=f"No implementation found for agent {task.agent}",
                sources=[],
            )
            pipeline.update_printer(
                key=f"{iteration_group}:tool:{task.agent}",
                message=f"Completed {task.agent}",
                is_done=True,
                group_id=iteration_group,
            )
            return output

        raw_result = await pipeline.agent_step(
            agent=agent,
            instructions=task.model_dump_json(),
            span_name=task.agent,
            span_type="tool",
            output_model=ToolAgentOutput,
            printer_key=f"tool:{task.agent}",
            printer_title=f"Tool: {task.agent}",
            printer_group_id=iteration_group,
        )

        if isinstance(raw_result, ToolAgentOutput):
            output = raw_result
        elif hasattr(raw_result, "final_output_as"):
            output = raw_result.final_output_as(ToolAgentOutput)
        elif hasattr(raw_result, "final_output"):
            output = ToolAgentOutput(output=str(raw_result.final_output), sources=[])
        else:
            output = ToolAgentOutput(output=str(raw_result), sources=[])

        try:
            context.state.record_payload(output)
        except Exception as exc:
            logger.debug(f"Failed to record tool payload for {task.agent}: {exc}")

        pipeline.update_printer(
            key=f"{iteration_group}:tool:{task.agent}",
            message=f"Completed {task.agent}",
            is_done=True,
            group_id=iteration_group,
        )
        return output

    coroutines = [run_single(task) for task in plan.tasks]
    for coro in asyncio.as_completed(coroutines):
        tool_output = await coro
        state.current_iteration.tools.append(tool_output)


class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(self, config):
        super().__init__(config)

        self.context = Context(["profiles", "states"])
        profiles = self.context.profiles
        llm = self.config.llm.main_model

        # Create manager agents from profiles dict (name auto-derived from key)
        self.observe_agent = ContextAgent.from_profile(profiles["observe"], llm)
        self.evaluate_agent = ContextAgent.from_profile(profiles["evaluate"], llm)
        self.routing_agent = ContextAgent.from_profile(profiles["routing"], llm)
        self.writer_agent = ContextAgent.from_profile(profiles["writer"], llm)

        # Define tool agent names
        tool_agent_names = [
            "data_loader",
            "data_analysis",
            "preprocessing",
            "model_training",
            "evaluation",
            "visualization",
        ]

        # Load all tool agents from profiles
        self.tool_agents = {
            f"{name}_agent": ContextAgent.from_profile(profiles[name], llm)
            for name in tool_agent_names
        }

    def _record_structured_payload(self, value: object, *, context_label: str) -> None:
        if isinstance(value, BaseModel):
            try:
                self.context.state.record_payload(value)
            except Exception as exc:  # pragma: no cover - diagnostic-only
                logger.debug(f"Failed to record payload for {context_label}: {exc}")

    @auto_trace
    async def run(self, query: DataScienceQuery):
        self.iteration = 0

        formatted_query = (
            f"Task: {query.prompt}\n"
            f"Dataset path: {query.data_path}\n"
            "Provide a comprehensive data science workflow"
        )
        self.context.state.set_query(formatted_query)
        self.update_printer("research", "Executing research workflow...")

        loop_index = 0
        while not self.context.state.complete and self._check_constraints():
            iteration = self.context.begin_iteration()
            loop_index = iteration.index
            self.iteration = loop_index
            iteration_group = f"iter-{iteration.index}"

            self.start_group(
                iteration_group,
                title=f"Iteration {iteration.index}",
                border_style="white",
                iteration=iteration.index,
            )

            self.current_printer_group = iteration_group

            observations = await self.observe_agent(input)
            if isinstance(observations, BaseModel):
                iteration.observation = observations.model_dump_json(indent=2)
            elif isinstance(observations, str):
                iteration.observation = observations
            self._record_structured_payload(observations, context_label="observe_agent")

            evaluations = await self.evaluate_agent(observations)
            self._record_structured_payload(evaluations, context_label="evaluate_agent")

            route_plan = None
            if not self.context.state.complete:
                route_plan = await self.routing_agent(evaluations)
                self._record_structured_payload(route_plan, context_label="routing_agent")

            if not self.context.state.complete:
                # Retrieve route_plan from payloads
                plan = None
                if isinstance(route_plan, AgentSelectionPlan):
                    plan = route_plan
                elif route_plan:
                    # Try to find AgentSelectionPlan in payloads
                    for payload in self.context.state.current_iteration.payloads:
                        if isinstance(payload, AgentSelectionPlan):
                            plan = payload
                            break
                if plan and plan.tasks:
                    await execute_tool_plan(
                        pipeline=self,
                        context=self.context,
                        tool_agents=self.tool_agents,
                        iteration_group=iteration_group,
                    )

            self.context.mark_iteration_complete()
            self.end_group(iteration_group, is_done=True)
            self.current_printer_group = None

            if self.context.state.complete:
                break

        final_group = "iter-final"
        self.start_group(
            final_group,
            title="Final Report",
            border_style="white",
        )
        self.current_printer_group = final_group
        await self.writer_agent(self.context.state.findings_text())
        self.end_group(final_group, is_done=True)
        self.current_printer_group = None

        self.update_printer("research", "Research workflow complete", is_done=True)
        logger.info("Research workflow completed")

        if self.context.state.final_report:
            timestamped_report = f"Experiment {self.experiment_id}\n\n{(self.context.state.final_report or '').strip()}"
            global_memory.store(
                key=f"report_{self.experiment_id}",
                value=timestamped_report,
                tags=["research_report"],
            )
            self.update_printer("writer_agent", "Final report generated", is_done=True)

        return self.context.state.final_report
