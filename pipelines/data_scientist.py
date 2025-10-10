from __future__ import annotations



import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional

from loguru import logger
from pydantic import BaseModel

from agentz.agent.agent_base import ContextAgent
from agentz.profiles.manager.evaluate import EvaluateOutput
from agentz.profiles.manager.routing import AgentSelectionPlan, AgentTask
from agentz.profiles.base import ToolAgentOutput
from agentz.context.conversation import ConversationState, ToolExecutionResult
from agentz.context.engine import ContextEngine
from agentz.context.global_memory import global_memory
from agentz.flow import auto_trace
from pipelines.base import BasePipeline
from agentz.profiles.base import load_all_profiles



async def execute_tool_plan(
    *,
    pipeline: BasePipeline,
    context: ContextEngine,
    tool_agents: Dict[str, object],
    iteration_group: str,
) -> None:
    state = context.state
    plan = state.current_iteration.route_plan
    if not plan or not plan.tasks:
        return

    state.current_iteration.tools.clear()
    state.current_iteration.findings.clear()

    async def run_single(task: AgentTask) -> ToolExecutionResult:
        agent = tool_agents.get(task.agent)
        if agent is None:
            output = ToolAgentOutput(
                output=f"No implementation found for agent {task.agent}",
                sources=[],
            )
            result = ToolExecutionResult(task=task, output=output)
            pipeline.update_printer(
                key=f"{iteration_group}:tool:{task.agent}",
                message=f"Completed {task.agent}",
                is_done=True,
                group_id=iteration_group,
            )
            return result

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

        result = ToolExecutionResult(task=task, output=output)
        pipeline.update_printer(
            key=f"{iteration_group}:tool:{task.agent}",
            message=f"Completed {task.agent}",
            is_done=True,
            group_id=iteration_group,
        )
        return result

    coroutines = [run_single(task) for task in plan.tasks]
    for coro in asyncio.as_completed(coroutines):
        tool_result = await coro
        context.record_tool_execution(tool_result)
        output_value = getattr(tool_result.output, "output", None)
        if output_value:
            context.add_finding(output_value)


class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(self, config):
        super().__init__(config)
        
        profiles = load_all_profiles()
        output_models = [
            profile.output_schema
            for profile in profiles.values()
            if getattr(profile, "output_schema", None)
        ]
        state = ConversationState(extra_output_models=output_models or None)

        self.context = ContextEngine(
            state=state,
            config=config,
        )

        self.observe_agent = ContextAgent(profiles["observe_profile"], llm = config.llm)
        self.evaluate_agent = ContextAgent(profiles["evaluate_profile"], llm = config.llm)
        self.routing_agent = ContextAgent(profiles["route_profile"], llm = config.llm)
        self.writer_agent = ContextAgent(profiles["writer_profile"], llm = config.llm)

        self.tool_agents = ContextAgent(
            {
                "data_loader_agent": {"profile": "data_loader", "llm": config.llm},
                "data_analysis_agent": {"profile": "data_analysis", "llm": config.llm},
                "preprocessing_agent": {"profile": "preprocessing", "llm": config.llm},
                "model_training_agent": {"profile": "model_training", "llm": config.llm},
                "evaluation_agent": {"profile": "evaluation", "llm": config.llm},
                "visualization_agent": {"profile": "visualization", "llm": config.llm},
                "code_generation_agent": {"profile": "code_generation", "llm": config.llm},
            },
        )

    def _record_structured_payload(self, value: object, *, context_label: str) -> None:
        if isinstance(value, BaseModel):
            try:
                self.context.state.record_payload(value)
            except Exception as exc:  # pragma: no cover - diagnostic-only
                logger.debug(f"Failed to record payload for {context_label}: {exc}")

    @auto_trace
    async def run(self, query: Optional[str] = None):
        logger.info(f"Data path: {query.data_path}")
        logger.info(f"User prompt: {query.prompt}")
        self.iteration = 0
        state = self.context.state

        formatted_query = (
            f"Task: {query.prompt}\n"
            f"Dataset path: {query.data_path}\n"
            "Provide a comprehensive data science workflow"
        )
        self.context.state.set_query(formatted_query)
        self.update_printer("research", "Executing research workflow...")
        self.start_time = time.time()
        self.context.state.start_timer()

        for agent in (self.observe_agent, self.evaluate_agent, self.routing_agent, self.writer_agent):
            agent.bind(self)

        loop_index = 0
        while not state.complete and self._check_constraints():
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
            if isinstance(evaluations, EvaluateOutput):
                iteration.evaluation = evaluations
            self._record_structured_payload(evaluations, context_label="evaluate_agent")

            route_plan = None
            if not state.complete:
                route_plan = await self.routing_agent(evaluations)
                if isinstance(route_plan, AgentSelectionPlan):
                    iteration.route_plan = route_plan
                self._record_structured_payload(route_plan, context_label="routing_agent")

            if not state.complete:
                plan = state.current_iteration.route_plan
                if not plan and isinstance(route_plan, AgentSelectionPlan):
                    plan = route_plan
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

            if state.complete:
                break

        final_group = "iter-final"
        self.start_group(
            final_group,
            title="Final Report",
            border_style="white",
        )
        self.current_printer_group = final_group
        await self.writer_agent(state.findings_text())
        self.end_group(final_group, is_done=True)
        self.current_printer_group = None

        self.update_printer("research", "Research workflow complete", is_done=True)
        logger.info("Research workflow completed")

        if state.final_report:
            timestamped_report = f"Experiment {self.experiment_id}\n\n{(state.final_report or '').strip()}"
            global_memory.store(
                key=f"report_{self.experiment_id}",
                value=timestamped_report,
                tags=["research_report"],
            )
            self.update_printer("writer_agent", "Final report generated", is_done=True)

        return state.final_report
