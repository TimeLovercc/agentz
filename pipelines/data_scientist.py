from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from agentz.agents.manager_agents.evaluate_agent import KnowledgeGapOutput
from agentz.agents.manager_agents.routing_agent import AgentSelectionPlan, AgentTask
from agentz.agents.registry import ToolAgentOutput, create_agents
from agentz.context.behavior_profiles import behavior_registry
from agentz.context.engine import ContextEngine
from agentz.flow import auto_trace, IterationManager, WorkflowOrchestrator
from agentz.context.conversation import ConversationState, ToolExecutionResult
from agentz.context.global_memory import global_memory
from pipelines.base import BasePipeline


class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(self, config):
        super().__init__(config)

        # Centralized context engine with state and behaviors
        self.context = ContextEngine(
            state=ConversationState(
                query="",
                data_path=self.config.data_path,
                max_iterations=self.max_iterations,
                max_minutes=self.max_time_minutes,
            ),
            behaviors=["observe", "evaluate", "route", "writer"]
        )

        # Manager agents (created on-demand via config)
        manager_agents = {
            "observe_agent": create_agents("observe_agent", config),
            "evaluate_agent": create_agents("evaluate_agent", config),
            "routing_agent": create_agents("routing_agent", config),
            "writer_agent": create_agents("writer_agent", config),
        }

        # Tool agents for specialized tasks
        self.tool_agents: Dict[str, Any] = create_agents([
            "data_loader_agent",
            "data_analysis_agent",
            "preprocessing_agent",
            "model_training_agent",
            "evaluation_agent",
            "visualization_agent",
        ], config)

        # Workflow orchestration
        self.flow = WorkflowOrchestrator(
            engine=self.context,
            agent_registry=manager_agents,
            pipeline=self,
        )

        self.iteration_manager = IterationManager(
            engine=self.context,
            loop_condition=self._should_continue_loop,
            pipeline=self,
        )

        # Optional report configuration
        self.report_length: Optional[str] = None
        self.report_instructions: Optional[str] = None

    @property
    def state(self) -> ConversationState:
        return self.context.state

    # ------------------------------------------------------------------
    # Run entry point
    # ------------------------------------------------------------------
    @auto_trace
    async def run(self):
        """Execute the data science pipeline using compositional workflow."""
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"User prompt: {self.config.prompt}")
        self.iteration = 0

        # Prepare the structured query and initialise state timing
        query = self.prepare_query(
            content=(
                f"Task: {self.config.prompt}\n"
                f"Dataset path: {self.config.data_path}\n"
                "Provide a comprehensive data science workflow"
            )
        )
        self.state.set_query(query)

        self.update_printer("research", "Executing research workflow...")
        self.start_time = time.time()
        self.state.start_timer()

        # Execute iterative workflow
        while self.iteration_manager.should_continue():
            iteration, iteration_group = self.iteration_manager.begin_iteration()

            # Step 1: Observe
            await self.flow.run_iteration_step(
                behavior_name="observe",
                agent_name="observe_agent",
                snapshot_builder=lambda ctx: ctx.snapshot("observe"),
                output_handler=lambda ctx, result: ctx.apply_output("observe", result),
                span_name="generate_observations",
                span_type="function",
                printer_key="observe",
                printer_title="Observations",
                printer_group_id=iteration_group,
            )

            # Step 2: Evaluate
            await self.flow.run_iteration_step(
                behavior_name="evaluate",
                agent_name="evaluate_agent",
                snapshot_builder=lambda ctx: ctx.snapshot("evaluate"),
                output_handler=lambda ctx, result: ctx.apply_output("evaluate", result),
                output_model=KnowledgeGapOutput,
                span_name="evaluate_research_state",
                span_type="function",
                printer_key="evaluate",
                printer_title="Evaluation",
                printer_group_id=iteration_group,
            )

            # Step 3: Route (conditional)
            if not self.state.complete:
                await self.flow.run_iteration_step(
                    behavior_name="route",
                    agent_name="routing_agent",
                    snapshot_builder=lambda ctx: ctx.snapshot("route"),
                    output_handler=lambda ctx, result: ctx.apply_output("route", result),
                    output_model=AgentSelectionPlan,
                    span_name="route_tasks",
                    span_type="tool",
                    printer_key="route",
                    printer_title="Routing",
                    printer_group_id=iteration_group,
                )

            # Step 4: Execute tools (conditional)
            if self._has_pending_tools():
                await self._execute_tool_tasks(iteration_group)

            self.iteration_manager.end_iteration(iteration_group)

            if self.state.complete:
                break

        # Final report generation
        final_group = self.iteration_manager.start_final_group()
        await self.flow.run_iteration_step(
            behavior_name="writer",
            agent_name="writer_agent",
            snapshot_builder=lambda ctx: ctx.snapshot("writer"),
            output_handler=lambda ctx, result: ctx.apply_output("writer", result),
            span_name="writer_agent",
            span_type="agent",
            printer_key="writer",
            printer_title="Writer",
            printer_group_id=final_group,
        )
        self.iteration_manager.end_final_group(final_group)

        # Finalise persisted memory and artefacts
        self.update_printer("research", "Research workflow complete", is_done=True)
        logger.info("Research workflow completed")

        if self.state.final_report:
            await self._finalise_research(self.state.final_report)

        return self.state.final_report

    # ------------------------------------------------------------------
    # Flow condition helpers
    # ------------------------------------------------------------------
    def _should_continue_loop(self, engine: ContextEngine) -> bool:
        if engine.state.complete:
            return False
        return self._check_constraints()

    def _has_pending_tools(self) -> bool:
        state = self.state
        if state.complete:
            return False
        iteration = state.current_iteration
        return bool(iteration.route_plan and iteration.route_plan.tasks)

    # ------------------------------------------------------------------
    # Tool execution runner
    # ------------------------------------------------------------------
    async def _execute_tool_tasks(self, iteration_group: str) -> None:
        state = self.state
        iteration = state.current_iteration
        plan = iteration.route_plan
        if not plan or not plan.tasks:
            return

        iteration.tools.clear()
        iteration.findings.clear()

        async def _run_single(task: AgentTask) -> ToolExecutionResult:
            agent = self.tool_agents.get(task.agent)
            if agent is None:
                output = ToolAgentOutput(
                    output=f"No implementation found for agent {task.agent}",
                    sources=[],
                )
                return ToolExecutionResult(task=task, output=output)

            result = await self.agent_step(
                agent=agent,
                instructions=task.model_dump_json(),
                span_name=task.agent,
                span_type="tool",
                output_model=ToolAgentOutput,
                printer_key=f"tool:{task.agent}",
                printer_title=f"Tool: {task.agent}",
                printer_group_id=iteration_group,
            )

            output = result
            if isinstance(result, ToolAgentOutput):
                output = result
            elif hasattr(result, "final_output_as"):
                output = result.final_output_as(ToolAgentOutput)
            elif hasattr(result, "final_output"):
                output = ToolAgentOutput(output=str(result.final_output), sources=[])
            else:
                output = ToolAgentOutput(output=str(result), sources=[])

            return ToolExecutionResult(task=task, output=output)

        tasks = [_run_single(task) for task in plan.tasks]
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            printer_key = f"{iteration_group}:tool:{result.task.agent}"
            self.update_printer(
                key=printer_key,
                message=f"Completed {result.task.agent}",
                is_done=True,
                group_id=iteration_group,
            )
            results.append(result)

        for tool_result in results:
            self.context.record_tool_execution(tool_result)
            output_value = getattr(tool_result.output, "output", None)
            if output_value:
                self.context.add_finding(output_value)

    # ------------------------------------------------------------------
    # Finalisation helpers
    # ------------------------------------------------------------------
    async def _finalise_research(self, research_report: str) -> None:
        timestamped_report = f"Experiment {self.experiment_id}\n\n{(research_report or '').strip()}"
        global_memory.store(
            key=f"report_{self.experiment_id}",
            value=timestamped_report,
            tags=["research_report"],
        )
        logger.info(f"Stored research report with experiment_id {self.experiment_id}")
        self.update_printer("writer_agent", "Final report generated", is_done=True)
