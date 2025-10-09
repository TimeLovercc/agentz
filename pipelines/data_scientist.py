from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from agentz.agents.manager_agents.evaluate_agent import KnowledgeGapOutput
from agentz.agents.manager_agents.routing_agent import AgentSelectionPlan, AgentTask
from agentz.agents.registry import ToolAgentOutput, create_agents
from agentz.flow import auto_trace
from agentz.memory.conversation import ConversationState, ToolExecutionResult
from agentz.memory.global_memory import global_memory
from pipelines.base import BasePipeline
from pipelines.flow_runner import FlowNode, FlowRunner, IterationFlow
from agentz.flow.runtime_objects import AgentCapability, PipelineContext


class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(self, config):
        super().__init__(config)

        # Centralised conversation state for structured agent IO
        self.conversation = ConversationState(
            query="",
            data_path=self.config.data_path,
            max_iterations=self.max_iterations,
            max_minutes=self.max_time_minutes,
        )

        # Setup manager agents
        self.agents: Dict[str, AgentCapability] = {
            "observe_agent": AgentCapability("observe_agent", create_agents("observe_agent", config)),
            "evaluate_agent": AgentCapability("evaluate_agent", create_agents("evaluate_agent", config)),
            "routing_agent": AgentCapability("routing_agent", create_agents("routing_agent", config)),
            "writer_agent": AgentCapability("writer_agent", create_agents("writer_agent", config)),
        }

        # Setup specialised tool agents
        tool_agent_names = [
            "data_loader_agent",
            "data_analysis_agent",
            "preprocessing_agent",
            "model_training_agent",
            "evaluation_agent",
            "visualization_agent",
        ]
        self.tool_agents: Dict[str, Any] = create_agents(tool_agent_names, config)

        # Build declarative flow and runner
        self.iteration_flow = IterationFlow(
            nodes=self._build_iteration_nodes(),
            loop_condition=self._should_continue_loop,
        )
        self.final_nodes = self._build_final_nodes()
        # Optional report configuration
        self.report_length: Optional[str] = None
        self.report_instructions: Optional[str] = None

        self.flow_runner = FlowRunner(
            self,
            agents=self.agents,
            iteration_flow=self.iteration_flow,
            final_nodes=self.final_nodes,
        )
        self.pipeline_context = PipelineContext(self.conversation)
        self._register_context_bindings()

    # ------------------------------------------------------------------
    # Flow configuration
    # ------------------------------------------------------------------
    def _register_context_bindings(self) -> None:
        """Register snapshot and output handlers with the context engine."""
        ctx = self.pipeline_context
        ctx.register_snapshot("observe_agent", self._observation_snapshot)
        ctx.register_snapshot("evaluate_agent", self._evaluation_snapshot)
        ctx.register_snapshot("routing_agent", self._routing_snapshot)
        ctx.register_snapshot("writer_agent", self._writer_snapshot)

        ctx.register_output_handler("observe_agent", self._apply_observation_output)
        ctx.register_output_handler("evaluate_agent", self._apply_evaluation_output)
        ctx.register_output_handler("routing_agent", self._apply_routing_output)
        ctx.register_output_handler("writer_agent", self._apply_writer_output)

    def _observation_snapshot(self, state: ConversationState) -> Dict[str, Any]:
        history = state.iteration_history()
        if not history:
            history = "No previous actions, findings or thoughts available."
        return {
            "ITERATION": state.current_iteration.index,
            "QUERY": state.query,
            "HISTORY": history,
        }

    def _evaluation_snapshot(self, state: ConversationState) -> Dict[str, Any]:
        history = state.iteration_history(include_current=True)
        if not history:
            history = "No previous actions, findings or thoughts available."
        return {
            "ITERATION": state.current_iteration.index,
            "ELAPSED_MINUTES": f"{state.elapsed_minutes():.2f}",
            "MAX_MINUTES": self.max_time_minutes,
            "QUERY": state.query,
            "HISTORY": history,
        }

    def _routing_snapshot(self, state: ConversationState) -> Dict[str, Any]:
        history = state.iteration_history(include_current=True)
        if not history:
            history = "No previous actions, findings or thoughts available."
        gap = state.current_iteration.selected_gap or "No specific gap provided."
        return {
            "QUERY": state.query,
            "GAP": gap,
            "HISTORY": history,
        }

    def _writer_snapshot(self, state: ConversationState) -> Dict[str, Any]:
        findings_text = state.findings_text() or "No findings available yet."
        guidelines_chunks = []
        if self.report_length:
            guidelines_chunks.append(f"* The full response should be approximately {self.report_length}.")
        if self.report_instructions:
            guidelines_chunks.append(f"* {self.report_instructions}")
        guidelines_block = ""
        if guidelines_chunks:
            guidelines_block = "\n\nGUIDELINES:\n" + "\n".join(guidelines_chunks)

        return {
            "GUIDELINES_BLOCK": guidelines_block,
            "USER_PROMPT": self.config.prompt,
            "DATA_PATH": self.config.data_path or "Not provided",
            "FINDINGS": findings_text,
        }

    def _apply_observation_output(self, state: ConversationState, result: Any) -> None:
        final_output = getattr(result, "final_output", None)
        if final_output is None:
            final_output = str(result)
        state.current_iteration.observation = final_output

    def _apply_evaluation_output(self, state: ConversationState, result: KnowledgeGapOutput) -> None:
        state.current_iteration.evaluation = result
        if result.research_complete:
            state.mark_research_complete()
            return
        if result.outstanding_gaps:
            state.current_iteration.selected_gap = result.outstanding_gaps[0]

    def _apply_routing_output(self, state: ConversationState, result: AgentSelectionPlan) -> None:
        state.current_iteration.route_plan = result

    def _apply_writer_output(self, state: ConversationState, result: Any) -> None:
        final_output = getattr(result, "final_output", None)
        if final_output is None:
            final_output = str(result)
        state.final_report = final_output

    def _build_iteration_nodes(self) -> List[FlowNode]:
        return [
            FlowNode(
                name="observe",
                agent_key="observe_agent",
                profile="observe_agent",
                template="research_iteration",
                input_builder=self._build_observation_payload,
                output_handler=self._handle_observation_output,
                span_name="generate_observations",
                span_type="function",
                printer_key="observe",
                printer_title="Observations",
            ),
            FlowNode(
                name="evaluate",
                agent_key="evaluate_agent",
                profile="evaluate_agent",
                template="research_iteration",
                input_builder=self._build_evaluation_payload,
                output_model=KnowledgeGapOutput,
                output_handler=self._handle_evaluation_output,
                span_name="evaluate_research_state",
                span_type="function",
                printer_key="evaluate",
                printer_title="Evaluation",
            ),
            FlowNode(
                name="route",
                agent_key="routing_agent",
                profile="routing_agent",
                template="research_iteration",
                input_builder=self._build_routing_payload,
                output_model=AgentSelectionPlan,
                output_handler=self._handle_routing_output,
                span_name="route_tasks",
                span_type="tool",
                printer_key="route",
                printer_title="Routing",
                condition=lambda ctx: not ctx.state.complete,
            ),
            FlowNode(
                name="tools",
                custom_runner=self._execute_tool_tasks,
                printer_key="tools",
                printer_title="Tools",
                condition=self._has_pending_tools,
            ),
        ]

    def _build_final_nodes(self) -> List[FlowNode]:
        return [
            FlowNode(
                name="writer",
                agent_key="writer_agent",
                profile="writer_agent",
                template="final_report",
                input_builder=self._build_writer_payload,
                output_handler=self._handle_writer_output,
                span_name="writer_agent",
                span_type="agent",
                printer_key="writer",
                printer_title="Writer",
            )
        ]

    # ------------------------------------------------------------------
    # Run entry point
    # ------------------------------------------------------------------
    @auto_trace
    async def run(self):
        """Execute the data science pipeline using the declarative flow."""
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
        self.conversation.set_query(query)

        self.update_printer("research", "Executing research workflow...")
        self.start_time = time.time()
        self.conversation.start_timer()

        # Execute the flow
        await self.flow_runner.execute(self.pipeline_context)

        # Finalise persisted memory and artefacts
        self.update_printer("research", "Research workflow complete", is_done=True)
        logger.info("Research workflow completed")

        if self.conversation.final_report:
            await self._finalise_research(self.conversation.final_report)

        return self.conversation.final_report

    # ------------------------------------------------------------------
    # Flow condition helpers
    # ------------------------------------------------------------------
    def _should_continue_loop(self, context: PipelineContext) -> bool:
        if context.state.complete:
            return False
        return self._check_constraints()

    def _has_pending_tools(self, context: PipelineContext) -> bool:
        state = context.state
        if state.complete:
            return False
        iteration = state.current_iteration
        return bool(iteration.route_plan and iteration.route_plan.tasks)

    # ------------------------------------------------------------------
    # Input builders
    # ------------------------------------------------------------------
    def _build_observation_payload(self, context: PipelineContext) -> Dict[str, Any]:
        return self._observation_snapshot(context.state)

    def _build_evaluation_payload(self, context: PipelineContext) -> Dict[str, Any]:
        return self._evaluation_snapshot(context.state)

    def _build_routing_payload(self, context: PipelineContext) -> Dict[str, Any]:
        return self._routing_snapshot(context.state)

    def _build_writer_payload(self, context: PipelineContext) -> Dict[str, Any]:
        return self._writer_snapshot(context.state)

    # ------------------------------------------------------------------
    # Output handlers
    # ------------------------------------------------------------------
    def _handle_observation_output(self, context: PipelineContext, result: Any) -> None:
        context.apply_output("observe_agent", result)

    def _handle_evaluation_output(self, context: PipelineContext, result: KnowledgeGapOutput) -> None:
        context.apply_output("evaluate_agent", result)

    def _handle_routing_output(self, context: PipelineContext, result: AgentSelectionPlan) -> None:
        context.apply_output("routing_agent", result)

    def _handle_writer_output(self, context: PipelineContext, result: Any) -> None:
        context.apply_output("writer_agent", result)

    # ------------------------------------------------------------------
    # Tool execution runner
    # ------------------------------------------------------------------
    async def _execute_tool_tasks(self, pipeline_context: PipelineContext, exec_ctx) -> None:
        state = pipeline_context.state
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
                printer_key=f"{exec_ctx.iteration_group_id}:tool:{task.agent}"
                if exec_ctx.iteration_group_id
                else f"tool:{task.agent}",
                printer_title=f"Tool: {task.agent}",
                printer_group_id=exec_ctx.iteration_group_id,
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
            printer_key = (
                f"{exec_ctx.iteration_group_id}:tool:{result.task.agent}"
                if exec_ctx.iteration_group_id
                else f"tool:{result.task.agent}"
            )
            self.update_printer(
                key=printer_key,
                message=f"Completed {result.task.agent}",
                is_done=True,
                group_id=exec_ctx.iteration_group_id,
            )
            results.append(result)

        for tool_result in results:
            pipeline_context.engine.record_tool_execution(tool_result)
            output_value = getattr(tool_result.output, "output", None)
            if output_value:
                pipeline_context.engine.add_finding(output_value)

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
