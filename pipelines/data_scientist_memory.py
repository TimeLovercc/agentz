from __future__ import annotations

from typing import Any, Dict, Optional

from agentz.context.conversation import create_conversation_state
from agentz.context.context import Context
from agentz.profiles.manager.memory import MemoryAgentOutput
from agentz.profiles.manager.evaluate import EvaluateOutput
from agentz.profiles.manager.routing import AgentSelectionPlan
from agentz.profiles.base import load_all_profiles
from agentz.agent.registry import create_agents
from agentz.flow import auto_trace
from pipelines.data_scientist import DataScientistPipeline


class DataScientistMemoryPipeline(DataScientistPipeline):
    """Data scientist pipeline variant that maintains iterative memory compression."""

    def __init__(self, config):
        # Don't call super().__init__ yet - we need to customize
        # Call BasePipeline.__init__ directly
        from pipelines.base import BasePipeline
        BasePipeline.__init__(self, config)

        profiles = load_all_profiles()
        state = create_conversation_state(profiles=profiles)

        # Centralized context engine with state and behaviors (including memory)
        self.context = Context(
            state=state,
            behaviors=["observe", "evaluate", "route", "writer", "memory"],
            config=config,
        )

        # Manager agents with memory support
        manager_agents = {
            "observe_agent": create_agents("observe_agent", config),
            "evaluate_agent": create_agents("evaluate_agent", config),
            "routing_agent": create_agents("routing_agent", config),
            "writer_agent": create_agents("writer_agent", config),
            "memory_agent": create_agents("memory_agent", config),
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
        from agentz.flow import WorkflowOrchestrator, IterationManager
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


    # ------------------------------------------------------------------
    # Override run to include memory step
    # ------------------------------------------------------------------
    @auto_trace
    async def run(self):
        """Execute the data science pipeline with memory compression."""
        from loguru import logger
        import time

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

        # Execute iterative workflow with memory compression
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
                output_model=EvaluateOutput,
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

            # Step 5: Memory compression (conditional)
            if not self.state.complete and bool(self.state.unsummarized_history()):
                await self.flow.run_iteration_step(
                    behavior_name="memory",
                    agent_name="memory_agent",
                    snapshot_builder=lambda ctx: ctx.snapshot("memory"),
                    output_handler=lambda ctx, result: ctx.apply_output("memory", result),
                    output_model=MemoryAgentOutput,
                    span_name="update_memory",
                    span_type="function",
                    printer_key="memory",
                    printer_title="Memory",
                    printer_group_id=iteration_group,
                )

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
