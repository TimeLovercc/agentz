from __future__ import annotations

from typing import Any, Dict, Optional

from agentz.agent.base import ContextAgent
from agentz.context.conversation import create_conversation_state
from agentz.context.context import Context
from agentz.profiles.manager.memory import MemoryAgentOutput
from agentz.profiles.manager.evaluate import EvaluateOutput
from agentz.profiles.manager.routing import AgentSelectionPlan
from agentz.profiles.base import load_all_profiles
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
        llm = self.config.llm.main_model

        # Centralized context engine with state and behaviors (including memory)
        self.context = Context(
            state=state,
            behaviors=["observe", "evaluate", "route", "writer", "memory"],
            config=config,
        )

        # Manager agents with memory support
        self.observe_agent = ContextAgent.from_profile(profiles["observe"], llm)
        self.evaluate_agent = ContextAgent.from_profile(profiles["evaluate"], llm)
        self.routing_agent = ContextAgent.from_profile(profiles["routing"], llm)
        self.writer_agent = ContextAgent.from_profile(profiles["writer"], llm)
        self.memory_agent = ContextAgent.from_profile(profiles["memory"], llm)

        # Tool agents for specialized tasks
        self.tool_agents: Dict[str, Any] = {
            f"{name}_agent": ContextAgent.from_profile(profiles[name], llm)
            for name in ["data_loader", "data_analysis", "preprocessing",
                        "model_training", "evaluation", "visualization"]
        }

        # Optional report configuration
        self.report_length: Optional[str] = None
        self.report_instructions: Optional[str] = None

    async def execute(self) -> Any:
        """Execute data science workflow with memory compression.

        This shows the full workflow:
        1. Iterative loop with observe → evaluate → route → execute tools → memory compression
        2. Final report generation with writer agent
        """
        self.update_printer("research", "Executing research workflow...")

        return await self.run_iterative_loop(
            iteration_body=self._iteration_step,
            final_body=self._final_step
        )

    async def _iteration_step(self, iteration, group_id: str):
        """Execute one iteration: observe → evaluate → route → tools → memory."""
        # Step 1: Observe using behavior rendering
        observe_instructions = self.context.render_behavior("observe", self.context.snapshot("observe"))
        observe_result = await self.agent_step(
            agent=self.observe_agent,
            instructions=observe_instructions,
            span_name="observe",
            span_type="agent",
            printer_key="observe",
            printer_title="Observing",
            printer_group_id=group_id,
        )
        observe_output = observe_result.final_output if hasattr(observe_result, 'final_output') else observe_result
        self.context.apply_output("observe", observe_output)
        iteration.observation = self._serialize_output(observe_output)
        self._record_structured_payload(observe_output, context_label="observe")

        # Step 2: Evaluate using behavior rendering
        evaluate_instructions = self.context.render_behavior("evaluate", self.context.snapshot("evaluate"))
        evaluate_result = await self.agent_step(
            agent=self.evaluate_agent,
            instructions=evaluate_instructions,
            span_name="evaluate",
            span_type="agent",
            output_model=EvaluateOutput,
            printer_key="evaluate",
            printer_title="Evaluating",
            printer_group_id=group_id,
        )
        evaluate_output = evaluate_result.final_output if hasattr(evaluate_result, 'final_output') else evaluate_result
        self.context.apply_output("evaluate", evaluate_output)
        self._record_structured_payload(evaluate_output, context_label="evaluate")

        # Step 3: Route to appropriate tools if not complete
        if not self.state.complete:
            routing_instructions = self.context.render_behavior("route", self.context.snapshot("route"))
            routing_result = await self.agent_step(
                agent=self.routing_agent,
                instructions=routing_instructions,
                span_name="routing",
                span_type="agent",
                output_model=AgentSelectionPlan,
                printer_key="routing",
                printer_title="Routing",
                printer_group_id=group_id,
            )
            routing_output = routing_result.final_output if hasattr(routing_result, 'final_output') else routing_result
            self.context.apply_output("route", routing_output)
            self._record_structured_payload(routing_output, context_label="routing")

            # Step 4: Execute selected tools in parallel
            await self._execute_tools(routing_output, self.tool_agents, group_id)

        # Step 5: Memory compression (if needed)
        if not self.state.complete and bool(self.state.unsummarized_history()):
            memory_instructions = self.context.render_behavior("memory", self.context.snapshot("memory"))
            memory_result = await self.agent_step(
                agent=self.memory_agent,
                instructions=memory_instructions,
                span_name="memory",
                span_type="agent",
                output_model=MemoryAgentOutput,
                printer_key="memory",
                printer_title="Memory Compression",
                printer_group_id=group_id,
            )
            memory_output = memory_result.final_output if hasattr(memory_result, 'final_output') else memory_result
            self.context.apply_output("memory", memory_output)
            self._record_structured_payload(memory_output, context_label="memory")

    async def _final_step(self, final_group: str):
        """Generate final report using writer agent."""
        self.update_printer("research", "Research workflow complete", is_done=True)

        writer_instructions = self.context.render_behavior("writer", self.context.snapshot("writer"))
        await self.agent_step(
            agent=self.writer_agent,
            instructions=writer_instructions,
            span_name="writer",
            span_type="agent",
            printer_key="writer",
            printer_title="Writing Report",
            printer_group_id=final_group,
        )
