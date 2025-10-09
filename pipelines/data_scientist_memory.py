from __future__ import annotations

from typing import Any, Dict, List

from agentz.context.conversation import ConversationState
from agentz.agents.manager_agents.memory_agent import MemoryAgentOutput
from agentz.agents.registry import create_agents
from pipelines.data_scientist import DataScientistPipeline
from pipelines.flow_runner import FlowNode, FlowRunner, IterationFlow
from agentz.flow.runtime_objects import AgentCapability, PipelineContext


class DataScientistMemoryPipeline(DataScientistPipeline):
    """Data scientist pipeline variant that maintains iterative memory compression."""

    def __init__(self, config):
        # Initialise base pipeline (sets up conversation, agents, and runner)
        super().__init__(config)

        # Augment manager agents with memory agent
        self.agents["memory_agent"] = AgentCapability("memory_agent", create_agents("memory_agent", config))

        # Rebuild flow with memory node included
        self.iteration_flow = IterationFlow(
            nodes=self._build_iteration_nodes(),
            loop_condition=self._should_continue_loop,
        )
        self.final_nodes = self._build_final_nodes()
        self.flow_runner = FlowRunner(
            self,
            agents=self.agents,
            iteration_flow=self.iteration_flow,
            final_nodes=self.final_nodes,
        )
        self._register_memory_context_bindings()

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------
    def _build_observation_payload(self, context: PipelineContext) -> Dict[str, Any]:
        return self._observation_snapshot(context.state)

    def _build_evaluation_payload(self, context: PipelineContext) -> Dict[str, Any]:
        return self._evaluation_snapshot(context.state)

    def _build_routing_payload(self, context: PipelineContext) -> Dict[str, Any]:
        return self._routing_snapshot(context.state)

    # ------------------------------------------------------------------
    # Flow configuration with memory node
    # ------------------------------------------------------------------
    def _build_iteration_nodes(self) -> List[FlowNode]:
        nodes = super()._build_iteration_nodes()
        nodes.append(
            FlowNode(
                name="memory",
                agent_key="memory_agent",
                profile="memory_agent",
                template="compression_iteration",
                input_builder=self._build_memory_payload,
                output_model=MemoryAgentOutput,
                output_handler=self._handle_memory_output,
                span_name="update_memory",
                span_type="function",
                printer_key="memory",
                printer_title="Memory",
                condition=self._should_run_memory_node,
            )
        )
        return nodes

    def _build_memory_payload(self, context: PipelineContext) -> Dict[str, Any]:
        return self._memory_snapshot(context.state)

    def _handle_memory_output(self, context: PipelineContext, result: MemoryAgentOutput) -> None:
        context.apply_output("memory_agent", result)

    def _should_run_memory_node(self, context: PipelineContext) -> bool:
        state = context.state
        if state.complete:
            return False
        return bool(state.unsummarized_history())

    def _register_memory_context_bindings(self) -> None:
        ctx = self.pipeline_context
        ctx.register_snapshot("memory_agent", self._memory_snapshot)
        ctx.register_output_handler("memory_agent", self._apply_memory_output)

    def _observation_snapshot(self, state: ConversationState) -> Dict[str, Any]:
        history = state.history_with_summary()
        if not history:
            history = "No previous actions, findings or thoughts available."
        return {
            "ITERATION": state.current_iteration.index,
            "QUERY": state.query,
            "HISTORY": history,
        }

    def _evaluation_snapshot(self, state: ConversationState) -> Dict[str, Any]:
        history = state.history_with_summary()
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
        history = state.history_with_summary()
        if not history:
            history = "No previous actions, findings or thoughts available."
        gap = state.current_iteration.selected_gap or "No specific gap provided."
        return {
            "QUERY": state.query,
            "GAP": gap,
            "HISTORY": history,
        }

    def _memory_snapshot(self, state: ConversationState) -> Dict[str, Any]:
        unsummarized = state.unsummarized_history()
        if not unsummarized:
            unsummarized = "No previous unsummarized actions, findings or thoughts available."
        return {
            "ITERATION": state.current_iteration.index,
            "QUERY": state.query,
            "LAST_SUMMARY": state.summary or "No previous summary available.",
            "CONVERSATION_HISTORY": unsummarized,
        }

    def _apply_memory_output(self, state: ConversationState, result: MemoryAgentOutput) -> None:
        state.update_summary(result.summary)
