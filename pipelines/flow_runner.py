from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel

from agentz.memory.behavior_profiles import runtime_prompts
from agentz.memory.conversation import ConversationState


InputBuilder = Callable[[ConversationState], Dict[str, Any]]
OutputHandler = Callable[[ConversationState, Any], None]
Condition = Callable[[ConversationState], bool]
AsyncRunner = Callable[[ConversationState, "FlowExecutionContext"], Awaitable[None]]


@dataclass
class FlowNode:
    """A single executable node within a flow."""

    name: str
    agent_key: Optional[str] = None
    profile: Optional[str] = None
    template: Optional[str] = None
    input_builder: Optional[InputBuilder] = None
    output_model: Optional[type[BaseModel]] = None
    output_handler: Optional[OutputHandler] = None
    condition: Optional[Condition] = None
    custom_runner: Optional[AsyncRunner] = None
    span_name: Optional[str] = None
    span_type: str = "agent"
    printer_key: Optional[str] = None
    printer_title: Optional[str] = None

    def should_run(self, state: ConversationState) -> bool:
        if self.condition is None:
            return True
        return self.condition(state)


@dataclass
class FlowExecutionContext:
    """Runtime context passed to node runners."""

    pipeline: Any
    agents: Dict[str, Any]
    iteration_group_id: Optional[str] = None


@dataclass
class IterationFlow:
    """Flow definition for iterative execution."""

    nodes: List[FlowNode]
    loop_condition: Condition
    after_iteration: Optional[Callable[[ConversationState], None]] = None


class FlowRunner:
    """Execute declarative flows against a conversation state."""

    def __init__(
        self,
        pipeline: Any,
        *,
        agents: Dict[str, Any],
        iteration_flow: IterationFlow,
        final_nodes: Iterable[FlowNode],
    ):
        self.pipeline = pipeline
        self.agents = agents
        self.iteration_flow = iteration_flow
        self.final_nodes = list(final_nodes)

    async def execute(self, state: ConversationState) -> ConversationState:
        """Run the flow until completion and return the updated state."""
        while self.iteration_flow.loop_condition(state):
            iteration = state.begin_iteration()
            iteration_group = f"iter-{iteration.index}"
            self.pipeline.iteration = iteration.index
            self.pipeline.start_group(
                iteration_group,
                title=f"Iteration {iteration.index}",
                border_style="white",
                iteration=iteration.index,
            )

            context = FlowExecutionContext(
                pipeline=self.pipeline,
                agents=self.agents,
                iteration_group_id=iteration_group,
            )

            for node in self.iteration_flow.nodes:
                if not node.should_run(state):
                    continue
                await self._execute_node(node, state, context)
                if state.complete:
                    break

            state.mark_iteration_complete()
            if self.iteration_flow.after_iteration:
                self.iteration_flow.after_iteration(state)

            self.pipeline.end_group(iteration_group, is_done=True)

            if state.complete:
                break

        # Run finalisation nodes
        if self.final_nodes:
            final_group = "iter-final"
            self.pipeline.start_group(
                final_group,
                title="Final Report",
                border_style="white",
            )
            final_context = FlowExecutionContext(
                pipeline=self.pipeline,
                agents=self.agents,
                iteration_group_id=final_group,
            )
            for node in self.final_nodes:
                if not node.should_run(state):
                    continue
                await self._execute_node(node, state, final_context)
            self.pipeline.end_group(final_group, is_done=True)

        return state

    async def _execute_node(
        self,
        node: FlowNode,
        state: ConversationState,
        context: FlowExecutionContext,
    ) -> None:
        printer_args = {}
        if node.printer_key:
            printer_args["printer_key"] = self._format_printer_key(node, context)
        if node.printer_title:
            printer_args["printer_title"] = node.printer_title
        if context.iteration_group_id:
            printer_args.setdefault("printer_group_id", context.iteration_group_id)

        if node.custom_runner is not None:
            await node.custom_runner(state, context)
            return

        if node.agent_key is None:
            raise ValueError(f"Flow node '{node.name}' must define an agent_key or custom_runner.")
        if node.profile is None or node.template is None:
            raise ValueError(f"Flow node '{node.name}' must define profile and template for prompt rendering.")
        if node.input_builder is None:
            raise ValueError(f"Flow node '{node.name}' must define an input_builder.")
        if node.output_handler is None:
            raise ValueError(f"Flow node '{node.name}' must define an output_handler.")

        payload = node.input_builder(state)
        instructions = runtime_prompts.render(
            node.profile,
            node.template,
            **payload,
        )

        agent = context.agents[node.agent_key]
        result = await self.pipeline.agent_step(
            agent=agent,
            instructions=instructions,
            span_name=node.span_name or node.name,
            span_type=node.span_type,
            output_model=node.output_model,
            **printer_args,
        )

        node.output_handler(state, result)

    def _format_printer_key(self, node: FlowNode, context: FlowExecutionContext) -> str:
        if context.iteration_group_id:
            return f"{context.iteration_group_id}:{node.printer_key}"
        return node.printer_key
