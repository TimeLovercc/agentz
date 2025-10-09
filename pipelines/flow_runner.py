from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from pydantic import BaseModel

from agentz.memory.conversation import ConversationState
from agentz.flow.runtime_objects import AgentCapability, PipelineContext


InputBuilder = Callable[[PipelineContext], Dict[str, Any]]
OutputHandler = Callable[[PipelineContext, BaseModel | str | Dict], None]
Condition = Callable[[PipelineContext], bool]
AsyncRunner = Callable[[PipelineContext, "FlowExecutionContext"], Awaitable[None]]


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

    def should_run(self, context: PipelineContext) -> bool:
        if self.condition is None:
            return True
        return self.condition(context)


@dataclass
class FlowExecutionContext:
    """Runtime context passed to node runners."""

    pipeline: any
    agents: Dict[str, AgentCapability]
    iteration_group_id: Optional[str] = None
    pipeline_context: PipelineContext | None = None


@dataclass
class IterationFlow:
    """Flow definition for iterative execution."""

    nodes: List[FlowNode]
    loop_condition: Condition
    after_iteration: Optional[Callable[[PipelineContext], None]] = None


class FlowRunner:
    """Execute declarative flows against a pipeline context."""

    def __init__(
        self,
        pipeline: any,
        *,
        agents: Dict[str, AgentCapability],
        iteration_flow: IterationFlow,
        final_nodes: Iterable[FlowNode],
    ):
        self.pipeline = pipeline
        self.agents = agents
        self.iteration_flow = iteration_flow
        self.final_nodes = list(final_nodes)

    async def execute(self, pipeline_context: PipelineContext) -> ConversationState:
        """Run the flow until completion and return the updated conversation state."""
        state = pipeline_context.state
        engine = pipeline_context.engine

        while self.iteration_flow.loop_condition(pipeline_context):
            iteration = engine.begin_iteration()
            iteration_group = f"iter-{iteration.index}"
            self.pipeline.iteration = iteration.index
            self.pipeline.start_group(
                iteration_group,
                title=f"Iteration {iteration.index}",
                border_style="white",
                iteration=iteration.index,
            )

            exec_ctx = FlowExecutionContext(
                pipeline=self.pipeline,
                agents=self.agents,
                iteration_group_id=iteration_group,
                pipeline_context=pipeline_context,
            )

            for node in self.iteration_flow.nodes:
                if not node.should_run(pipeline_context):
                    continue
                await self._execute_node(node, pipeline_context, exec_ctx)
                if state.complete:
                    break

            engine.mark_iteration_complete()
            if self.iteration_flow.after_iteration:
                self.iteration_flow.after_iteration(pipeline_context)

            self.pipeline.end_group(iteration_group, is_done=True)

            if state.complete:
                break

        # Finalisation
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
                pipeline_context=pipeline_context,
            )
            for node in self.final_nodes:
                if not node.should_run(pipeline_context):
                    continue
                await self._execute_node(node, pipeline_context, final_context)
            self.pipeline.end_group(final_group, is_done=True)

        return state

    async def _execute_node(
        self,
        node: FlowNode,
        pipeline_context: PipelineContext,
        exec_ctx: FlowExecutionContext,
    ) -> None:
        printer_args: Dict[str, str] = {}
        if node.printer_key:
            printer_args["printer_key"] = self._format_printer_key(node, exec_ctx)
        if node.printer_title:
            printer_args["printer_title"] = node.printer_title
        if exec_ctx.iteration_group_id:
            printer_args.setdefault("printer_group_id", exec_ctx.iteration_group_id)

        if node.custom_runner is not None:
            await node.custom_runner(pipeline_context, exec_ctx)
            return

        if node.agent_key is None:
            raise ValueError(f"Flow node '{node.name}' must define an agent_key or custom_runner.")
        if node.profile is None or node.template is None:
            raise ValueError(f"Flow node '{node.name}' must define profile and template for prompt rendering.")
        if node.input_builder is None:
            raise ValueError(f"Flow node '{node.name}' must define an input_builder.")
        if node.output_handler is None:
            raise ValueError(f"Flow node '{node.name}' must define an output_handler.")

        payload = node.input_builder(pipeline_context)
        instructions = pipeline_context.render_prompt(node.profile, node.template, payload)

        capability = exec_ctx.agents[node.agent_key]
        result = await capability.invoke(
            pipeline=exec_ctx.pipeline,
            instructions=instructions,
            span_name=node.span_name or node.name,
            span_type=node.span_type,
            output_model=node.output_model,
            printer_kwargs=printer_args,
        )

        node.output_handler(pipeline_context, result)

    def _format_printer_key(self, node: FlowNode, context: FlowExecutionContext) -> str:
        if context.iteration_group_id:
            return f"{context.iteration_group_id}:{node.printer_key}"
        return node.printer_key or node.name
