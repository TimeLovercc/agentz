"""High-level execution patterns for common pipeline workflows."""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from agentz.runner.utils import record_structured_payload, serialize_output


async def execute_tool_plan(
    plan: Any,
    tool_agents: Dict[str, Any],
    group_id: str,
    context: Any,
    agent_step_fn: Any,
    update_printer_fn: Optional[Any] = None,
) -> None:
    """Execute a routing plan with tool agents.

    Args:
        plan: AgentSelectionPlan with tasks to execute
        tool_agents: Dict mapping agent names to agent instances
        group_id: Group ID for printer updates
        context: Pipeline context with state
        agent_step_fn: Function to execute agent steps
        update_printer_fn: Optional function for printer updates
    """
    # Import here to avoid circular dependency
    from agentz.profiles.manager.routing import AgentSelectionPlan, AgentTask
    from agentz.profiles.base import ToolAgentOutput

    if not isinstance(plan, AgentSelectionPlan) or not plan.tasks:
        return

    state = context.state
    state.current_iteration.tools.clear()

    async def run_single(task: AgentTask) -> ToolAgentOutput:
        agent = tool_agents.get(task.agent)
        if agent is None:
            output = ToolAgentOutput(
                output=f"No implementation found for agent {task.agent}",
                sources=[],
            )
            if update_printer_fn:
                update_printer_fn(
                    key=f"{group_id}:tool:{task.agent}",
                    message=f"Completed {task.agent}",
                    is_done=True,
                    group_id=group_id,
                )
            return output

        raw_result = await agent_step_fn(
            agent=agent,
            instructions=task.model_dump_json(),
            span_name=task.agent,
            span_type="tool",
            output_model=ToolAgentOutput,
            printer_key=f"tool:{task.agent}",
            printer_title=f"Tool: {task.agent}",
            printer_group_id=group_id,
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
            state.record_payload(output)
        except Exception as exc:
            logger.debug(f"Failed to record tool payload for {task.agent}: {exc}")

        if update_printer_fn:
            update_printer_fn(
                key=f"{group_id}:tool:{task.agent}",
                message=f"Completed {task.agent}",
                is_done=True,
                group_id=group_id,
            )
        return output

    coroutines = [run_single(task) for task in plan.tasks]
    for coro in asyncio.as_completed(coroutines):
        tool_output = await coro
        state.current_iteration.tools.append(tool_output)


async def execute_tools(
    route_plan: Any,
    tool_agents: Dict[str, Any],
    group_id: str,
    context: Any,
    agent_step_fn: Any,
    update_printer_fn: Optional[Any] = None,
) -> None:
    """Execute tool agents based on routing plan.

    Args:
        route_plan: The routing plan (can be AgentSelectionPlan or other)
        tool_agents: Dict mapping agent names to agent instances
        group_id: Group ID for printer updates
        context: Pipeline context with state
        agent_step_fn: Function to execute agent steps
        update_printer_fn: Optional function for printer updates
    """
    from agentz.profiles.manager.routing import AgentSelectionPlan

    # Retrieve route_plan from payloads if needed
    plan = None
    if isinstance(route_plan, AgentSelectionPlan):
        plan = route_plan
    elif route_plan and hasattr(context, 'state'):
        # Try to find AgentSelectionPlan in payloads
        for payload in context.state.current_iteration.payloads:
            if isinstance(payload, AgentSelectionPlan):
                plan = payload
                break

    if plan and plan.tasks:
        await execute_tool_plan(
            plan, tool_agents, group_id, context, agent_step_fn, update_printer_fn
        )


async def run_manager_tool_loop(
    manager_agents: Dict[str, Any],
    tool_agents: Dict[str, Any],
    workflow: List[str],
    context: Any,
    agent_step_fn: Any,
    run_iterative_loop_fn: Any,
    update_printer_fn: Optional[Any] = None,
) -> Any:
    """Execute standard manager-tool iterative pattern.

    This pattern implements: observe → evaluate → route → execute tools → repeat.

    Args:
        manager_agents: Dict of manager agents (observe, evaluate, routing, writer)
        tool_agents: Dict of tool agents
        workflow: List of manager agent names to execute in order (e.g., ["observe", "evaluate", "routing"])
        context: Pipeline context with state
        agent_step_fn: Function to execute agent steps
        run_iterative_loop_fn: Function to run iterative loop
        update_printer_fn: Optional function for printer updates

    Returns:
        Result from final step

    Example:
        result = await run_manager_tool_loop(
            manager_agents=self.manager_agents,
            tool_agents=self.tool_agents,
            workflow=["observe", "evaluate", "routing"],
            context=self.context,
            agent_step_fn=self.agent_step,
            run_iterative_loop_fn=self.run_iterative_loop,
            update_printer_fn=self.update_printer,
        )
    """
    async def iteration_step(iteration, group_id: str):
        """Execute manager workflow + tool execution."""
        previous_output = context.state.query

        # Execute manager workflow in sequence
        for agent_name in workflow:
            agent = manager_agents.get(agent_name)
            if agent is None:
                logger.warning(f"Manager agent '{agent_name}' not found, skipping")
                continue

            output = await agent(previous_output)

            # Record observation for first step
            if agent_name == workflow[0]:
                iteration.observation = serialize_output(output)

            record_structured_payload(context.state, output, context_label=agent_name)
            previous_output = output

        # Execute tools if not complete
        if not context.state.complete and previous_output:
            await execute_tools(
                previous_output,
                tool_agents,
                group_id,
                context,
                agent_step_fn,
                update_printer_fn,
            )

    async def final_step(final_group: str):
        """Generate final report."""
        if update_printer_fn:
            update_printer_fn("research", "Research workflow complete", is_done=True)
        logger.info("Research workflow completed")

        writer = manager_agents.get("writer")
        if writer:
            await writer(context.state.findings_text())

    if update_printer_fn:
        update_printer_fn("research", "Executing research workflow...")

    return await run_iterative_loop_fn(
        iteration_body=iteration_step,
        final_body=final_step
    )
