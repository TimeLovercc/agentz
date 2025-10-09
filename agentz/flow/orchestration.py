"""Behavior orchestration with dynamic agent selection.

This module provides the glue between behaviors and agents, enabling
truly decoupled execution where:
- Behaviors define WHAT needs to happen
- Agents define HOW to execute
- Orchestration connects them dynamically
"""

from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

from agentz.context.engine import ContextEngine


class BehaviorExecutor:
    """Executes behaviors with dynamic agent selection.

    This class decouples behaviors from agents by:
    1. Rendering the behavior using ContextEngine
    2. Selecting a capable agent (currently via explicit mapping)
    3. Executing the agent with the rendered instructions

    Future enhancement: Dynamic agent selection based on behavior requirements.
    """

    def __init__(
        self,
        engine: ContextEngine,
        agent_registry: Dict[str, Any],
        pipeline: Any,  # For agent_step access
    ):
        """Initialize behavior executor.

        Args:
            engine: ContextEngine for behavior rendering and state management
            agent_registry: Mapping of agent names to agent instances
            pipeline: Pipeline instance for agent execution
        """
        self.engine = engine
        self.agent_registry = agent_registry
        self.pipeline = pipeline

    async def execute_behavior(
        self,
        behavior_name: str,
        agent_name: str,
        snapshot_builder: Callable[[ContextEngine], Dict[str, Any]],
        output_handler: Callable[[ContextEngine, Any], None],
        *,
        output_model: Optional[type[BaseModel]] = None,
        span_name: Optional[str] = None,
        span_type: str = "agent",
        printer_key: Optional[str] = None,
        printer_title: Optional[str] = None,
        printer_group_id: Optional[str] = None,
    ) -> Any:
        """Execute a behavior with the specified agent.

        Args:
            behavior_name: Name of the behavior to execute
            agent_name: Name of the agent to use
            snapshot_builder: Function to build behavior input from state
            output_handler: Function to apply behavior output to state
            output_model: Optional pydantic model for output parsing
            span_name: Name for tracing span
            span_type: Type of span (agent or function)
            printer_key: Key for printer updates
            printer_title: Title for printer display
            printer_group_id: Optional group to nest this item in

        Returns:
            The result from the agent execution
        """
        # Build input snapshot
        payload = snapshot_builder(self.engine)

        # Render behavior instructions
        instructions = self.engine.render_behavior(behavior_name, payload)

        # Get agent
        agent = self.agent_registry.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent '{agent_name}' not found in registry")

        # Execute agent
        result = await self.pipeline.agent_step(
            agent=agent,
            instructions=instructions,
            span_name=span_name or behavior_name,
            span_type=span_type,
            output_model=output_model,
            printer_key=printer_key,
            printer_title=printer_title,
            printer_group_id=printer_group_id,
        )

        # Apply output to state
        output_handler(self.engine, result)

        return result

    async def execute_custom(
        self,
        runner: Callable[[ContextEngine, Any], Any],
        execution_context: Any,
    ) -> None:
        """Execute a custom runner function.

        This allows for non-behavior execution (e.g., tool tasks)
        while maintaining the same orchestration interface.

        Args:
            runner: Async callable that performs the execution
            execution_context: Context passed to the runner
        """
        await runner(self.engine, execution_context)


class WorkflowOrchestrator:
    """High-level workflow orchestration combining iteration and behavior execution.

    This provides a simple API for common workflow patterns while keeping
    the underlying components decoupled.
    """

    def __init__(
        self,
        engine: ContextEngine,
        agent_registry: Dict[str, Any],
        pipeline: Any,
    ):
        """Initialize workflow orchestrator.

        Args:
            engine: ContextEngine for state management
            agent_registry: Mapping of agent names to agent instances
            pipeline: Pipeline instance for execution
        """
        self.engine = engine
        self.behavior_executor = BehaviorExecutor(engine, agent_registry, pipeline)
        self.pipeline = pipeline

    async def run_iteration_step(
        self,
        behavior_name: str,
        agent_name: str,
        snapshot_builder: Callable[[ContextEngine], Dict[str, Any]],
        output_handler: Callable[[ContextEngine, Any], None],
        *,
        condition: Optional[Callable[[ContextEngine], bool]] = None,
        output_model: Optional[type[BaseModel]] = None,
        span_name: Optional[str] = None,
        span_type: str = "agent",
        printer_key: Optional[str] = None,
        printer_title: Optional[str] = None,
        printer_group_id: Optional[str] = None,
    ) -> Optional[Any]:
        """Run a single iteration step with optional condition.

        Args:
            behavior_name: Name of the behavior to execute
            agent_name: Name of the agent to use
            snapshot_builder: Function to build behavior input
            output_handler: Function to apply behavior output
            condition: Optional condition to check before execution
            output_model: Optional output model for parsing
            span_name: Span name for tracing
            span_type: Span type
            printer_key: Printer key
            printer_title: Printer title
            printer_group_id: Printer group

        Returns:
            Result if executed, None if condition failed
        """
        # Check condition
        if condition is not None and not condition(self.engine):
            return None

        # Execute behavior
        return await self.behavior_executor.execute_behavior(
            behavior_name=behavior_name,
            agent_name=agent_name,
            snapshot_builder=snapshot_builder,
            output_handler=output_handler,
            output_model=output_model,
            span_name=span_name,
            span_type=span_type,
            printer_key=printer_key,
            printer_title=printer_title,
            printer_group_id=printer_group_id,
        )
