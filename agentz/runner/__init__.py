"""Agent runtime execution module.

This module provides the complete agent runtime infrastructure:

Base Runners:
- Runner: Base runner from agents library (re-exported)
- ContextRunner: Context-aware runner with output parsing

Runtime Infrastructure:
- RuntimeTracker: Manages runtime state (tracing, printing, reporting, iteration tracking, data store)
- AgentExecutor: High-level agent execution with full pipeline infrastructure
- AgentStep: Abstraction for a single agent execution step

Pipeline Infrastructure:
- HookRegistry: Event-driven hook system for pipeline lifecycle
- IterationManager: Iteration management for iterative workflows
- WorkflowHelpers: Control flow helpers (loops, conditionals, parallel execution)

High-Level Patterns:
- execute_tool_plan: Execute tool agents from routing plans
- execute_tools: Execute tools based on routing plan
- run_manager_tool_loop: Standard manager-tool iterative pattern

Utilities:
- record_structured_payload: Record payloads to state
- serialize_output: Serialize agent outputs

Runtime Access:
- get_current_tracker: Access the current RuntimeTracker from anywhere
- get_current_data_store: Convenience for accessing the data store

"""

from agentz.runner.base import Runner, ContextRunner
from agentz.runner.tracker import (
    RuntimeTracker,
    get_current_tracker,
    get_current_data_store,
)
from agentz.runner.executor import AgentExecutor, AgentStep, PrinterConfig
from agentz.runner.hooks import HookRegistry
from agentz.runner.iteration import IterationManager
from agentz.runner.workflow import WorkflowHelpers
from agentz.runner.patterns import (
    execute_tool_plan,
    execute_tools,
    run_manager_tool_loop,
)
from agentz.runner.utils import record_structured_payload, serialize_output

__all__ = [
    # Base runners
    "Runner",
    "ContextRunner",
    # Runtime infrastructure
    "RuntimeTracker",
    "AgentExecutor",
    "AgentStep",
    "PrinterConfig",
    # Pipeline infrastructure
    "HookRegistry",
    "IterationManager",
    "WorkflowHelpers",
    # High-level patterns
    "execute_tool_plan",
    "execute_tools",
    "run_manager_tool_loop",
    # Utilities
    "record_structured_payload",
    "serialize_output",
    # Runtime access
    "get_current_tracker",
    "get_current_data_store",
]
