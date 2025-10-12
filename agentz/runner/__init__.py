"""Agent runtime execution module.

This module provides the complete agent runtime infrastructure:

Base Runners:
- Runner: Base runner from agents library (re-exported)
- ContextRunner: Context-aware runner with output parsing

Runtime Infrastructure:
- RuntimeTracker: Manages runtime state (tracing, printing, reporting, iteration tracking, data store)
- AgentExecutor: High-level agent execution with full pipeline infrastructure
- AgentStep: Abstraction for a single agent execution step

Runtime Access:
- get_current_tracker: Access the current RuntimeTracker from anywhere
- get_current_data_store: Convenience for accessing the data store

Iteration Management:
- IterationManager: Controls iteration lifecycle for iterative workflows

Orchestration:
- WorkflowOrchestrator: High-level workflow orchestration with behavior execution

"""

from agentz.runner.base import Runner, ContextRunner
from agentz.runner.tracker import (
    RuntimeTracker,
    get_current_tracker,
    get_current_data_store,
)
from agentz.runner.executor import AgentExecutor, AgentStep, PrinterConfig
from agentz.runner.iteration import IterationManager
from agentz.runner.orchestration import WorkflowOrchestrator

__all__ = [
    # Base runners
    "Runner",
    "ContextRunner",
    # Runtime infrastructure
    "RuntimeTracker",
    "AgentExecutor",
    "AgentStep",
    "PrinterConfig",
    # Runtime access
    "get_current_tracker",
    "get_current_data_store",
    # Iteration
    "IterationManager",
    # Orchestration
    "WorkflowOrchestrator",
]
