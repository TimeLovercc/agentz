"""Agent runtime execution flow module.

This module provides core runtime execution primitives and patterns for orchestrating
agent workflows.

Core Components:
- ExecutionContext: Manages runtime state (tracing, printing, iteration tracking)
- AgentExecutor: Core execution primitive for running agents
- AgentStep: Abstraction for a single agent execution step

Iteration Management:
- IterationManager: Controls iteration lifecycle for iterative workflows

Orchestration:
- WorkflowOrchestrator: High-level workflow orchestration with behavior execution

Decorators:
- auto_trace: Wraps run methods with context management
- with_span_step: Wraps functions with span context and printer updates

Patterns:
- SequentialPattern: Execute steps in sequence
- ParallelPattern: Execute steps concurrently
- ConditionalPattern: Execute based on conditions
- LoopPattern: Iterative execution with termination
- RetryPattern: Retry with exponential backoff
- PipelinePattern: Compose multiple patterns
"""

from agentz.flow.context import (
    ExecutionContext,
    auto_trace,
    with_span_step,
    get_current_context,
    get_current_data_store,
)
from agentz.flow.executor import AgentExecutor, AgentStep, PrinterConfig
from agentz.flow.iteration import IterationManager
from agentz.flow.orchestration import WorkflowOrchestrator
from agentz.flow.patterns import (
    ConditionalPattern,
    ExecutionPattern,
    LoopPattern,
    ParallelPattern,
    PipelinePattern,
    RetryPattern,
    SequentialPattern,
)

__all__ = [
    # Core
    "ExecutionContext",
    "AgentExecutor",
    "AgentStep",
    "PrinterConfig",
    # Context access
    "get_current_context",
    "get_current_data_store",
    # Iteration
    "IterationManager",
    # Orchestration
    "WorkflowOrchestrator",
    # Decorators
    "auto_trace",
    "with_span_step",
    # Patterns
    "ExecutionPattern",
    "SequentialPattern",
    "ParallelPattern",
    "ConditionalPattern",
    "LoopPattern",
    "RetryPattern",
    "PipelinePattern",
]
