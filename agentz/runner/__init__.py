"""Agent runtime execution module.

This module provides the complete agent runtime infrastructure:

Base Runners:
- Runner: Base runner from agents library (re-exported)
- ContextRunner: Context-aware runner with output parsing

Execution Infrastructure:
- ExecutionContext: Manages runtime state (tracing, printing, iteration tracking)
- AgentExecutor: High-level agent execution with full pipeline infrastructure
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

from agentz.runner.base import Runner, ContextRunner
from agentz.runner.context import (
    ExecutionContext,
    auto_trace,
    with_span_step,
    get_current_context,
    get_current_data_store,
)
from agentz.runner.executor import AgentExecutor, AgentStep, PrinterConfig
from agentz.runner.iteration import IterationManager
from agentz.runner.orchestration import WorkflowOrchestrator
from agentz.runner.patterns import (
    ConditionalPattern,
    ExecutionPattern,
    LoopPattern,
    ParallelPattern,
    PipelinePattern,
    RetryPattern,
    SequentialPattern,
)

__all__ = [
    # Base runners
    "Runner",
    "ContextRunner",
    # Execution infrastructure
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
