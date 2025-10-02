from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger

from agents import Runner
from agents.tracing.create import agent_span, function_span
from agentz.agents.manager_agents.evaluate_agent import (
    KnowledgeGapOutput,
    create_evaluate_agent,
)
from agentz.agents.manager_agents.observe_agent import create_observe_agent
from agentz.agents.manager_agents.routing_agent import (
    AgentSelectionPlan,
    create_routing_agent,
)
from agentz.agents.manager_agents.writer_agent import create_writer_agent
from agentz.agents.worker_agents.tool_agents import init_tool_agents
from agentz.configuration.base import PipelineConfigSource
from agentz.configuration.data_science import (
    ManagerAgentInput,
    instantiate_agent_spec,
    instantiate_tool_agent_spec,
    normalise_manager_agent_specs,
    resolve_data_science_config,
)
from agentz.memory.global_memory import global_memory
from agentz.utils import get_experiment_timestamp
from pipelines.base import BasePipeline, Conversation

DEFAULT_CONFIG_PATH = "pipelines/configs/data_science.yaml"


class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(
        self,
        *,
        config: Optional[PipelineConfigSource] = None,
        config_file: Optional[str] = None,
        data_path: Optional[str] = None,
        user_prompt: Optional[str] = None,
        agents: Optional[ManagerAgentInput] = None,
        workflow_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        enable_tracing: bool = True,
        trace_include_sensitive_data: bool = False,
    ):
        """
        Initialize the DataScientistPipeline.

        Args:
            config: Configuration object, mapping, or path resolving to pipeline settings.
            config_file: Backwards compatible path to configuration file.
            data_path: Optional dataset path for this run; falls back to config value.
            user_prompt: Optional task description; falls back to config value.
            agents: Optional manager agent overrides provided either as a mapping
                or a list in (evaluate, routing, observe, writer) order.
            workflow_name: Optional custom name used when tracing spans.
            overrides: Optional dictionary merged into the loaded config for
                ad-hoc adjustments.
            enable_tracing: Whether to enable OpenAI Agents SDK tracing.
            trace_include_sensitive_data: Whether sensitive data is included in traces.
        """
        if config is not None and config_file is not None:
            raise ValueError("Provide either 'config' or 'config_file', not both")

        config_source: PipelineConfigSource
        self._direct_manager_agent_specs = (
            normalise_manager_agent_specs(agents) if agents is not None else {}
        )

        if config is not None:
            config_source = config
        else:
            config_source = config_file or DEFAULT_CONFIG_PATH

        # Initialize base class (handles env loading, config creation)
        super().__init__(
            config_source=config_source,
            config_resolver=resolve_data_science_config,
            data_path=data_path,
            user_prompt=user_prompt,
            overrides=overrides,
            enable_tracing=enable_tracing,
            trace_include_sensitive_data=trace_include_sensitive_data,
        )

        if not self.data_path:
            raise ValueError(
                "DataScientistPipeline requires 'data_path' in config or argument"
            )
        if not self.user_prompt:
            raise ValueError(
                "DataScientistPipeline requires 'user_prompt' in config or argument"
            )

        self.experiment_id = get_experiment_timestamp()
        self.workflow_name = (
            workflow_name or f"data_science_pipeline_{self.experiment_id}"
        )

        # Get pipeline settings from config file if available
        pipeline_settings = {}
        pipeline_settings = (
            self.full_config.get("pipeline", {})
            if isinstance(self.full_config.get("pipeline"), dict)
            else {}
        )

        # Research workflow configuration
        self.max_iterations = pipeline_settings.get("max_iterations", 5)
        self.max_time_minutes = pipeline_settings.get("max_time_minutes", 10)
        self.verbose = pipeline_settings.get("verbose", True)
        self.research_workflow_name = f"researcher_{self.experiment_id}"

        # State for iterative research loop
        self.iteration = 0
        self.start_time: Optional[float] = None
        self.conversation = Conversation()
        self.should_continue = True
        self.constraint_reason = ""

        # Initialize specialist agents leveraged during the research loop
        self.evaluate_agent = self._resolve_manager_agent(
            "evaluate_agent", create_evaluate_agent
        )
        self.routing_agent = self._resolve_manager_agent(
            "routing_agent", create_routing_agent
        )
        self.observe_agent = self._resolve_manager_agent(
            "observe_agent", create_observe_agent
        )
        self.writer_agent = self._resolve_manager_agent(
            "writer_agent", create_writer_agent
        )
        self.tool_agents = self._build_tool_agents()

        # Handle tracing configuration and setup with user-friendly defaults
        self._setup_tracing()

        logger.info(
            f"Initialized DataAgentPipeline with experiment_id: {self.experiment_id}, "
            f"tracing: {enable_tracing}, sensitive_data: {trace_include_sensitive_data}"
        )

    def _resolve_manager_agent(self, name: str, factory):
        if name in self._direct_manager_agent_specs:
            return instantiate_agent_spec(
                self._direct_manager_agent_specs[name], self.config
            )

        overrides = self.config_attachments.get("manager_agents", {})
        if name in overrides:
            return instantiate_agent_spec(overrides[name], self.config)
        return factory(self.config)

    def _build_tool_agents(self) -> Dict[str, Any]:
        overrides = self.config_attachments.get("tool_agents", {})
        if overrides and not isinstance(overrides, dict):
            raise TypeError("tool_agents overrides must be provided as a mapping")

        tool_agents = init_tool_agents(self.config)
        for tool_name, spec in overrides.items():
            tool_agents[tool_name] = instantiate_tool_agent_spec(spec, self.config)
        return tool_agents

    def _setup_tracing(self):
        """Setup tracing configuration with user-friendly output."""
        provider = self.provider_name
        if self.enable_tracing:
            self.console.print("🍌 Starting Data Science Pipeline with Tracing")
            self.console.print(f"📊 Data: {self.data_path}")
            self.console.print(f"🔧 Provider: {provider}")
            self.console.print(f"🤖 Model: {self.config.model_name}")
            self.console.print(f"📋 Task: {self.user_prompt[:100]}...")
            self.console.print("🔍 Tracing: Enabled")
            self.console.print(
                f"🔒 Sensitive Data in Traces: {'Yes' if self.trace_include_sensitive_data else 'No'}"
            )
            self.console.print(f"🏷️ Workflow: {self.workflow_name}")
        else:
            self.console.print("🍌 Starting Data Science Pipeline")
            self.console.print(f"📊 Data: {self.data_path}")
            self.console.print(f"🔧 Provider: {provider}")
            self.console.print(f"🤖 Model: {self.config.model_name}")

    def _start_printer(self) -> None:
        """Create and attach a live status printer for this run."""
        if self.printer is None:
            self.start_printer()

    def _stop_printer(self) -> None:
        """Stop the live printer if it's currently active."""
        if self.printer is not None:
            self.stop_printer()

    async def run(self):
        """Run the data analysis pipeline."""
        logger.info(
            f"Running DataAgentPipeline with experiment_id: {self.experiment_id}"
        )
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"User prompt: {self.user_prompt}")
        provider = self.provider_name
        logger.info(f"Provider: {provider}, Model: {self.config.model_name}")

        self._start_printer()
        if self.printer:
            self.printer.update_item(
                "workflow",
                f"Workflow: {self.workflow_name}",
                is_done=True,
                hide_checkmark=True,
            )
            self.printer.update_item("prepare_query", "Preparing research query...")

        trace_metadata = {
            "experiment_id": self.experiment_id,
            "includes_sensitive_data": "true"
            if self.trace_include_sensitive_data
            else "false",
        }
        trace_context = self.trace_context(
            self.workflow_name, metadata=trace_metadata
        )

        research_report = ""

        try:
            with trace_context:
                # Create comprehensive research query that includes the task and data context
                with self.span_context(
                    function_span,
                    name="prepare_research_query",
                    input=(
                        f"experiment_id={self.experiment_id}, "
                        f"data_path={self.data_path}, provider={provider}"
                    ),
                ) as span:
                    research_query = self._prepare_research_query()
                    if span and hasattr(span, "set_output"):
                        span.set_output({"query_length": len(research_query)})
                    if self.printer:
                        self.printer.update_item(
                            "prepare_query",
                            "Research query prepared",
                            is_done=True,
                        )

                if self.printer:
                    self.printer.update_item(
                        "research", "Executing research workflow..."
                    )

                # Run iterative research workflow with span
                with self.span_context(
                    agent_span,
                    name=self.research_workflow_name,
                    tools=list(self.tool_agents.keys()),
                ) as span:
                    research_report = await self._run_research_workflow(
                        query=research_query,
                        output_length="detailed analysis with code examples",
                        output_instructions=(
                            "Provide complete data science workflow with "
                            "explanations, code, and results"
                        ),
                        background_context=(
                            f"Experiment: {self.experiment_id}, Dataset: {self.data_path}, "
                            f"Provider: {provider}, Model: {self.config.model_name}"
                        ),
                    )
                    if span and hasattr(span, "set_output"):
                        span.set_output({"report_preview": research_report[:200]})
                    if self.printer:
                        self.printer.update_item(
                            "research",
                            "Research workflow complete",
                            is_done=True,
                        )

        finally:
            self._stop_printer()

        logger.info("Research workflow completed")
        await self._finalise_research(research_report)

    async def _run_research_workflow(
        self,
        *,
        query: str,
        output_length: str,
        output_instructions: str,
        background_context: str,
    ) -> str:
        """Orchestrate the research workflow across iterations."""

        start_time = time.time()
        self.start_time = start_time
        self.conversation.add_iteration()

        while self.should_continue:
            elapsed_minutes = (time.time() - start_time) / 60
            if elapsed_minutes >= self.max_time_minutes:
                self.should_continue = False
                self.constraint_reason = "max_time"
                break

            if self.iteration >= self.max_iterations:
                self.should_continue = False
                self.constraint_reason = "max_iterations"
                break

            iteration_index = self.iteration + 1
            logger.info(f"Starting iteration {iteration_index}")
            self.conversation.add_iteration()

            iteration_complete = await self._execute_iteration(
                query=query,
                output_length=output_length,
                output_instructions=output_instructions,
                background_context=background_context,
            )

            if iteration_complete:
                break

        return await self._create_final_report()

    async def _execute_iteration(
        self,
        *,
        query: str,
        output_length: str,
        output_instructions: str,
        background_context: str,
    ) -> bool:
        """Perform a single iteration of the research workflow."""

        iteration_num = self.iteration + 1
        logger.info(f"Executing iteration {iteration_num}")

        with self.span_context(
            agent_span,
            name=f"iteration_{iteration_num}",
            tools=list(self.tool_agents.keys()),
        ):
            evaluation = await self._evaluate_research_state(query=query)

        if evaluation.research_complete:
            logger.info(
                f"Research marked complete by evaluation agent at iteration {iteration_num}"
            )
            self.should_continue = False
            return True

        self._route_tasks(evaluation)
        return False

    async def _evaluate_research_state(self, *, query: str) -> KnowledgeGapOutput:
        """Evaluate whether research is complete and identify gaps."""

        instructions = (
            f"Evaluate progress on: {query}. Current findings: "
            f"{self.conversation.get_all_findings()}"
        )

        with self.span_context(
            function_span,
            name="evaluate_research_state",
            input=instructions,
        ) as span:
            result = await Runner.run(self.evaluate_agent, instructions)
            gap_output = KnowledgeGapOutput.model_validate_json(result.final_output)
            if span and hasattr(span, "set_output"):
                span.set_output(gap_output.model_dump())
            return gap_output

    def _route_tasks(self, evaluation: KnowledgeGapOutput) -> None:
        """Route tasks to the appropriate specialized agents."""

        with self.span_context(
            agent_span,
            name="routing_agent",
            tools=list(self.tool_agents.keys()),
        ) as span:
            routing_prompt = self._prepare_routing_prompt(evaluation)
            routing_plan_raw = Runner.run_sync(self.routing_agent, routing_prompt)
            routing_plan = AgentSelectionPlan.model_validate_json(
                routing_plan_raw.final_output
            )
            if span and hasattr(span, "set_output"):
                span.set_output(routing_plan.model_dump())

            self._dispatch_tool_tasks(routing_plan)

    def _dispatch_tool_tasks(self, routing_plan: AgentSelectionPlan) -> None:
        """Dispatch tasks to tool agents based on routing plan."""

        for task in routing_plan.tasks:
            if task.agent not in self.tool_agents:
                logger.warning(f"Unknown tool agent requested: {task.agent}")
                continue

            tool_agent = self.tool_agents[task.agent]
            Runner.submit_task(tool_agent, task.json())

    async def _create_final_report(self) -> str:
        """Generate the final report using the writer agent."""

        collated_findings = "\n".join(self.conversation.get_all_findings())
        prompt = (
            "Create a final research report summarizing the findings, "
            "methodology, and recommendations.\n\n"
            f"Dataset: {self.data_path}\n"
            f"Task: {self.user_prompt}\n"
            f"Findings: {collated_findings}"
        )

        with self.span_context(
            agent_span,
            name="writer_agent",
            tools=list(self.tool_agents.keys()),
        ) as span:
            result = await Runner.run(self.writer_agent, prompt)
            if span and hasattr(span, "set_output"):
                span.set_output({"report_length": len(result.final_output)})
            return result.final_output

    async def _finalise_research(self, research_report: str) -> None:
        """Finalize logging, persist conversation, and update memory."""

        timestamped_report = (
            f"Experiment {self.experiment_id}\n\n{research_report.strip()}"
        )
        global_memory.store_report(timestamped_report)

        logger.info(f"Stored research report with experiment_id {self.experiment_id}")
        if self.printer:
            self.printer.update_item(
                "writer_agent",
                "Final report generated",
                is_done=True,
            )

    def _prepare_research_query(self) -> str:
        """Prepare initial research query for the research workflow."""

        query = (
            f"Task: {self.user_prompt}\n"
            f"Dataset path: {self.data_path}\n"
            "Provide a comprehensive data science workflow"
        )
        logger.debug(f"Prepared research query: {query}")
        return query

    def _prepare_routing_prompt(self, evaluation: KnowledgeGapOutput) -> str:
        """Prepare routing agent prompt based on evaluation."""

        prompt = (
            "Route the next tasks based on the following knowledge gaps:\n"
            f"{evaluation.outstanding_gaps}\n"
            f"Iteration: {self.iteration}\n"
        )
        logger.debug(f"Prepared routing prompt: {prompt}")
        return prompt
