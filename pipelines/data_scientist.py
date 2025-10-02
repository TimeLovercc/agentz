from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

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
    AgentTask,
    create_routing_agent,
)
from agentz.agents.manager_agents.writer_agent import create_writer_agent
from agentz.agents.worker_agents.tool_agents import ToolAgentOutput, init_tool_agents
from agentz.memory.global_memory import global_memory
from agentz.utils import get_experiment_timestamp
from pipelines.base import BasePipeline, Conversation


class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(
        self,
        *,
        config_file: str,
        data_path: str,
        user_prompt: str,
        workflow_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        enable_tracing: bool = True,
        trace_include_sensitive_data: bool = False,
    ):
        """
        Initialize the DataScientistPipeline.

        Args:
            config_file: Path to the YAML/JSON configuration file.
            data_path: Location of the dataset to analyse for this run.
            user_prompt: Description of the task provided by the user.
            workflow_name: Optional custom name used when tracing spans.
            overrides: Optional dictionary merged into the loaded config for
                ad-hoc adjustments.
            enable_tracing: Whether to enable OpenAI Agents SDK tracing.
            trace_include_sensitive_data: Whether sensitive data is included in traces.
        """
        # Initialize base class (handles env loading, config creation)
        super().__init__(
            config_file=config_file,
            data_path=data_path,
            user_prompt=user_prompt,
            overrides=overrides,
            enable_tracing=enable_tracing,
            trace_include_sensitive_data=trace_include_sensitive_data,
        )

        if not self.data_path:
            raise ValueError("DataScientistPipeline requires 'data_path' in config or argument")
        if not self.user_prompt:
            raise ValueError("DataScientistPipeline requires 'user_prompt' in config or argument")

        self.experiment_id = get_experiment_timestamp()
        self.workflow_name = workflow_name or f"data_science_pipeline_{self.experiment_id}"

        # Get pipeline settings from config file if available
        pipeline_settings = {}
        pipeline_settings = self.full_config.get('pipeline', {}) if isinstance(self.full_config.get('pipeline'), dict) else {}

        # Research workflow configuration
        self.max_iterations = pipeline_settings.get('max_iterations', 5)
        self.max_time_minutes = pipeline_settings.get('max_time_minutes', 10)
        self.verbose = pipeline_settings.get('verbose', True)
        self.research_workflow_name = f"researcher_{self.experiment_id}"

        # State for iterative research loop
        self.iteration = 0
        self.start_time: Optional[float] = None
        self.conversation = Conversation()
        self.should_continue = True
        self.constraint_reason = ""

        # Initialize specialist agents leveraged during the research loop
        self.evaluate_agent = create_evaluate_agent(self.config)
        self.routing_agent = create_routing_agent(self.config)
        self.observe_agent = create_observe_agent(self.config)
        self.writer_agent = create_writer_agent(self.config)
        self.tool_agents = init_tool_agents(self.config)

        # Handle tracing configuration and setup with user-friendly defaults
        self._setup_tracing()

        logger.info(f"Initialized DataAgentPipeline with experiment_id: {self.experiment_id}, tracing: {enable_tracing}, sensitive_data: {trace_include_sensitive_data}")

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
        logger.info(f"Running DataAgentPipeline with experiment_id: {self.experiment_id}")
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
            "includes_sensitive_data": "true" if self.trace_include_sensitive_data else "false",
        }
        trace_context = self.trace_context(self.workflow_name, metadata=trace_metadata)

        research_report = ""

        try:
            with trace_context:
                # Create comprehensive research query that includes the task and data context
                with self.span_context(
                    function_span,
                    name="prepare_research_query",
                    input=f"experiment_id={self.experiment_id}, data_path={self.data_path}, provider={provider}",
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
                    self.printer.update_item("research", "Executing research workflow...")

                # Run iterative research workflow with span
                with self.span_context(
                    agent_span,
                    name=self.research_workflow_name,
                    tools=list(self.tool_agents.keys()),
                ) as span:
                    research_report = await self._run_research_workflow(
                        query=research_query,
                        output_length="detailed analysis with code examples",
                        output_instructions="Provide complete data science workflow with explanations, code, and results",
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

                if self.printer:
                    self.printer.update_item("store_results", "Saving final results...")

                # Store final results with span
                with self.span_context(
                    function_span,
                    name="store_final_results",
                    input=f"experiment_id={self.experiment_id}",
                ) as span:
                    self._store_final_results(research_report)
                    if span and hasattr(span, "set_output"):
                        span.set_output({"stored_bytes": len(research_report.encode('utf-8'))})
                    if self.printer:
                        self.printer.update_item(
                            "store_results",
                            "Final results saved",
                            is_done=True,
                        )

            self._stop_printer()

            self.console.print("=== Final Analysis Report ===")
            self.console.print(research_report)

            if self.enable_tracing:
                self.console.print("\n🎉 Example completed successfully!")
                self.console.print("📈 Analysis complete - check the detailed report above")
                self.console.print("🔍 Trace data has been captured for this pipeline execution")
                self.console.print("   - Pipeline spans: initialization, research workflow, completion")
                self.console.print("   - Research spans: observations, gap evaluation, agent selection, tool execution")
                self.console.print("   - Individual agent execution spans for detailed analysis")

            logger.info("DataAgentPipeline completed successfully")
            return research_report
        finally:
            self._stop_printer()


    def _prepare_research_query(self) -> str:
        """Prepare the research query and store initial task information."""
        provider = self.provider_name
        research_query = f"""
        Complete the following data science task:

        TASK: {self.user_prompt}

        DATA SOURCE: {self.data_path}

        Please provide a comprehensive analysis including:
        1. Data exploration and understanding
        2. Appropriate preprocessing steps
        3. Model selection and training
        4. Performance evaluation
        5. Insights and recommendations

        Focus on delivering actionable results and clear explanations.
        """

        # Store initial task information in global memory
        global_memory.store(
            f"task_definition_{self.experiment_id}",
            {
                "data_path": self.data_path,
                "user_prompt": self.user_prompt,
                "provider": provider,
                "research_query": research_query,
                "experiment_id": self.experiment_id
            },
            agent_id="data_pipeline"
        )
        return research_query

    def _store_final_results(self, research_report: str) -> None:
        """Store final results in global memory."""
        global_memory.store(
            f"final_report_{self.experiment_id}",
            {
                "report": research_report,
                "experiment_id": self.experiment_id,
                "timestamp": self.experiment_id
            },
            agent_id="data_pipeline"
        )
    def _reset_research_state(self) -> None:
        """Reset loop counters and conversation history before a new run."""

        self.iteration = 0
        self.should_continue = True
        self.conversation = Conversation()
        self.constraint_reason = ""

    async def _run_research_workflow(
        self,
        query: str,
        *,
        output_length: str = "",
        output_instructions: str = "",
        background_context: str = "",
    ) -> str:
        """Execute the iterative research loop to produce the final report."""

        self._reset_research_state()
        self.start_time = time.time()

        self._log_message("=== Starting Data Research Workflow ===")
        self._update_status("loop", "Starting research loop...")
        self._update_status("iteration", "Preparing first iteration...")

        while self.should_continue and self._check_constraints():
            self.iteration += 1
            self._log_message(f"\n=== Starting Iteration {self.iteration} ===")
            self._update_status("iteration", f"Iteration {self.iteration} in progress...")

            self.conversation.add_iteration()

            # 1. Generate observations
            self._update_status("observe", f"Iteration {self.iteration}: generating observations...")
            with self.span_context(
                agent_span,
                name="observe_agent",
                output_type="observations",
            ) as span:
                observations = await self._generate_observations(
                    query, background_context=background_context
                )
                if span and hasattr(span, "set_output"):
                    span.set_output({"preview": observations[:200]})
            self._update_status("observe", f"Iteration {self.iteration}: observations captured", done=True)

            # 2. Evaluate gaps
            with self.span_context(
                agent_span,
                name="evaluate_agent",
                output_type=KnowledgeGapOutput.__name__,
            ) as span:
                evaluation = await self._evaluate_gaps(
                    query, background_context=background_context
                )
                if span and hasattr(span, "set_output"):
                    span.set_output(
                        {
                            "research_complete": evaluation.research_complete,
                            "remaining_gaps": len(evaluation.outstanding_gaps),
                        }
                    )

            if not evaluation.research_complete:
                next_gap = evaluation.outstanding_gaps[0]

                # 3. Plan tool usage
                with self.span_context(
                    agent_span,
                    name="routing_agent",
                    output_type=AgentSelectionPlan.__name__,
                ) as span:
                    selection_plan = await self._select_agents(
                        next_gap, query, background_context=background_context
                    )
                    if span and hasattr(span, "set_output"):
                        span.set_output({"num_tasks": len(selection_plan.tasks)})

                # 4. Execute selected tools
                with self.span_context(
                    function_span,
                    name="execute_tool_tasks",
                    input=f"iteration={self.iteration}, num_tasks={len(selection_plan.tasks)}",
                ) as span:
                    results = await self._execute_tools(selection_plan.tasks)
                    if span and hasattr(span, "set_output"):
                        span.set_output({"completed_tasks": len(results)})
                self._update_status("iteration", f"Iteration {self.iteration} complete", done=True)
            else:
                self.should_continue = False
                self._update_status(
                    "plan",
                    "Knowledge gaps resolved; no routing required",
                    done=True,
                )
                self._update_status(
                    "tools",
                    "Knowledge gaps resolved; no tool execution required",
                    done=True,
                )
                self._update_status(
                    "iteration",
                    f"Iteration {self.iteration}: objectives satisfied",
                    done=True,
                )

        self._log_message("=== DataResearcher Marked As Complete - Finalizing Output ===")
        completion_note = self.constraint_reason or "Research objectives satisfied"
        self._update_status("loop", f"Research loop complete ({completion_note})", done=True)
        self._update_status("iteration", f"Iterations executed: {self.iteration}", done=True)

        with self.span_context(
            agent_span,
            name="writer_agent",
            output_type="final_report",
        ) as span:
            self._update_status("writer", "Compiling final report...")
            report = await self._create_final_report(
                query, length=output_length, instructions=output_instructions
            )
            if span and hasattr(span, "set_output"):
                span.set_output({"report_length": len(report)})
        self._update_status("writer", "Final report ready", done=True)

        elapsed_time = time.time() - (self.start_time or time.time())
        self._log_message(
            "DataResearcher completed in "
            f"{int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds "
            f"after {self.iteration} iterations."
        )

        return report

    def _update_status(self, suffix: str, message: str, *, done: bool = False) -> None:
        """Send a status update to the live printer if available."""

        if self.printer:
            item_id = f"research_{suffix}"
            self.printer.update_item(item_id, message, is_done=done)

    def _check_constraints(self) -> bool:
        """Check if we've exceeded our constraints (max iterations or time)."""

        if self.iteration >= self.max_iterations:
            self._log_message("\n=== Ending Research Loop ===")
            self._log_message(f"Reached maximum iterations ({self.max_iterations})")
            self.constraint_reason = f"Reached maximum iterations ({self.max_iterations})"
            return False

        if self.start_time is not None:
            elapsed_minutes = (time.time() - self.start_time) / 60
            if elapsed_minutes >= self.max_time_minutes:
                self._log_message("\n=== Ending Research Loop ===")
                self._log_message(f"Reached maximum time ({self.max_time_minutes} minutes)")
                self.constraint_reason = (
                    f"Reached maximum time ({self.max_time_minutes} minutes)"
                )
                return False

        return True

    async def _evaluate_gaps(
        self,
        query: str,
        *,
        background_context: str = "",
    ) -> KnowledgeGapOutput:
        """Evaluate the current state of research and identify knowledge gaps."""

        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

        self._update_status("evaluate", f"Iteration {self.iteration}: evaluating knowledge gaps...")
        input_str = f"""
        Current Iteration Number: {self.iteration}
        Time Elapsed: {(time.time() - (self.start_time or time.time())) / 60:.2f} minutes of maximum {self.max_time_minutes} minutes

        ORIGINAL QUERY:
        {query}

        {background}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}
        """

        result = await Runner.run(self.evaluate_agent, input_str)
        evaluation = result.final_output_as(KnowledgeGapOutput)

        if not evaluation.research_complete:
            next_gap = evaluation.outstanding_gaps[0]
            self.conversation.set_latest_gap(next_gap)
            self._log_message(f"<task>\nAddress this knowledge gap: {next_gap}\n</task>")

        outstanding = len(evaluation.outstanding_gaps)
        if outstanding:
            message = f"Iteration {self.iteration}: evaluation complete — {outstanding} gap(s) remaining"
        else:
            message = "Iteration {self.iteration}: evaluation complete — no gaps remaining"
        self._update_status("evaluate", message, done=True)

        return evaluation

    async def _select_agents(
        self,
        gap: str,
        query: str,
        *,
        background_context: str = "",
    ) -> AgentSelectionPlan:
        """Select agents to address the identified knowledge gap."""

        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

        self._update_status("plan", f"Iteration {self.iteration}: planning tool usage...")
        input_str = f"""
        ORIGINAL QUERY:
        {query}

        KNOWLEDGE GAP TO ADDRESS:
        {gap}

        {background}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}
        """

        result = await Runner.run(self.routing_agent, input_str)
        selection_plan = result.final_output_as(AgentSelectionPlan)

        self.conversation.set_latest_tool_calls(
            [
                f"[Agent] {task.agent} [Query] {task.query} [Entity] {task.entity_website if task.entity_website else 'null'}"
                for task in selection_plan.tasks
            ]
        )

        joined_calls = "\n".join(self.conversation.history[-1].tool_calls)
        self._log_message(
            "<action>\nCalling the following tools to address the knowledge gap:\n"
            f"{joined_calls}\n</action>"
        )

        task_count = len(selection_plan.tasks)
        message = f"Iteration {self.iteration}: planned {task_count} tool task(s)"
        self._update_status("plan", message, done=True)

        return selection_plan

    async def _execute_tools(self, tasks: List[AgentTask]) -> Dict[str, ToolAgentOutput]:
        """Execute the selected tools concurrently to gather information."""

        async_tasks = [self._run_agent_task(task) for task in tasks]

        num_completed = 0
        results: Dict[str, ToolAgentOutput] = {}
        total_tasks = len(async_tasks)

        if total_tasks == 0:
            self._update_status(
                "tools", f"Iteration {self.iteration}: no tool executions required", done=True
            )
            return results

        self._update_status(
            "tools", f"Iteration {self.iteration}: executing tools 0/{total_tasks}"
        )
        for future in asyncio.as_completed(async_tasks):
            gap, agent_name, result = await future
            results[f"{agent_name}_{gap}"] = result
            num_completed += 1
            self._log_message(
                f"<processing>\nTool execution progress: {num_completed}/{total_tasks}\n</processing>"
            )
            self._update_status(
                "tools",
                f"Iteration {self.iteration}: executing tools {num_completed}/{total_tasks}",
            )

        findings = [tool_output.output for tool_output in results.values()]
        self.conversation.set_latest_findings(findings)

        joined_findings = "\n\n".join(findings)
        self._log_message(f"<findings>\n{joined_findings}\n</findings>")
        self._update_status("tools", f"Iteration {self.iteration}: tool execution complete", done=True)

        return results

    async def _run_agent_task(self, task: AgentTask) -> tuple[str, str, ToolAgentOutput]:
        """Run a single agent task and return the result."""

        with self.span_context(
            function_span,
            name=f"agent_task_{task.agent}",
            input=(
                f"gap={task.gap}; query={task.query[:100]}"
                if task.query
                else f"gap={task.gap}; query="
            ),
        ):
            return await self._execute_single_task(task)

    async def _execute_single_task(self, task: AgentTask) -> tuple[str, str, ToolAgentOutput]:
        """Execute a single agent task."""

        try:
            agent_name = task.agent
            agent = self.tool_agents.get(agent_name)
            if agent:
                output = await agent.run(task.model_dump_json())
            else:
                output = ToolAgentOutput(
                    output=f"No implementation found for agent {agent_name}",
                    sources=[],
                )

            return task.gap, agent_name, output
        except Exception as error:
            error_output = ToolAgentOutput(
                output=
                f"Error executing {task.agent} for gap '{task.gap}': {str(error)}",
                sources=[],
            )
            return task.gap, task.agent, error_output

    async def _generate_observations(
        self, query: str, *, background_context: str = ""
    ) -> str:
        """Generate observations from the current state of the research."""

        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

        input_str = f"""
        You are starting iteration {self.iteration} of your research process.

        ORIGINAL QUERY:
        {query}

        {background}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}
        """
        result = await Runner.run(self.observe_agent, input_str)
        observations = result.final_output

        self.conversation.set_latest_thought(observations)
        self._log_message(f"<thought>\n{observations}\n</thought>")
        return observations

    async def _create_final_report(
        self,
        query: str,
        *,
        length: str = "",
        instructions: str = "",
    ) -> str:
        """Create the final response from the completed draft."""

        self._log_message("=== Drafting Final Response ===")

        length_str = f"* The full response should be approximately {length}.\n" if length else ""
        instructions_str = f"* {instructions}" if instructions else ""
        guidelines_str = (
            "\n\nGUIDELINES:\n" + length_str + instructions_str
        ).strip("\n") if length or instructions else ""

        all_findings = "\n\n".join(self.conversation.get_all_findings()) or "No findings available yet."

        input_str = f"""
        Provide a response based on the query and findings below with as much detail as possible. {guidelines_str}

        QUERY: {query}

        FINDINGS:
        {all_findings}
        """

        try:
            result = await Runner.run(self.writer_agent, input_str)
            report = result.final_output
            self._log_message("Final response from DataResearcher created successfully")
            return report
        except Exception as error:
            return f"Error generating report: {str(error)}"

    def _log_message(self, message: str) -> None:
        """Log a message if verbose output is enabled."""

        if self.verbose:
            if self.console:
                self.console.print(message)
            else:
                print(message)
