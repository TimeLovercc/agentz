from __future__ import annotations

import asyncio
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console

from agents import Runner
from agents.tracing.create import agent_span, function_span, trace
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
from agentz.utils import Printer, get_experiment_timestamp
from agentz.llm.llm_setup import LLMConfig
from pipelines.base import BasePipeline


class IterationData(BaseModel):
    """Data for a single iteration of the research loop."""
    gap: str = Field(description="The gap addressed in the iteration", default="")
    tool_calls: List[str] = Field(description="The tool calls made", default_factory=list)
    findings: List[str] = Field(description="The findings collected from tool calls", default_factory=list)
    thought: str = Field(description="The thinking done to reflect on the success of the iteration and next steps", default="")


class Conversation(BaseModel):
    """A conversation between the user and the iterative researcher."""
    history: List[IterationData] = Field(description="The data for each iteration of the research loop", default_factory=list)

    def add_iteration(self, iteration_data: Optional[IterationData] = None):
        if iteration_data is None:
            iteration_data = IterationData()
        self.history.append(iteration_data)

    def set_latest_gap(self, gap: str):
        if self.history:
            self.history[-1].gap = gap

    def set_latest_tool_calls(self, tool_calls: List[str]):
        if self.history:
            self.history[-1].tool_calls = tool_calls

    def set_latest_findings(self, findings: List[str]):
        if self.history:
            self.history[-1].findings = findings

    def set_latest_thought(self, thought: str):
        if self.history:
            self.history[-1].thought = thought

    def get_all_findings(self) -> List[str]:
        return [finding for iteration_data in self.history for finding in iteration_data.findings]

    def compile_conversation_history(self) -> str:
        """Compile the conversation history into a string."""
        conversation = ""
        for iteration_num, iteration_data in enumerate(self.history):
            conversation += f"[ITERATION {iteration_num + 1}]\n\n"
            if iteration_data.thought:
                conversation += f"<thought>\n{iteration_data.thought}\n</thought>\n\n"
            if iteration_data.gap:
                conversation += f"<task>\nAddress this knowledge gap: {iteration_data.gap}\n</task>\n\n"
            if iteration_data.tool_calls:
                joined_calls = '\n'.join(iteration_data.tool_calls)
                conversation += f"<action>\nCalling the following tools to address the knowledge gap:\n{joined_calls}\n</action>\n\n"
            if iteration_data.findings:
                joined_findings = '\n\n'.join(iteration_data.findings)
                conversation += f"<findings>\n{joined_findings}\n</findings>\n\n"

        return conversation


class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(
        self,
        data_path: str,
        user_prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_tracing: bool = True,
        trace_include_sensitive_data: bool = False,
        workflow_name: Optional[str] = None,
        config_dict: Optional[dict] = None,
        llm_config: Optional[LLMConfig] = None,
        config_file: Optional[str] = None
    ):
        """
        Initialize the DataScientistPipeline.

        Args:
            data_path: Path to the dataset file
            user_prompt: User's description of the task
            provider: LLM provider name (openai, gemini, deepseek, etc.)
            model: Model name (optional, uses provider defaults)
            api_key: API key (optional, auto-loads from env)
            base_url: Custom base URL (optional)
            enable_tracing: Whether to enable OpenAI Agents SDK tracing
            trace_include_sensitive_data: Whether to include sensitive data in traces
            workflow_name: Custom workflow name for tracing
            config_dict: Pre-built config dictionary (alternative to individual params)
            llm_config: Pre-created LLM configuration (alternative to config_dict)
            config_file: Path to config file (YAML/JSON) - loads all settings from file
        """
        # Initialize base class (handles env loading, config creation)
        super().__init__(
            data_path=data_path,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            enable_tracing=enable_tracing,
            trace_include_sensitive_data=trace_include_sensitive_data,
            config_dict=config_dict,
            llm_config=llm_config,
            config_file=config_file
        )

        self.console = Console()
        self.printer: Optional[Printer] = None
        self.experiment_id = get_experiment_timestamp()
        self.workflow_name = workflow_name or f"data_science_pipeline_{self.experiment_id}"

        # Get pipeline settings from config file if available
        pipeline_settings = {}
        if self.config.full_config:
            pipeline_settings = self.config.full_config.get('pipeline', {})

        # Initialize data researcher with tracing configuration and config settings
        self.researcher = DataScientist(
            max_iterations=pipeline_settings.get('max_iterations', 5),
            max_time_minutes=pipeline_settings.get('max_time_minutes', 10),
            verbose=pipeline_settings.get('verbose', True),
            tracing=enable_tracing,
            config=self.config,
            trace_include_sensitive_data=trace_include_sensitive_data,
            workflow_name=f"researcher_{self.experiment_id}",
            console=self.console,
        )

        # Handle tracing configuration and setup with user-friendly defaults
        self._setup_tracing()

        logger.info(f"Initialized DataAgentPipeline with experiment_id: {self.experiment_id}, tracing: {enable_tracing}, sensitive_data: {trace_include_sensitive_data}")

    def _setup_tracing(self):
        """Setup tracing configuration with user-friendly output."""
        provider = (self.config_dict or {}).get('provider', 'unknown provider')
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

    def _span_context(self, span_factory, **kwargs):
        """Return a span context when tracing is enabled, else a no-op context."""
        return span_factory(**kwargs) if self.enable_tracing else nullcontext()

    def _start_printer(self) -> None:
        """Create and attach a live status printer for this run."""
        if self.printer is None:
            self.printer = Printer(self.console)
            self.researcher.attach_printer(self.printer)

    def _stop_printer(self) -> None:
        """Stop the live printer if it's currently active."""
        if self.printer is not None:
            self.printer.end()
            self.researcher.attach_printer(None)
            self.printer = None

    async def run(self):
        """Run the data analysis pipeline."""
        logger.info(f"Running DataAgentPipeline with experiment_id: {self.experiment_id}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"User prompt: {self.user_prompt}")
        provider = (self.config_dict or {}).get('provider', 'unknown provider')
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
        trace_context = trace(self.workflow_name, metadata=trace_metadata) if self.enable_tracing else nullcontext()

        research_report = ""

        try:
            with trace_context:
                # Create comprehensive research query that includes the task and data context
                with self._span_context(
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
                with self._span_context(
                    agent_span,
                    name=self.researcher.workflow_name,
                    tools=list(self.researcher.tool_agents.keys()),
                ) as span:
                    research_report = await self.researcher.run(
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
                with self._span_context(
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
        provider = (self.config_dict or {}).get('provider', 'unknown provider')
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

    def run_sync(self):
        """Synchronous wrapper for the async run method."""
        return asyncio.run(self.run())


class DataScientist:
    """Manager for the data research workflow that conducts research on a topic by running a continuous research loop."""

    def __init__(
        self,
        max_iterations: int = 5,
        max_time_minutes: int = 10,
        verbose: bool = True,
        tracing: bool = False,
        config: Optional[LLMConfig] = None,
        trace_include_sensitive_data: bool = False,
        workflow_name: Optional[str] = None,
        console: Optional[Console] = None,
        status_printer: Optional[Printer] = None,
    ):
        self.max_iterations = max_iterations
        self.max_time_minutes = max_time_minutes
        self.start_time = None
        self.iteration = 0
        self.conversation = Conversation()
        self.should_continue = True
        self.verbose = verbose
        self.tracing = tracing
        self.trace_include_sensitive_data = trace_include_sensitive_data
        self.workflow_name = workflow_name or "data_researcher_workflow"
        self.config = config
        self.console = console
        self.status_printer = status_printer
        self.constraint_reason: str = ""

        self.evaluate_agent = create_evaluate_agent(self.config)
        self.routing_agent = create_routing_agent(self.config)
        self.observe_agent = create_observe_agent(self.config)
        self.writer_agent = create_writer_agent(self.config)
        self.tool_agents = init_tool_agents(self.config)

    def _span_context(self, span_factory, **kwargs):
        """Return a span context when manager tracing is enabled, else a no-op context."""
        return span_factory(**kwargs) if self.tracing else nullcontext()

    def attach_printer(self, printer: Optional[Printer]) -> None:
        """Attach or detach the live status printer used for progress updates."""
        self.status_printer = printer

    def _update_status(self, suffix: str, message: str, *, done: bool = False) -> None:
        """Send a status update to the live printer if available."""
        if self.status_printer:
            item_id = f"research_{suffix}"
            self.status_printer.update_item(item_id, message, is_done=done)

    async def run(
        self,
        query: str,
        output_length: str = "",
        output_instructions: str = "",
        background_context: str = "",
    ) -> str:
        """Run the deep research workflow for a given query."""
        self.start_time = time.time()
        self.constraint_reason = ""
        self._log_message("=== Starting Data Research Workflow ===")
        self._update_status("loop", "Starting research loop...")
        self._update_status("iteration", "Preparing first iteration...")

        # Iterative research loop
        while self.should_continue and self._check_constraints():
            self.iteration += 1
            self._log_message(f"\n=== Starting Iteration {self.iteration} ===")
            self._update_status("iteration", f"Iteration {self.iteration} in progress...")

            # Set up blank IterationData for this iteration
            self.conversation.add_iteration()

            # 1. Generate observations with a span (only if tracing enabled)
            self._update_status("observe", f"Iteration {self.iteration}: generating observations...")
            with self._span_context(
                agent_span,
                name="observe_agent",
                output_type="observations",
            ) as span:
                observations = await self._generate_observations(query, background_context=background_context)
                if span and hasattr(span, "set_output"):
                    span.set_output({"preview": observations[:200]})
            self._update_status("observe", f"Iteration {self.iteration}: observations captured", done=True)

            # 2. Evaluate current gaps in the research and capture span data
            with self._span_context(
                agent_span,
                name="evaluate_agent",
                output_type=KnowledgeGapOutput.__name__,
            ) as span:
                evaluation: KnowledgeGapOutput = await self._evaluate_gaps(query, background_context=background_context)
                if span and hasattr(span, "set_output"):
                    span.set_output({
                        "research_complete": evaluation.research_complete,
                        "remaining_gaps": len(evaluation.outstanding_gaps),
                    })

            # Check if we should continue or break the loop
            if not evaluation.research_complete:
                next_gap = evaluation.outstanding_gaps[0]

                # 3. Select agents to address knowledge gap with tracing span
                with self._span_context(
                    agent_span,
                    name="routing_agent",
                    output_type=AgentSelectionPlan.__name__,
                ) as span:
                    selection_plan: AgentSelectionPlan = await self._select_agents(next_gap, query, background_context=background_context)
                    if span and hasattr(span, "set_output"):
                        span.set_output({"num_tasks": len(selection_plan.tasks)})

                # 4. Run the selected agents to gather information with function span
                with self._span_context(
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

        # Create final report with function span
        with self._span_context(
            agent_span,
            name="writer_agent",
            output_type="final_report",
        ) as span:
            self._update_status("writer", "Compiling final report...")
            report = await self._create_final_report(query, length=output_length, instructions=output_instructions)
            if span and hasattr(span, "set_output"):
                span.set_output({"report_length": len(report)})
        self._update_status("writer", "Final report ready", done=True)

        elapsed_time = time.time() - self.start_time
        self._log_message(f"DataResearcher completed in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds after {self.iteration} iterations.")

        return report

    def _check_constraints(self) -> bool:
        """Check if we've exceeded our constraints (max iterations or time)."""
        if self.iteration >= self.max_iterations:
            self._log_message("\n=== Ending Research Loop ===")
            self._log_message(f"Reached maximum iterations ({self.max_iterations})")
            self.constraint_reason = f"Reached maximum iterations ({self.max_iterations})"
            return False

        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes >= self.max_time_minutes:
            self._log_message("\n=== Ending Research Loop ===")
            self._log_message(f"Reached maximum time ({self.max_time_minutes} minutes)")
            self.constraint_reason = f"Reached maximum time ({self.max_time_minutes} minutes)"
            return False

        return True

    async def _evaluate_gaps(
        self,
        query: str,
        background_context: str = ""
    ) -> KnowledgeGapOutput:
        """Evaluate the current state of research and identify knowledge gaps."""

        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

        self._update_status("evaluate", f"Iteration {self.iteration}: evaluating knowledge gaps...")
        input_str = f"""
        Current Iteration Number: {self.iteration}
        Time Elapsed: {(time.time() - self.start_time) / 60:.2f} minutes of maximum {self.max_time_minutes} minutes

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
            message = f"Iteration {self.iteration}: evaluation complete — no gaps remaining"
        self._update_status("evaluate", message, done=True)

        return evaluation

    async def _select_agents(
        self,
        gap: str,
        query: str,
        background_context: str = ""
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

        # Add the tool calls to the conversation
        self.conversation.set_latest_tool_calls([
            f"[Agent] {task.agent} [Query] {task.query} [Entity] {task.entity_website if task.entity_website else 'null'}" for task in selection_plan.tasks
        ])

        joined_calls = '\n'.join(self.conversation.history[-1].tool_calls)
        self._log_message(f"<action>\nCalling the following tools to address the knowledge gap:\n{joined_calls}\n</action>")

        task_count = len(selection_plan.tasks)
        message = f"Iteration {self.iteration}: planned {task_count} tool task(s)"
        self._update_status("plan", message, done=True)

        return selection_plan

    async def _execute_tools(self, tasks: List[AgentTask]) -> Dict[str, ToolAgentOutput]:
        """Execute the selected tools concurrently to gather information."""
        # Create a task for each agent
        async_tasks = []
        for task in tasks:
            async_tasks.append(self._run_agent_task(task))

        # Run all tasks concurrently
        num_completed = 0
        results = {}
        total_tasks = len(async_tasks)

        if total_tasks == 0:
            self._update_status("tools", f"Iteration {self.iteration}: no tool executions required", done=True)
            return results

        self._update_status("tools", f"Iteration {self.iteration}: executing tools 0/{total_tasks}")
        for future in asyncio.as_completed(async_tasks):
            gap, agent_name, result = await future
            results[f"{agent_name}_{gap}"] = result
            num_completed += 1
            self._log_message(f"<processing>\nTool execution progress: {num_completed}/{len(async_tasks)}\n</processing>")
            self._update_status(
                "tools",
                f"Iteration {self.iteration}: executing tools {num_completed}/{total_tasks}",
            )

        # Add findings from the tool outputs to the conversation
        findings = []
        for tool_output in results.values():
            findings.append(tool_output.output)
        self.conversation.set_latest_findings(findings)

        joined_findings = '\n\n'.join(findings)
        self._log_message(f"<findings>\n{joined_findings}\n</findings>")
        self._update_status("tools", f"Iteration {self.iteration}: tool execution complete", done=True)

        return results

    async def _run_agent_task(self, task: AgentTask) -> tuple[str, str, ToolAgentOutput]:
        """Run a single agent task and return the result."""
        with self._span_context(
            function_span,
            name=f"agent_task_{task.agent}",
            input=f"gap={task.gap}; query={task.query[:100]}" if task.query else f"gap={task.gap}; query=",
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
                    sources=[]
                )

            return task.gap, agent_name, output
        except Exception as e:
            error_output = ToolAgentOutput(
                output=f"Error executing {task.agent} for gap '{task.gap}': {str(e)}",
                sources=[]
            )
            return task.gap, task.agent, error_output

    async def _generate_observations(self, query: str, background_context: str = "") -> str:
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

        # Add the observations to the conversation
        self.conversation.set_latest_thought(observations)
        self._log_message(f"<thought>\n{observations}\n</thought>")
        return observations

    async def _create_final_report(
        self,
        query: str,
        length: str = "",
        instructions: str = ""
    ) -> str:
        """Create the final response from the completed draft."""

        self._log_message("=== Drafting Final Response ===")

        length_str = f"* The full response should be approximately {length}.\n" if length else ""
        instructions_str = f"* {instructions}" if instructions else ""
        guidelines_str = ("\n\nGUIDELINES:\n" + length_str + instructions_str).strip('\n') if length or instructions else ""

        all_findings = '\n\n'.join(self.conversation.get_all_findings()) or "No findings available yet."

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
        except Exception as e:
            return f"Error generating report: {str(e)}"

    def _log_message(self, message: str) -> None:
        """Log a message if verbose is True"""
        if self.verbose:
            if self.console:
                self.console.print(message)
            else:
                print(message)
