from __future__ import annotations

import asyncio
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from loguru import logger

from agents import Runner
from agents.tracing.create import trace, agent_span, function_span
from src.agents.manager_agents.evaluate_agent import create_evaluate_agent, KnowledgeGapOutput
from src.agents.manager_agents.observe_agent import create_observe_agent
from src.agents.manager_agents.routing_agent import create_routing_agent, AgentTask, AgentSelectionPlan
from src.agents.manager_agents.writer_agent import create_writer_agent
from src.agents.manager_agents.tool_agents import init_tool_agents, ToolAgentOutput
from src.memory.global_memory import global_memory
from src.utils import get_experiment_timestamp
from ds1.src.llm.llm_setup import LLMConfig
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
        llm_config: Optional[LLMConfig] = None
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
            llm_config=llm_config
        )

        self.experiment_id = get_experiment_timestamp()
        self.workflow_name = workflow_name or f"data_science_pipeline_{self.experiment_id}"

        # Initialize data researcher with tracing configuration
        self.researcher = DataScientist(
            max_iterations=5,
            max_time_minutes=10,
            verbose=True,
            tracing=enable_tracing,
            config=self.config,
            trace_include_sensitive_data=trace_include_sensitive_data,
            workflow_name=f"researcher_{self.experiment_id}"
        )

        # Handle tracing configuration and setup with user-friendly defaults
        self._setup_tracing()

        logger.info(f"Initialized DataAgentPipeline with experiment_id: {self.experiment_id}, tracing: {enable_tracing}, sensitive_data: {trace_include_sensitive_data}")

    def _setup_tracing(self):
        """Setup tracing configuration with user-friendly output."""
        if self.enable_tracing:
            print("🍌 Starting Data Science Pipeline with Tracing")
            print(f"📊 Data: {self.data_path}")
            print(f"🔧 Provider: {self.config_dict['provider']}")
            print(f"🤖 Model: {self.config.model_name}")
            print(f"📋 Task: {self.user_prompt[:100]}...")
            print(f"🔍 Tracing: Enabled")
            print(f"🔒 Sensitive Data in Traces: {'Yes' if self.trace_include_sensitive_data else 'No'}")
            print(f"🏷️ Workflow: {self.workflow_name}")
        else:
            print("🍌 Starting Data Science Pipeline")
            print(f"📊 Data: {self.data_path}")
            print(f"🔧 Provider: {self.config_dict['provider']}")
            print(f"🤖 Model: {self.config.model_name}")

    def _span_context(self, span_factory, **kwargs):
        """Return a span context when tracing is enabled, else a no-op context."""
        return span_factory(**kwargs) if self.enable_tracing else nullcontext()

    async def run(self):
        """Run the data analysis pipeline."""
        logger.info(f"Running DataAgentPipeline with experiment_id: {self.experiment_id}")
        logger.info(f"Data path: {self.data_path}")
        logger.info(f"User prompt: {self.user_prompt}")
        logger.info(f"Provider: {self.config_dict['provider']}, Model: {self.config.model_name}")

        trace_metadata = {
            "experiment_id": self.experiment_id,
            "includes_sensitive_data": "true" if self.trace_include_sensitive_data else "false",
        }
        trace_context = trace(self.workflow_name, metadata=trace_metadata) if self.enable_tracing else nullcontext()

        with trace_context:
            # Create comprehensive research query that includes the task and data context
            with self._span_context(
                function_span,
                name="prepare_research_query",
                input=f"experiment_id={self.experiment_id}, data_path={self.data_path}, provider={self.config_dict['provider']}",
            ) as span:
                research_query = self._prepare_research_query()
                if span and hasattr(span, "set_output"):
                    span.set_output({"query_length": len(research_query)})

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
                    background_context=f"Experiment: {self.experiment_id}, Dataset: {self.data_path}, Provider: {self.config_dict['provider']}, Model: {self.config.model_name}"
                )
                if span and hasattr(span, "set_output"):
                    span.set_output({"report_preview": research_report[:200]})

            print("=== Final Analysis Report ===")
            print(research_report)

            if self.enable_tracing:
                print("\n🎉 Example completed successfully!")
                print("📈 Analysis complete - check the detailed report above")
                print("🔍 Trace data has been captured for this pipeline execution")
                print("   - Pipeline spans: initialization, research workflow, completion")
                print("   - Research spans: observations, gap evaluation, agent selection, tool execution")
                print("   - Individual agent execution spans for detailed analysis")

            # Store final results with span
            with self._span_context(
                function_span,
                name="store_final_results",
                input=f"experiment_id={self.experiment_id}",
            ) as span:
                self._store_final_results(research_report)
                if span and hasattr(span, "set_output"):
                    span.set_output({"stored_bytes": len(research_report.encode('utf-8'))})

            logger.info("DataAgentPipeline completed successfully")
            return research_report


    def _prepare_research_query(self) -> str:
        """Prepare the research query and store initial task information."""
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
                "provider": self.config_dict['provider'],
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
        workflow_name: Optional[str] = None
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

        self.evaluate_agent = create_evaluate_agent(self.config)
        self.routing_agent = create_routing_agent(self.config)
        self.observe_agent = create_observe_agent(self.config)
        self.writer_agent = create_writer_agent(self.config)
        self.tool_agents = init_tool_agents(self.config)

    def _span_context(self, span_factory, **kwargs):
        """Return a span context when manager tracing is enabled, else a no-op context."""
        return span_factory(**kwargs) if self.tracing else nullcontext()

    async def run(
        self,
        query: str,
        output_length: str = "",
        output_instructions: str = "",
        background_context: str = "",
    ) -> str:
        """Run the deep research workflow for a given query."""
        self.start_time = time.time()
        self._log_message("=== Starting Data Research Workflow ===")

        # Iterative research loop
        while self.should_continue and self._check_constraints():
            self.iteration += 1
            self._log_message(f"\n=== Starting Iteration {self.iteration} ===")

            # Set up blank IterationData for this iteration
            self.conversation.add_iteration()

            # 1. Generate observations with a span (only if tracing enabled)
            with self._span_context(
                agent_span,
                name="observe_agent",
                output_type="observations",
            ) as span:
                observations = await self._generate_observations(query, background_context=background_context)
                if span and hasattr(span, "set_output"):
                    span.set_output({"preview": observations[:200]})

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
            else:
                self.should_continue = False

        self._log_message("=== DataResearcher Marked As Complete - Finalizing Output ===")

        # Create final report with function span
        with self._span_context(
            agent_span,
            name="writer_agent",
            output_type="final_report",
        ) as span:
            report = await self._create_final_report(query, length=output_length, instructions=output_instructions)
            if span and hasattr(span, "set_output"):
                span.set_output({"report_length": len(report)})

        elapsed_time = time.time() - self.start_time
        self._log_message(f"DataResearcher completed in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds after {self.iteration} iterations.")

        return report

    def _check_constraints(self) -> bool:
        """Check if we've exceeded our constraints (max iterations or time)."""
        if self.iteration >= self.max_iterations:
            self._log_message("\n=== Ending Research Loop ===")
            self._log_message(f"Reached maximum iterations ({self.max_iterations})")
            return False

        elapsed_minutes = (time.time() - self.start_time) / 60
        if elapsed_minutes >= self.max_time_minutes:
            self._log_message("\n=== Ending Research Loop ===")
            self._log_message(f"Reached maximum time ({self.max_time_minutes} minutes)")
            return False

        return True

    async def _evaluate_gaps(
        self,
        query: str,
        background_context: str = ""
    ) -> KnowledgeGapOutput:
        """Evaluate the current state of research and identify knowledge gaps."""

        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

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

        return evaluation

    async def _select_agents(
        self,
        gap: str,
        query: str,
        background_context: str = ""
    ) -> AgentSelectionPlan:
        """Select agents to address the identified knowledge gap."""

        background = f"BACKGROUND CONTEXT:\n{background_context}" if background_context else ""

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
        for future in asyncio.as_completed(async_tasks):
            gap, agent_name, result = await future
            results[f"{agent_name}_{gap}"] = result
            num_completed += 1
            self._log_message(f"<processing>\nTool execution progress: {num_completed}/{len(async_tasks)}\n</processing>")

        # Add findings from the tool outputs to the conversation
        findings = []
        for tool_output in results.values():
            findings.append(tool_output.output)
        self.conversation.set_latest_findings(findings)

        joined_findings = '\n\n'.join(findings)
        self._log_message(f"<findings>\n{joined_findings}\n</findings>")

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
            print(message)
