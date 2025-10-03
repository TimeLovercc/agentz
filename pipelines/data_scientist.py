from __future__ import annotations

import time
from typing import Any, List
from loguru import logger

from agents import Runner
from agents.tracing.create import agent_span
from agentz.agents.registry import create_agents
from agentz.memory.global_memory import global_memory
from pipelines.base import BasePipeline, with_run_context

class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""
    
    def __init__(self, config):
        super().__init__()

        # Create manager agents
        self.observe_agent = create_agents("observe_agent", config.llm)
        self.evaluate_agent = create_agents("evaluate_agent", config.llm)
        self.routing_agent = create_agents("routing_agent", config.llm)
        self.writer_agent = create_agents("writer_agent", config.llm)

        # Create worker agents
        self.tool_agents = create_agents([
            "data_loader_agent",
            "data_analysis_agent",
            "preprocessing_agent",
            "model_training_agent",
            "evaluation_agent",
            "visualization_agent",
            "code_generation_agent",
        ], config.llm)

    @with_run_context
    async def run(self):
        """Run the data analysis pipeline."""
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"User prompt: {self.config.prompt}")

        # Prepare research query
        query = self.prepare_query(
            content=f"Task: {self.config.prompt}\n"
                f"Dataset path: {self.config.data_path}\n"
                "Provide a comprehensive data science workflow"
        )

        # Run research workflow
        self.update_printer("research", "Executing research workflow...")
        self.start_time = time.time()
        while self.should_continue and self._check_constraints():
            iteration_num = self.iteration + 1
            logger.info(f"Starting iteration {iteration_num}")
            self.conversation.add_iteration()

            observations = await self._generate_observations(query=query)
            evaluation = await self._evaluate_research_state(query=query)

            if not evaluation.research_complete:
                next_gap = evaluation.outstanding_gaps[0]
                self.conversation.set_latest_gap(next_gap)
                selection_plan = await self._route_tasks(next_gap, query)
                results = await self._execute_tools(selection_plan.tasks)

            else:
                logger.info(f"Research marked complete by evaluation agent at iteration {iteration_num}")
                self.should_continue = False

            self.iteration += 1

        research_report = await self._create_final_report()

        self.update_printer("research", "Research workflow complete", is_done=True)
        logger.info("Research workflow completed")
        return research_report

    async def _generate_observations(self, query: str) -> Any:
        """Generate observations based on the query."""

        instructions = f"""
        You are starting iteration {self.iteration} of your research process.

        ORIGINAL QUERY:
        {query}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}
        """

        result = await self.agent_step(
            agent=self.observe_agent,
            instructions=instructions,
            span_name="generate_observations",
            span_type="function",
        )

        # Store observations as thought in conversation
        if hasattr(result, 'final_output'):
            self.conversation.set_latest_thought(result.final_output)

        return result


    async def _evaluate_research_state(self, *, query: str) -> Any:
        """Evaluate whether research is complete and identify gaps."""
        
        instructions = f"""
        Current Iteration Number: {self.iteration}
        Time Elapsed: {(self.time.time() - self.start_time) / 60:.2f} minutes of maximum {self.max_time_minutes} minutes

        ORIGINAL QUERY:
        {query}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}        
        """

        return await self.agent_step(
            agent=self.evaluate_agent,
            instructions=instructions,
            span_name="evaluate_research_state",
            span_type="function",
        )

    async def _route_tasks(self, gap: str, query: str) -> Any:
        """Route tasks to the appropriate specialized agents."""

        instructions = f"""
        ORIGINAL QUERY:
        {query}

        KNOWLEDGE GAP TO ADDRESS:
        {gap}


        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history() or "No previous actions, findings or thoughts available."}
        """

        return await self.agent_step(
            agent=self.routing_agent,
            instructions=instructions,
            span_name="route_tasks",
            span_type="tool",
        )

    async def _execute_tools(self, tasks: List[Any]) -> List[str]:
        """Execute tool agent tasks and collect results.

        Args:
            tasks: List of AgentTask objects from routing plan

        Returns:
            List of findings from tool executions
        """
        findings = []
        tool_calls = []

        for task in tasks:
            if task.agent not in self.tool_agents:
                logger.warning(f"Unknown tool agent requested: {task.agent}")
                continue

            tool_agent = self.tool_agents[task.agent]
            logger.info(f"Executing {task.agent} for gap: {task.gap}")

            # Track the tool call
            tool_calls.append(f"{task.agent}: {task.query}")

            result = await self.agent_step(
                agent=tool_agent,
                instructions=task.query,
                span_name=task.agent,
                span_type="tool",
            )

            # Extract output from result
            if hasattr(result, 'final_output'):
                finding = result.final_output
            else:
                finding = str(result)

            findings.append(finding)
            logger.info(f"Completed {task.agent}")

        # Store tool calls and findings in conversation history
        self.conversation.set_latest_tool_calls(tool_calls)
        self.conversation.set_latest_findings(findings)

        return findings

    async def _create_final_report(self) -> str:
        """Generate the final report using the writer agent."""

        collated_findings = "\n".join(self.conversation.get_all_findings())
        prompt = (
            "Create a final research report summarizing the findings, "
            "methodology, and recommendations.\n\n"
            f"Dataset: {self.config.data_path}\n"
            f"Task: {self.config.prompt}\n"
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
        self.update_printer("writer_agent", "Final report generated", is_done=True)

