from __future__ import annotations

import asyncio
import time
from typing import Any, List
from loguru import logger

from agentz.agents.manager_agents.routing_agent import AgentTask
from agentz.agents.registry import create_agents
from agentz.flow import auto_trace
from agentz.memory.global_memory import global_memory
from agentz.memory.conversation import Conversation
from pipelines.base import BasePipeline

# TODO: Merge this into the DataScientistPipeline
class DataScientistMemoryPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(self, config):
        super().__init__(config)

        # Setup manager agents - check if they're already Agent instances or need to be created
        self.observe_agent = create_agents("observe_agent", config)
        self.evaluate_agent = create_agents("evaluate_agent", config)
        self.routing_agent = create_agents("routing_agent", config)
        self.writer_agent = create_agents("writer_agent", config)
        self.memory_agent = create_agents("memory_agent", config)

        # Create worker agents - these are typically from registry
        tool_agent_names = [
            "data_loader_agent",
            "data_analysis_agent",
            "preprocessing_agent",
            "model_training_agent",
            "evaluation_agent",
            "visualization_agent",
            "code_generation_agent",
        ]
        self.tool_agents = create_agents(tool_agent_names, config)
        self.conversation = Conversation()

    @auto_trace
    async def run(self):
        """Run the data analysis pipeline."""
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"User prompt: {self.config.prompt}")
        self.iteration = 0

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
            self.iteration += 1
            logger.info(f"Starting iteration {self.iteration}")
            self.conversation.add_iteration()

            # Start iteration group
            if self.printer:
                self.printer.start_group(
                    f"iter-{self.iteration}",
                    title=f"Iteration {self.iteration}",
                    border_style="white"
                )

            await self._generate_observations(query=query)
            evaluation = await self._evaluate_research_state(query=query)

            if not evaluation.research_complete:
                next_gap = evaluation.outstanding_gaps[0]
                self.conversation.set_latest_gap(next_gap)
                selection_plan = await self._route_tasks(next_gap, query)
                await self._execute_tools(selection_plan.tasks)
                await self._compress_memory(query=query)
                # End iteration group
                if self.printer:
                    self.printer.end_group(f"iter-{self.iteration}", is_done=True)
            else:
                logger.info(f"Research marked complete by evaluation agent at iteration {self.iteration}")
                # End iteration group before exiting
                if self.printer:
                    self.printer.end_group(f"iter-{self.iteration}", is_done=True)
                self.should_continue = False

        # Create final report in its own group
        if self.printer:
            self.printer.start_group(
                "iter-final",
                title="Final Report",
                border_style="white"
            )

        research_report = await self._create_final_report()

        if self.printer:
            self.printer.end_group("iter-final", is_done=True)

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
        {self.conversation.compile_conversation_history_with_summary() or "No previous actions, findings or thoughts available."}
        """

        result = await self.agent_step(
            agent=self.observe_agent,
            instructions=instructions,
            span_name="generate_observations",
            span_type="function",
            printer_key="observe",
            printer_title="Observations",
            printer_group_id=f"iter-{self.iteration}",
        )

        # Store observations as thought in conversation
        self.conversation.set_latest_thought(result.final_output)
        

        return result

    async def _compress_memory(self, query: str) -> Any:
        """Compress the conversation history to a memory with constant size."""
        instructions = f"""
        You are at the end of iteration {self.iteration}. You need to generate a comprehensive and useful summary.

        ORIGINAL QUERY:
        {query}

        LAST SUMMARY:
        {self.conversation.get_latest_summary() or "No previous summary available."}

        CONVERSATION HISTORY:
        {self.conversation.compile_unsummarized_conversation_history() or "No previous unsummarized actions, findings or thoughts available."}
        """
        result = await self.agent_step(
            agent=self.memory_agent, 
            instructions=instructions,
            span_name="update_memory",
            span_type="function",
            printer_key="memory",
            printer_title="Memory",
            printer_group_id=f"iter-{self.iteration}",
            output_model=self.memory_agent.output_type,
        )
        
        self.conversation.set_latest_summary(result.summary)
        return result

    async def _evaluate_research_state(self, *, query: str) -> Any:
        """Evaluate whether research is complete and identify gaps."""
        
        instructions = f"""
        Current Iteration Number: {self.iteration}
        Time Elapsed: {(time.time() - self.start_time) / 60:.2f} minutes of maximum {self.max_time_minutes} minutes

        ORIGINAL QUERY:
        {query}

        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history_with_summary() or "No previous actions, findings or thoughts available."}
        """

        result = await self.agent_step(
            agent=self.evaluate_agent,
            instructions=instructions,
            span_name="evaluate_research_state",
            span_type="function",
            output_model=self.evaluate_agent.output_type,
            printer_key="evaluate",
            printer_title="Evaluation",
            printer_group_id=f"iter-{self.iteration}",
        )
        
        evaluation = result
        
        if not evaluation.research_complete:
            next_gap = evaluation.outstanding_gaps[0]
            self.conversation.set_latest_gap(next_gap)
            
        return evaluation

    async def _route_tasks(self, gap: str, query: str) -> Any:
        """Route tasks to the appropriate specialized agents."""

        instructions = f"""
        ORIGINAL QUERY:
        {query}

        KNOWLEDGE GAP TO ADDRESS:
        {gap}


        HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
        {self.conversation.compile_conversation_history_with_summary() or "No previous actions, findings or thoughts available."}
        """

        selection_plan = await self.agent_step(
            agent=self.routing_agent,
            instructions=instructions,
            span_name="route_tasks",
            span_type="tool",
            output_model=self.routing_agent.output_type,
            printer_key="route",
            printer_title="Routing",
            printer_group_id=f"iter-{self.iteration}",
        )

        self.conversation.set_latest_tool_calls([
            f"[Agent] {task.agent} [Query] {task.query} [Entity] {task.entity_website if task.entity_website else 'null'}" for task in selection_plan.tasks
        ])
        return selection_plan

    async def _execute_tools(self, tasks: List[Any]) -> List[str]:
        """Execute tool agent tasks and collect results.

        Args:
            tasks: List of AgentTask objects from routing plan

        Returns:
            List of findings from tool executions
        """
        async_tasks = []
        for task in tasks:
            async_tasks.append(self._run_agent_task(task))

        num_completed = 0
        results = {}
        for future in asyncio.as_completed(async_tasks):
            gap, agent_name, result = await future
            results[f"{agent_name}_{gap}"] = result
            num_completed += 1
            self.update_printer(f"tool:{agent_name}", f"Completed {agent_name}", is_done=True)

        findings = []
        for tool_output in results.values():
            findings.append(tool_output.output)
        self.conversation.set_latest_findings(findings)
        return findings

    async def _run_agent_task(self, task: AgentTask) -> tuple[str, str, str]:
        """Run a single agent task and return the result."""
        try:
            agent_name = task.agent
            agent = self.tool_agents.get(agent_name)
            if agent:
                result = await self.agent_step(
                    agent=agent,
                    instructions=task.model_dump_json(),
                    span_name=task.agent,
                    span_type="tool",
                    printer_key=f"tool:{task.agent}",
                    printer_title=f"Tool: {task.agent}",
                    printer_group_id=f"iter-{self.iteration}",
                )
                # Extract output from result
                output = result.final_output if hasattr(result, 'final_output') else str(result)
            else:
                output = f"No implementation found for agent {agent_name}"

            return task.gap, agent_name, output
        except Exception as e:
            error_output = f"Error executing {task.agent} for gap '{task.gap}': {str(e)}"
            return task.gap, task.agent, error_output

    async def _create_final_report(
        self,
        length: str = "",
        instructions: str = ""
    ) -> str:
        """Generate the final report using the writer agent."""
        logger.info("Drafting final response")

        length_str = f"* The full response should be approximately {length}.\n" if length else ""
        instructions_str = f"* {instructions}" if instructions else ""
        guidelines_str = ("\n\nGUIDELINES:\n" + length_str + instructions_str).strip('\n') if length or instructions else ""

        all_findings = '\n\n'.join(self.conversation.get_all_findings()) or "No findings available yet."

        prompt = f"""
        Provide a response based on the query and findings below with as much detail as possible.{guidelines_str}

        QUERY: {self.config.prompt}

        DATASET: {self.config.data_path}

        FINDINGS:
        {all_findings}
        """

        result = await self.agent_step(
            agent=self.writer_agent,
            instructions=prompt,
            span_name="writer_agent",
            span_type="agent",
            printer_key="writer",
            printer_title="Writer",
            printer_group_id="iter-final",
        )

        logger.info("Final response created successfully")
        return result.final_output

    async def _finalise_research(self, research_report: str) -> None:
        """Finalize logging, persist conversation, and update memory."""

        timestamped_report = (
            f"Experiment {self.experiment_id}\n\n{research_report.strip()}"
        )
        global_memory.store(
            key=f"report_{self.experiment_id}",
            value=timestamped_report,
            tags=["research_report"]
        )

        logger.info(f"Stored research report with experiment_id {self.experiment_id}")
        self.update_printer("writer_agent", "Final report generated", is_done=True)
