from __future__ import annotations

from typing import Any
from loguru import logger

from agents import Runner
from agents.tracing.create import agent_span
from agentz.agents.manager_agents.evaluate_agent import create_evaluate_agent
from agentz.agents.manager_agents.observe_agent import create_observe_agent
from agentz.agents.manager_agents.routing_agent import create_routing_agent
from agentz.agents.manager_agents.writer_agent import create_writer_agent
from agentz.agents.worker_agents.tool_agents import create_tool_agents
from agentz.memory.global_memory import global_memory
from pipelines.base import BasePipeline, with_run_context

class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""
    
    def __init__(self, config):
        super().__init__()
        self.observe_agent = create_observe_agent(config.llm)
        self.evaluate_agent = create_evaluate_agent(config.llm)
        self.routing_agent = create_routing_agent(config.llm)
        self.writer_agent = create_writer_agent(config.llm)
        self.tool_agents = create_tool_agents(config.llm)

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
        while self.should_continue and self._check_constraints():
            iteration_num = self.iteration + 1
            logger.info(f"Starting iteration {iteration_num}")
            self.conversation.add_iteration()

            observations = await self._generate_observations(query=query)
            evaluation = await self._evaluate_research_state(query=query)

            if not evaluation.research_complete:
                next_gap = evaluation.outstanding_gaps[0]
                selection_plan = await self._route_tasks(next_gap, query)
                result = await self._execute_tools(selection_plan)
                
            else:
                logger.info(f"Research marked complete by evaluation agent at iteration {iteration_num}")
                self.should_continue = False

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

        return await self.agent_step(
            agent=self.observe_agent,
            instructions=instructions,
            span_name="generate_observations",
            span_type="function",
        )


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

    def _route_tasks(self, gap: str, query: str) -> Any:
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

    def _dispatch_tool_tasks(self, routing_plan: Any) -> None:
        """Dispatch tasks to tool agents based on routing plan."""

        for task in routing_plan.tasks:
            if task.agent not in self.tool_agents:
                logger.warning(f"Unknown tool agent requested: {task.agent}")
                continue

            tool_agent = self.tool_agents[task.agent]
            Runner.submit_task(tool_agent, task.model_dump_json())

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
            result = await Runner.run(self.manager_agents["writer_agent"], prompt)
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

