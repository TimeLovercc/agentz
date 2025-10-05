from __future__ import annotations

import asyncio
import time
from loguru import logger

from agentz.agents.registry import create_agents
from agentz.flow import AgentExecutor, AgentStep, PrinterConfig, with_run_context
from agentz.memory.global_memory import global_memory
from pipelines.base import BasePipeline

class DataScientistPipeline(BasePipeline):
    """Main pipeline orchestrator for data analysis tasks using iterative research."""

    def __init__(self, config):
        super().__init__(config)

        # Setup manager agents - check if they're already Agent instances or need to be created
        self.observe_agent = create_agents("observe_agent", config)
        self.evaluate_agent = create_agents("evaluate_agent", config)
        self.routing_agent = create_agents("routing_agent", config)
        self.writer_agent = create_agents("writer_agent", config)

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

    def _extract_by_path(self, obj, path: str):
        """Generic extractor supporting dotted paths and [index] lookups."""
        if not path:
            return None
        cur = obj
        for token in path.replace("]", "").split("."):
            if not token:
                continue
            if "[" in token:
                name, idx = token.split("[", 1)
                if name:
                    cur = getattr(cur, name, cur[name] if isinstance(cur, dict) else None)
                try:
                    cur = cur[int(idx)]
                except Exception:
                    return None
            else:
                cur = getattr(cur, token, cur.get(token) if isinstance(cur, dict) else None)
            if cur is None:
                return None
        return cur

    def _apply_conversation_emits(self, *, agent, result, final_text: str, ctx: dict):
        """Apply agent.emits rules to self.conversation in a generic way."""
        emits = getattr(agent, "emits", []) or []
        for rule in emits:
            if callable(rule.get("when")) and not rule["when"](result, ctx):
                continue
            ev_type = rule.get("type")
            source = rule.get("source", "final_text")
            data = None

            if source == "final_text":
                data = final_text
            elif source == "path":
                data = self._extract_by_path(result, rule.get("path", ""))

            if ev_type == "thought" and isinstance(data, str):
                self.conversation.set_latest_thought(data)

            elif ev_type == "gap" and isinstance(data, str):
                self.conversation.set_latest_gap(data)

            elif ev_type == "tool_calls":
                items = data or []
                fmt = rule.get("format")
                if fmt:
                    rendered = []
                    for item in items:
                        try:
                            rendered.append(fmt.format(item=item))
                        except Exception:
                            rendered.append(str(item))
                    self.conversation.set_latest_tool_calls(rendered)
                else:
                    self.conversation.set_latest_tool_calls([str(x) for x in items])

            elif ev_type == "findings":
                if data is None:
                    continue
                if isinstance(data, list):
                    self.conversation.set_latest_findings([str(x) for x in data])
                else:
                    self.conversation.set_latest_findings([str(data)] if rule.get("wrap_list", False) else [str(data)])

    async def _agent_step_generic(
        self,
        *,
        agent,
        span_name: str,
        printer_key: str,
        printer_title: str,
        output_model=None,
        phase: str | None = None,
        extra: dict | None = None,
    ):
        """Generic agent step execution with automatic conversation updates."""
        minutes_elapsed = (time.time() - self.start_time) / 60 if hasattr(self, "start_time") else 0.0
        ctx = {
            "phase": phase or getattr(agent, "name", "default"),
            "iteration": self.iteration,
            "query": self.prepare_query(
                content=f"Task: {self.config.prompt}\nDataset path: {self.config.data_path}\nProvide a comprehensive data science workflow"
            ),
            "history": self.conversation.compile_conversation_history(),
            "gap": getattr(self.conversation, "latest_gap", None),
            "minutes_elapsed": minutes_elapsed,
            "max_time_minutes": self.max_time_minutes,
            "extra": extra or {},
        }
        instructions = agent.prepare_instructions(ctx) if hasattr(agent, "prepare_instructions") else ctx["query"]

        step = AgentStep(
            agent=agent,
            instructions=instructions,
            span_name=span_name,
            output_model=output_model or getattr(agent, "output_type", None),
            printer_config=PrinterConfig(key=printer_key, title=printer_title),
        )
        executor = AgentExecutor(self.execution_context)
        result = await executor.execute_step(step)

        final_text = getattr(result, "final_output", None) or getattr(result, "output", None) or str(result)

        # APPLY GENERIC EMIT RULES (no phase branching):
        self._apply_conversation_emits(agent=agent, result=result, final_text=final_text, ctx=ctx)

        return result

    @with_run_context
    async def run(self):
        """Run the data analysis pipeline."""
        logger.info(f"Data path: {self.config.data_path}")
        logger.info(f"User prompt: {self.config.prompt}")
        self.iteration = 0
        self.start_time = time.time()
        self.update_printer("research", "Executing research workflow...")

        while self.should_continue and self._check_constraints():
            self.iteration += 1
            self.conversation.add_iteration()

            await self._agent_step_generic(
                agent=self.observe_agent,
                phase="observe",
                span_name="generate_observations",
                printer_key="observe",
                printer_title="Observations",
            )

            evaluation = await self._agent_step_generic(
                agent=self.evaluate_agent,
                phase="evaluate",
                span_name="evaluate_research_state",
                printer_key="evaluate",
                printer_title="Evaluation",
                output_model=getattr(self.evaluate_agent, "output_type", None),
            )
            if getattr(evaluation, "research_complete", False):
                break

            selection_plan = await self._agent_step_generic(
                agent=self.routing_agent,
                phase="route",
                span_name="route_tasks",
                printer_key="route",
                printer_title="Routing",
                output_model=getattr(self.routing_agent, "output_type", None),
            )

            tasks = getattr(selection_plan, "tasks", [])
            async def _run_tool(t):
                tool_agent = self.tool_agents.get(t.agent)
                if not tool_agent:
                    self.update_printer(f"tool:{t.agent}", f"No implementation for {t.agent}", is_done=True)
                    return
                await self._agent_step_generic(
                    agent=tool_agent,
                    phase=f"tool:{t.agent}",
                    span_name=t.agent,
                    printer_key=f"tool:{t.agent}",
                    printer_title=f"Tool: {t.agent}",
                    extra={"task_json": t.model_dump_json()},
                )
                self.update_printer(f"tool:{t.agent}", f"Completed {t.agent}", is_done=True)

            await asyncio.gather(*[_run_tool(t) for t in tasks])

        # Writer
        all_findings = "\n\n".join(self.conversation.get_all_findings()) or "No findings available yet."
        writer_extra = {"findings_text": all_findings}
        writer_result = await self._agent_step_generic(
            agent=self.writer_agent,
            phase="writer",
            span_name="writer_agent",
            printer_key="writer",
            printer_title="Writer",
            extra=writer_extra,
        )
        final_text = getattr(writer_result, "final_output", None) or getattr(writer_result, "output", None) or str(writer_result)
        self.update_printer("research", "Research workflow complete", is_done=True)
        return final_text

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

