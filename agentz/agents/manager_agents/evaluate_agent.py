from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from loguru import logger

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent


class KnowledgeGapOutput(BaseModel):
    """Output model for evaluation of research gaps."""
    research_complete: bool = Field(description="Whether the research is complete")
    outstanding_gaps: List[str] = Field(description="List of outstanding knowledge gaps", default_factory=list)
    reasoning: str = Field(description="Reasoning behind the evaluation", default="")


@register_agent("evaluate_agent", aliases=["evaluate"])
def create_evaluate_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create an evaluation agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for research evaluation
    """
    if spec is None:
        spec = get_agent_spec(cfg, "evaluate_agent")

    instructions = spec["instructions"]
    params = spec.get("params", {})

    agent = Agent(
        name="Research Evaluator",
        instructions=instructions,
        output_type=KnowledgeGapOutput,
        model=cfg.llm.main_model,
        **params
    )

    # Add instruction template
    agent.instructions_template = """{header}

ORIGINAL QUERY:
{query}

{gap_block}

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
{history}
"""

    # Add prepare_instructions method
    def prepare_instructions(self, ctx: dict) -> str:
        header = f"Iteration {ctx['iteration']} • Phase: {ctx.get('phase', 'evaluate')}"
        header += f"\nTime Elapsed: {ctx.get('minutes_elapsed', 0):.2f} minutes of maximum {ctx.get('max_time_minutes', 10)} minutes"
        gap_block = f"KNOWLEDGE GAP TO ADDRESS:\n{ctx['gap']}\n" if ctx.get("gap") else "No specific gap provided.\n"
        return self.instructions_template.format(
            header=header,
            query=ctx["query"],
            gap_block=gap_block,
            history=ctx["history"] or "No previous actions, findings or thoughts available.",
        )

    # Bind method to agent
    import types
    agent.prepare_instructions = types.MethodType(prepare_instructions, agent)

    # Add emit rules
    agent.emits: List[Dict[str, Any]] = [
        {
            "type": "gap",
            "source": "path",
            "path": "outstanding_gaps[0]",
            "when": lambda r, ctx: not getattr(r, "research_complete", False),
        }
    ]

    return agent