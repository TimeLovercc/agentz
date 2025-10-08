from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from agentz.agents.base import ResearchAgent as Agent
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

    return Agent(
        name="Research Evaluator",
        instructions=spec["instructions"],
        output_type=KnowledgeGapOutput,
        model=cfg.llm.main_model,
        **spec.get("params", {})
    )
