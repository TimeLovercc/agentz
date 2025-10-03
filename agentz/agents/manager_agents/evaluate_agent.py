from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.registry import register_agent


class KnowledgeGapOutput(BaseModel):
    """Output model for evaluation of research gaps."""
    research_complete: bool = Field(description="Whether the research is complete")
    outstanding_gaps: List[str] = Field(description="List of outstanding knowledge gaps", default_factory=list)
    reasoning: str = Field(description="Reasoning behind the evaluation", default="")


@register_agent("evaluate_agent", aliases=["evaluate"])
def create_evaluate_agent(config: LLMConfig) -> Agent:
    """Create an evaluation agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for research evaluation
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'evaluate_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('evaluate_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'evaluate_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Research Evaluator",
        instructions=instructions,
        output_type=KnowledgeGapOutput,
        model=config.main_model
    )

    logger.info("Created EvaluateAgent")
    return agent