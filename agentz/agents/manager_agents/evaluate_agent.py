from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig


class KnowledgeGapOutput(BaseModel):
    """Output model for evaluation of research gaps."""
    research_complete: bool = Field(description="Whether the research is complete")
    outstanding_gaps: List[str] = Field(description="List of outstanding knowledge gaps", default_factory=list)
    reasoning: str = Field(description="Reasoning behind the evaluation", default="")


def create_evaluate_agent(config: LLMConfig, full_config: Optional[Dict[str, Any]] = None) -> Agent:
    """Create an evaluation agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration
        full_config: Optional full config dictionary with agent prompts

    Returns:
        Agent instance configured for research evaluation
    """

    # Get instructions from config or use default
    default_instructions = """You are a research evaluation agent. Your role is to assess the current state of research and identify any outstanding knowledge gaps.

Analyze the provided context including:
- The original query/task
- Background context
- Previous iterations, actions, findings, and thoughts

Determine:
1. Whether the research is complete (all necessary information has been gathered)
2. If not complete, what specific knowledge gaps remain
3. Prioritize the gaps by importance

For data science tasks, consider these aspects:
- Data understanding and exploration
- Data preprocessing and cleaning
- Model selection and training
- Performance evaluation
- Results interpretation and insights

Provide a structured evaluation with completion status and specific remaining gaps."""

    instructions = default_instructions
    if full_config:
        instructions = full_config.get('agents', {}).get('evaluate_agent', {}).get('instructions', default_instructions)

    agent = Agent(
        name="Research Evaluator",
        instructions=instructions,
        output_type=KnowledgeGapOutput,
        model=config.main_model
    )

    logger.info("Created EvaluateAgent")
    return agent