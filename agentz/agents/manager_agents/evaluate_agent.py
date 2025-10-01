from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
from loguru import logger

from agents import Agent
from ds1.src.llm.llm_setup import LLMConfig


class KnowledgeGapOutput(BaseModel):
    """Output model for evaluation of research gaps."""
    research_complete: bool = Field(description="Whether the research is complete")
    outstanding_gaps: List[str] = Field(description="List of outstanding knowledge gaps", default_factory=list)
    reasoning: str = Field(description="Reasoning behind the evaluation", default="")


def create_evaluate_agent(config: LLMConfig) -> Agent:
    """Create an evaluation agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration

    Returns:
        Agent instance configured for research evaluation
    """

    agent = Agent(
        name="Research Evaluator",
        instructions="""You are a research evaluation agent. Your role is to assess the current state of research and identify any outstanding knowledge gaps.

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

Provide a structured evaluation with completion status and specific remaining gaps.""",
        output_type=KnowledgeGapOutput,
        model=config.main_model
    )

    logger.info("Created EvaluateAgent")
    return agent