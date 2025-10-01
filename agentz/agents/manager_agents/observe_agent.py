from __future__ import annotations

from loguru import logger

from agents import Agent
from ds1.src.llm.llm_setup import LLMConfig


def create_observe_agent(config: LLMConfig) -> Agent:
    """Create an observation agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration

    Returns:
        Agent instance configured for research observation
    """

    agent = Agent(
        name="Research Observer",
        instructions="""You are a research observation agent. Your role is to analyze the current state of research and provide thoughtful observations.

Your responsibilities:
1. Reflect on the progress made so far
2. Identify patterns and insights from previous iterations
3. Consider what has been learned and what remains unclear
4. Provide strategic thinking about next steps
5. Generate actionable observations that guide the research process

Analyze the provided context including:
- The original query/task
- Current iteration number and time elapsed
- Background context
- Previous iterations, actions, findings, and thoughts

Provide concise but insightful observations that help guide the research process. Focus on:
- What we've learned so far
- What patterns are emerging
- What areas need deeper investigation
- Strategic recommendations for next steps""",
        model=config.main_model
    )

    logger.info("Created ObserveAgent")
    return agent