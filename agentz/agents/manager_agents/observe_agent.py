from __future__ import annotations

from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.registry import register_agent


@register_agent("observe_agent", aliases=["observe"])
def create_observe_agent(config: LLMConfig) -> Agent:
    """Create an observation agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for research observation
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'observe_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('observe_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'observe_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Research Observer",
        instructions=instructions,
        model=config.main_model
    )

    logger.info("Created ObserveAgent")
    return agent