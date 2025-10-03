from __future__ import annotations

from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.registry import register_agent


@register_agent("writer_agent", aliases=["writer"])
def create_writer_agent(config: LLMConfig) -> Agent:
    """Create a writer agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for technical writing
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'writer_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('writer_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'writer_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Technical Writer",
        instructions=instructions,
        model=config.main_model
    )

    logger.info("Created WriterAgent")
    return agent