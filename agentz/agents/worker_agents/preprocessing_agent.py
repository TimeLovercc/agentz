from __future__ import annotations

from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.worker_agents.tool_agents import ToolAgentOutput


def create_preprocessing_agent(config: LLMConfig) -> Agent:
    """Create a preprocessing agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for data preprocessing tasks
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'preprocessing_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('preprocessing_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'preprocessing_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Preprocessing",
        instructions=instructions,
        output_type=ToolAgentOutput,
        model=config.main_model
    )

    logger.info("Created PreprocessingAgent")
    return agent
