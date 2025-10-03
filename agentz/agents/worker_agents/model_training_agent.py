from __future__ import annotations

from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.worker_agents.tool_agents import ToolAgentOutput


def create_model_training_agent(config: LLMConfig) -> Agent:
    """Create a model training agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for model training tasks
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'model_training_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('model_training_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'model_training_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Model Training",
        instructions=instructions,
        output_type=ToolAgentOutput,
        model=config.main_model
    )

    logger.info("Created ModelTrainingAgent")
    return agent
