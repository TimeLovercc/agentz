from __future__ import annotations

from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.registry import register_agent, ToolAgentOutput


@register_agent("data_analysis_agent", aliases=["data_analysis", "analysis"])
def create_data_analysis_agent(config: LLMConfig) -> Agent:
    """Create a data analysis agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for data analysis tasks
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'data_analysis_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('data_analysis_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'data_analysis_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Data Analysis",
        instructions=instructions,
        output_type=ToolAgentOutput,
        model=config.main_model
    )

    logger.info("Created DataAnalysisAgent")
    return agent
