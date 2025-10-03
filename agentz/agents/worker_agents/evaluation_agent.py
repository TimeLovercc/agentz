from __future__ import annotations

from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.registry import register_agent, ToolAgentOutput


@register_agent("evaluation_agent", aliases=["evaluation", "eval_tool"])
def create_evaluation_agent(config: LLMConfig) -> Agent:
    """Create an evaluation agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for model evaluation tasks
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'evaluation_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('evaluation_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'evaluation_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Evaluation",
        instructions=instructions,
        output_type=ToolAgentOutput,
        model=config.main_model
    )

    logger.info("Created EvaluationAgent")
    return agent
