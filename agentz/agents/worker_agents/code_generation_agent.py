from __future__ import annotations

from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.registry import register_agent, ToolAgentOutput


@register_agent("code_generation_agent", aliases=["code_generation", "codegen"])
def create_code_generation_agent(config: LLMConfig) -> Agent:
    """Create a code generation agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for code generation tasks
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'code_generation_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('tool_agents', {}).get('code_generation_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'code_generation_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Code Generation",
        instructions=instructions,
        output_type=ToolAgentOutput,
        model=config.main_model
    )

    logger.info("Created CodeGenerationAgent")
    return agent
