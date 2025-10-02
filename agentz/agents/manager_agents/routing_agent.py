from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig


class AgentTask(BaseModel):
    """Task definition for routing to specific agents."""
    agent: str = Field(description="Name of the agent to use")
    query: str = Field(description="Query/task for the agent")
    gap: str = Field(description="The knowledge gap this task addresses")
    entity_website: Optional[str] = Field(description="Optional entity or website context", default=None)


class AgentSelectionPlan(BaseModel):
    """Plan containing multiple agent tasks to address knowledge gaps."""
    tasks: List[AgentTask] = Field(description="List of tasks for different agents", default_factory=list)
    reasoning: str = Field(description="Reasoning for the agent selection", default="")


def create_routing_agent(config: LLMConfig) -> Agent:
    """Create a routing agent using OpenAI Agents SDK.

    Args:
        config: LLM configuration with full_config containing agent prompts

    Returns:
        Agent instance configured for task routing
    """

    if not config.full_config:
        raise ValueError("Agent instructions for 'routing_agent' not found in config. Please provide config_file with agent instructions.")

    instructions = config.full_config.get('agents', {}).get('routing_agent', {}).get('instructions')
    if not instructions:
        raise ValueError("Agent instructions for 'routing_agent' not found in config. Please provide config_file with agent instructions.")

    agent = Agent(
        name="Task Router",
        instructions=instructions,
        output_type=AgentSelectionPlan,
        model=config.main_model
    )

    logger.info("Created RoutingAgent")
    return agent