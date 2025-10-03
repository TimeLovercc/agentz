from __future__ import annotations

from typing import Dict, List, Union
from loguru import logger

from agents import Agent
from agentz.llm.llm_setup import LLMConfig
from agentz.agents.manager_agents.base import MANAGER_AGENT_FACTORIES
from agentz.agents.worker_agents.base import AGENT_FACTORIES as WORKER_AGENT_FACTORIES


# Combine all agent factories (manager + worker)
ALL_AGENT_FACTORIES = {
    **MANAGER_AGENT_FACTORIES,
    **WORKER_AGENT_FACTORIES,
}


def create_agents(
    agent_names: Union[str, List[str]],
    config: LLMConfig
) -> Union[Agent, Dict[str, Agent]]:
    """Create one or more agents from agent name(s).

    Supports both manager agents (observe, evaluate, routing, writer) and
    worker agents (data_loader, data_analysis, preprocessing, etc.).

    Args:
        agent_names: Single agent name (str) or list of agent names
        config: LLM configuration with full_config containing agent prompts

    Returns:
        - If agent_names is a string: returns a single Agent instance
        - If agent_names is a list: returns Dict mapping agent names to Agent instances

    Raises:
        ValueError: If an unknown agent name is provided

    Examples:
        # Create single agent
        observe_agent = create_agents("observe_agent", config)

        # Create multiple agents
        agents = create_agents(["observe_agent", "data_loader_agent"], config)
        observe_agent = agents["observe_agent"]
    """
    # Handle single agent name (string input)
    if isinstance(agent_names, str):
        agent_name = agent_names
        if agent_name not in ALL_AGENT_FACTORIES:
            available = list(ALL_AGENT_FACTORIES.keys())
            raise ValueError(
                f"Unknown agent name: {agent_name}. "
                f"Available agents: {available}"
            )

        factory = ALL_AGENT_FACTORIES[agent_name]
        agent = factory(config)
        logger.info(f"Created agent: {agent_name}")
        return agent

    # Handle multiple agent names (list input)
    agents = {}
    for agent_name in agent_names:
        if agent_name not in ALL_AGENT_FACTORIES:
            available = list(ALL_AGENT_FACTORIES.keys())
            raise ValueError(
                f"Unknown agent name: {agent_name}. "
                f"Available agents: {available}"
            )

        factory = ALL_AGENT_FACTORIES[agent_name]
        agents[agent_name] = factory(config)
        logger.info(f"Created agent: {agent_name}")

    return agents
