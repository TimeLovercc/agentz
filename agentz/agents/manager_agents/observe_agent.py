from __future__ import annotations

from typing import Optional
from loguru import logger

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent


@register_agent("observe_agent", aliases=["observe"])
def create_observe_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create an observation agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for research observation
    """
    if spec is None:
        spec = get_agent_spec(cfg, "observe_agent")

    instructions = spec["instructions"]
    params = spec.get("params", {})

    agent = Agent(
        name="Research Observer",
        instructions=instructions,
        model=cfg.llm.main_model,
        **params
    )

    logger.info("Created ObserveAgent")
    return agent