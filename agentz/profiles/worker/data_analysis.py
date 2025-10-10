from __future__ import annotations

from typing import Optional
from loguru import logger

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.profiles.registry import register_agent, ToolAgentOutput


@register_agent("data_analysis_agent", aliases=["data_analysis", "analysis"])
def create_data_analysis_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a data analysis agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for data analysis tasks
    """
    if spec is None:
        spec = get_agent_spec(cfg, "data_analysis_agent")

    instructions = spec["instructions"]
    params = spec.get("params", {})

    agent = Agent(
        name="Data Analyzer",
        instructions=instructions,
        output_type=ToolAgentOutput,
        model=cfg.llm.main_model,
        **params
    )

    logger.info("Created DataAnalysisAgent")
    return agent
