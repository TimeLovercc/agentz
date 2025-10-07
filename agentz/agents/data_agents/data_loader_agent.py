from __future__ import annotations

from typing import Optional

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent, ToolAgentOutput


@register_agent("data_loader_agent", aliases=["data_loader"])
def create_data_loader_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a data loader agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for data loading tasks
    """
    if spec is None:
        spec = get_agent_spec(cfg, "data_loader_agent")

    return Agent(
        name="Data Loader",
        instructions=spec["instructions"],
        output_type=ToolAgentOutput,
        model=cfg.llm.main_model,
        **spec.get("params", {})
    )
