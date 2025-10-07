from __future__ import annotations

from typing import Optional

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent


@register_agent("writer_agent", aliases=["writer"])
def create_writer_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a writer agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for technical writing
    """
    if spec is None:
        spec = get_agent_spec(cfg, "writer_agent")

    return Agent(
        name="Technical Writer",
        instructions=spec["instructions"],
        model=cfg.llm.main_model,
        **spec.get("params", {})
    )
