from __future__ import annotations

from typing import Optional

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent, ToolAgentOutput


@register_agent("code_generation_agent", aliases=["code_generation", "codegen"])
def create_code_generation_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a code generation agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for code generation tasks
    """
    if spec is None:
        spec = get_agent_spec(cfg, "code_generation_agent")

    return Agent(
        name="Code Generator",
        instructions=spec["instructions"],
        output_type=ToolAgentOutput,
        model=cfg.llm.main_model,
        **spec.get("params", {})
    )
