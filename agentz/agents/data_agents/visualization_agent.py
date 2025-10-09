"""Visualization Agent - Create data visualizations."""

from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import create_visualization
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser
from agentz.memory.behavior_profiles import behavior_profiles


@register_agent("visualization_agent", aliases=["visualization", "viz"])
def create_visualization_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a visualization agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for visualization tasks
    """
    selected_model = cfg.llm.main_model
    spec = spec or {}

    profile_name = spec.get("profile") or "visualization_agent"
    profile = behavior_profiles.get_optional(profile_name) or behavior_profiles.get("visualization_agent")

    instructions = spec.get(
        "instructions",
        profile.render({"OUTPUT_SCHEMA": ToolAgentOutput.model_json_schema()}),
    )
    agent_kwargs = profile.params_with(spec.get("params"))
    for reserved in ("name", "instructions", "tools", "model", "output_type", "output_parser"):
        agent_kwargs.pop(reserved, None)

    return Agent(
        name="Data Visualizer",
        instructions=instructions,
        tools=[create_visualization],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None,
        **agent_kwargs,
    )
