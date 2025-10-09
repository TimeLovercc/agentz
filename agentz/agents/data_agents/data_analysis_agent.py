"""Data Analysis Agent - Perform exploratory data analysis."""

from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import analyze_data
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser
from agentz.memory.behavior_profiles import behavior_profiles


@register_agent("data_analysis_agent", aliases=["data_analysis", "analysis"])
def create_data_analysis_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a data analysis agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for data analysis tasks
    """
    selected_model = cfg.llm.main_model
    spec = spec or {}

    profile_name = spec.get("profile") or "data_analysis_agent"
    profile = behavior_profiles.get_optional(profile_name) or behavior_profiles.get("data_analysis_agent")

    instructions = spec.get(
        "instructions",
        profile.render({"OUTPUT_SCHEMA": ToolAgentOutput.model_json_schema()}),
    )
    agent_kwargs = profile.params_with(spec.get("params"))
    for reserved in ("name", "instructions", "tools", "model", "output_type", "output_parser"):
        agent_kwargs.pop(reserved, None)

    return Agent(
        name="Data Analyzer",
        instructions=instructions,
        tools=[analyze_data],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None,
        **agent_kwargs,
    )
