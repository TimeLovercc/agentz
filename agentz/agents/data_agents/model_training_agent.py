"""Model Training Agent - Train machine learning models."""

from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import train_model
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser
from agentz.memory.behavior_profiles import behavior_profiles


@register_agent("model_training_agent", aliases=["model_training", "train"])
def create_model_training_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a model training agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for model training tasks
    """
    selected_model = cfg.llm.main_model
    spec = spec or {}

    profile_name = spec.get("profile") or "model_training_agent"
    profile = behavior_profiles.get_optional(profile_name) or behavior_profiles.get("model_training_agent")

    instructions = spec.get(
        "instructions",
        profile.render({"OUTPUT_SCHEMA": ToolAgentOutput.model_json_schema()}),
    )
    agent_kwargs = profile.params_with(spec.get("params"))
    for reserved in ("name", "instructions", "tools", "model", "output_type", "output_parser"):
        agent_kwargs.pop(reserved, None)

    return Agent(
        name="Model Trainer",
        instructions=instructions,
        tools=[train_model],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None,
        **agent_kwargs,
    )
