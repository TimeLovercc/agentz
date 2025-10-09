from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from agentz.agents.base import ResearchAgent as Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent
from agentz.context.behavior_profiles import behavior_profiles


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


@register_agent("routing_agent", aliases=["routing"])
def create_routing_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a routing agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for task routing
    """
    def _merge_spec(input_spec: Optional[dict]) -> dict:
        if input_spec is None:
            return get_agent_spec(cfg, "routing_agent")

        merged = dict(input_spec)
        profile_name = merged.get("profile") or "routing_agent"
        profile = behavior_profiles.get_optional(profile_name)

        params_override = merged.get("params")
        if profile:
            merged.setdefault("instructions", profile.instructions)
            params_override = profile.params_with(params_override)
            merged["profile"] = profile.name
        else:
            params_override = dict(params_override or {})

        if "instructions" not in merged:
            fallback = get_agent_spec(cfg, "routing_agent", required=False)
            if fallback:
                merged["instructions"] = fallback["instructions"]
                base_params = dict(fallback.get("params", {}))
                base_params.update(params_override)
                params_override = base_params

        if "instructions" not in merged:
            raise ValueError("Routing agent requires instructions via profile or config.")

        merged["params"] = params_override
        return merged

    spec = _merge_spec(spec)

    return Agent(
        name="Task Router",
        instructions=spec["instructions"],
        output_type=AgentSelectionPlan,
        model=cfg.llm.main_model,
        **spec.get("params", {})
    )
