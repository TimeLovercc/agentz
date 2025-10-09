from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent
from agentz.memory.behavior_profiles import behavior_profiles


@register_agent("observe_agent", aliases=["observe"])
def create_observe_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create an observation agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for research observation
    """
    def _merge_spec(input_spec: Optional[dict]) -> dict:
        if input_spec is None:
            return get_agent_spec(cfg, "observe_agent")

        merged = dict(input_spec)
        profile_name = merged.get("profile") or "observe_agent"
        profile = behavior_profiles.get_optional(profile_name)

        params_override = merged.get("params")
        if profile:
            merged.setdefault("instructions", profile.instructions)
            params_override = profile.params_with(params_override)
            merged["profile"] = profile.name
        else:
            params_override = dict(params_override or {})

        if "instructions" not in merged:
            fallback = get_agent_spec(cfg, "observe_agent", required=False)
            if fallback:
                merged["instructions"] = fallback["instructions"]
                base_params = dict(fallback.get("params", {}))
                base_params.update(params_override)
                params_override = base_params

        if "instructions" not in merged:
            raise ValueError("Observe agent requires instructions via profile or config.")

        merged["params"] = params_override
        return merged

    spec = _merge_spec(spec)

    return Agent(
        name="Research Observer",
        instructions=spec["instructions"],
        model=cfg.llm.main_model,
        **spec.get("params", {})
    )
