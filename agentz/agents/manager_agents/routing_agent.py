from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent


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
    if spec is None:
        spec = get_agent_spec(cfg, "routing_agent")

    return Agent(
        name="Task Router",
        instructions=spec["instructions"],
        output_type=AgentSelectionPlan,
        model=cfg.llm.main_model,
        **spec.get("params", {})
    )
