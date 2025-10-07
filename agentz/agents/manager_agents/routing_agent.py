from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from loguru import logger

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

    instructions = spec["instructions"]
    params = spec.get("params", {})

    agent = Agent(
        name="Task Router",
        instructions=instructions,
        output_type=AgentSelectionPlan,
        model=cfg.llm.main_model,
        **params
    )

    # Add instruction template
    agent.instructions_template = """{header}

ORIGINAL QUERY:
{query}

{gap_block}

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
{history}
"""

    # Add prepare_instructions method
    def prepare_instructions(self, ctx: dict) -> str:
        header = f"Iteration {ctx['iteration']} • Phase: {ctx.get('phase', 'route')}"
        gap_block = f"KNOWLEDGE GAP TO ADDRESS:\n{ctx['gap']}\n" if ctx.get("gap") else "No specific gap provided.\n"
        return self.instructions_template.format(
            header=header,
            query=ctx["query"],
            gap_block=gap_block,
            history=ctx["history"] or "No previous actions, findings or thoughts available.",
        )

    # Bind method to agent
    import types
    agent.prepare_instructions = types.MethodType(prepare_instructions, agent)

    # Add emit rules
    agent.emits: List[Dict[str, Any]] = [
        {
            "type": "tool_calls",
            "source": "path",
            "path": "tasks",
            "format": "[Agent] {item.agent} [Query] {item.query} [Entity] {item.entity_website or 'null'}",
        }
    ]

    return agent
