from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent


def _always_true(result, ctx):
    """Helper for emit rules - always returns True."""
    return True


@register_agent("observe_agent", aliases=["observe"])
def create_observe_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create an observation agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for research observation
    """
    if spec is None:
        spec = get_agent_spec(cfg, "observe_agent")

    instructions = spec["instructions"]
    params = spec.get("params", {})

    agent = Agent(
        name="Research Observer",
        instructions=instructions,
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
        header = f"Iteration {ctx['iteration']} • Phase: {ctx.get('phase', 'observe')}"
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
        {"type": "thought", "source": "final_text", "when": _always_true},
    ]

    logger.info("Created ObserveAgent")
    return agent