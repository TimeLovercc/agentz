from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger

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

    instructions = spec["instructions"]
    params = spec.get("params", {})

    agent = Agent(
        name="Technical Writer",
        instructions=instructions,
        model=cfg.llm.main_model,
        **params
    )

    # Add instruction template
    agent.instructions_template = """{header}

ORIGINAL QUERY:
{query}

{gap_block}

FINDINGS:
{findings_text}
"""

    # Add prepare_instructions method
    def prepare_instructions(self, ctx: dict) -> str:
        header = f"Iteration {ctx['iteration']} • Phase: {ctx.get('phase', 'writer')}"
        gap_block = f"KNOWLEDGE GAP TO ADDRESS:\n{ctx['gap']}\n" if ctx.get("gap") else "No specific gap provided.\n"
        findings_text = ctx.get("extra", {}).get("findings_text", "No findings available yet.")
        return self.instructions_template.format(
            header=header,
            query=ctx["query"],
            gap_block=gap_block,
            findings_text=findings_text,
        )

    # Bind method to agent
    import types
    agent.prepare_instructions = types.MethodType(prepare_instructions, agent)

    # Add emit rules (writer does not update conversation by default)
    agent.emits: List[Dict[str, Any]] = []

    logger.info("Created WriterAgent")
    return agent