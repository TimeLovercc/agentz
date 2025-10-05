from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent, ToolAgentOutput


def _always_true(result, ctx):
    """Helper for emit rules - always returns True."""
    return True


@register_agent("model_training_agent", aliases=["model_training", "train"])
def create_model_training_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a model training agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for model training tasks
    """
    if spec is None:
        spec = get_agent_spec(cfg, "model_training_agent")

    instructions = spec["instructions"]
    params = spec.get("params", {})

    agent = Agent(
        name="Model Trainer",
        instructions=instructions,
        output_type=ToolAgentOutput,
        model=cfg.llm.main_model,
        **params
    )

    # Add instruction template
    agent.instructions_template = """{header}

TASK:
{task_json}

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
{history}
"""

    # Add prepare_instructions method
    def prepare_instructions(self, ctx: dict) -> str:
        header = f"Iteration {ctx['iteration']} • Phase: {ctx.get('phase', 'tool')}"
        task_json = ctx.get("extra", {}).get("task_json", "{}")
        return self.instructions_template.format(
            header=header,
            task_json=task_json,
            history=ctx["history"] or "No previous actions, findings or thoughts available.",
        )

    # Bind method to agent
    import types
    agent.prepare_instructions = types.MethodType(prepare_instructions, agent)

    # Add emit rules (tool agents add their findings)
    agent.emits: List[Dict[str, Any]] = [
        {"type": "findings", "source": "final_text", "wrap_list": True, "when": _always_true},
    ]

    logger.info("Created ModelTrainingAgent")
    return agent
