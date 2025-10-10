from __future__ import annotations

from typing import Optional
from loguru import logger

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.profiles.registry import register_agent, ToolAgentOutput


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

    logger.info("Created ModelTrainingAgent")
    return agent
