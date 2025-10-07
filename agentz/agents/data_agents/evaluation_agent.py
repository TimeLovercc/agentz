"""Evaluation Agent - Evaluate machine learning model performance."""

from __future__ import annotations

from typing import Optional

from agents import Agent
from agentz.tools.data_tools import evaluate_model
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig


INSTRUCTIONS = f"""
You are a model evaluation specialist. Your task is to assess model performance comprehensively.

Steps:
1. Use the evaluate_model tool with file path, target_column, and model_type
2. The tool returns:
   - Classification: accuracy, precision, recall, F1, confusion matrix, per-class metrics, CV results
   - Regression: R², RMSE, MAE, MAPE, error analysis, CV results
3. Write a 3+ paragraph summary covering:
   - Overall performance with key metrics
   - Confusion matrix or error distribution analysis
   - Per-class/per-feature insights
   - Cross-validation and generalization
   - Model strengths and weaknesses
   - Improvement recommendations
   - Production readiness

Include specific numbers and identify weak areas.

Output JSON only following this schema:
{ToolAgentOutput.model_json_schema()}
"""


@register_agent("evaluation_agent", aliases=["evaluation", "eval_tool"])
def create_evaluation_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create an evaluation agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for model evaluation tasks
    """
    selected_model = cfg.llm.main_model

    return Agent(
        name="Model Evaluator",
        instructions=INSTRUCTIONS,
        tools=[evaluate_model],
        model=selected_model,
        output_type=ToolAgentOutput
    )
