"""Evaluation Agent - Evaluate machine learning model performance."""

from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import evaluate_model
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser


INSTRUCTIONS = f"""
You are a model evaluation specialist. Your task is to assess model performance comprehensively.

Steps:
1. Use the evaluate_model tool (it automatically uses the currently loaded dataset)
   - Required: target_column (which column was predicted)
   - Optional: model_type (default: random_forest)
   - The tool will evaluate on the dataset that was previously loaded/preprocessed
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
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None
    )
