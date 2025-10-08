"""Model Training Agent - Train machine learning models."""

from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import train_model
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser



INSTRUCTIONS = f"""
You are a machine learning specialist. Your task is to train and evaluate models.

Model types:
- auto: Auto-detect best model
- random_forest: Random Forest (classification/regression)
- logistic_regression: Logistic Regression
- linear_regression: Linear Regression
- decision_tree: Decision Tree

Steps:
1. Use the train_model tool with file path, target_column, model_type (default: auto)
2. The tool returns: model type, problem type, train/test scores, CV results, feature importance, predictions
3. Write a 3+ paragraph summary covering:
   - Model selection and problem type
   - Train/test performance with interpretation
   - Cross-validation results and stability
   - Top feature importances
   - Overfitting/underfitting analysis
   - Improvement recommendations

Include specific metrics (accuracy, R², CV mean±std).

Output JSON only following this schema:
{ToolAgentOutput.model_json_schema()}
"""


@register_agent("model_training_agent", aliases=["model_training", "train"])
def create_model_training_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a model training agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for model training tasks
    """
    selected_model = cfg.llm.main_model

    return Agent(
        name="Model Trainer",
        instructions=INSTRUCTIONS,
        tools=[train_model],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None
    )
