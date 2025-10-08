"""Preprocessing Agent - Clean and transform datasets."""

from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import preprocess_data
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser



INSTRUCTIONS = f"""
You are a data preprocessing specialist. Your task is to clean and transform datasets.

Available operations:
- handle_missing: Fill missing values (mean/median/mode)
- remove_duplicates: Remove duplicate rows
- encode_categorical: Encode categorical variables
- scale_standard: Z-score normalization
- scale_minmax: Min-max scaling [0, 1]
- remove_outliers: IQR method
- feature_engineering: Create interaction features

Steps:
1. Use the preprocess_data tool (it automatically uses the currently loaded dataset)
   - Required: operations list (which operations to perform)
   - Optional: target_column (if mentioned in the task)
   - The tool will preprocess the dataset that was previously loaded
2. The tool returns: operations applied, shape changes, summary of changes
3. Write a 2-3 paragraph summary covering:
   - Operations performed and justification
   - Shape changes and data modifications
   - Impact on data quality
   - Next steps (modeling, further preprocessing)

Include specific numbers (rows removed, values filled, etc.).

Output JSON only following this schema:
{ToolAgentOutput.model_json_schema()}
"""


@register_agent("preprocessing_agent", aliases=["preprocessing", "preprocess"])
def create_preprocessing_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a preprocessing agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for preprocessing tasks
    """
    selected_model = cfg.llm.main_model

    return Agent(
        name="Data Preprocessor",
        instructions=INSTRUCTIONS,
        tools=[preprocess_data],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None
    )
