"""Visualization Agent - Create data visualizations."""

from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import create_visualization
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser



INSTRUCTIONS = f"""
You are a data visualization specialist. Your task is to create insightful visualizations.

Plot types:
- distribution: Histograms for numerical columns
- correlation: Heatmap for feature relationships
- scatter: 2D relationship plot (needs 2 columns)
- box: Outlier detection
- bar: Categorical data comparison
- pairplot: Pairwise relationships

Steps:
1. Use the create_visualization tool (it automatically uses the currently loaded dataset)
   - Required: plot_type (which type of visualization to create)
   - Optional: columns (which columns to include), target_column (for coloring)
   - The tool will visualize the dataset that was previously loaded/preprocessed
2. The tool returns: plot type, columns plotted, output path, visual insights
3. Write a 2-3 paragraph summary covering:
   - Visualization type and purpose
   - Key patterns observed
   - Data interpretation and context
   - Actionable recommendations
   - Suggestions for additional plots

Include specific observations (correlation values, outlier %, distribution shapes).

Output JSON only following this schema:
{ToolAgentOutput.model_json_schema()}
"""


@register_agent("visualization_agent", aliases=["visualization", "viz"])
def create_visualization_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a visualization agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for visualization tasks
    """
    selected_model = cfg.llm.main_model

    return Agent(
        name="Data Visualizer",
        instructions=INSTRUCTIONS,
        tools=[create_visualization],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None
    )
