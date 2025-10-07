"""Visualization Agent - Create data visualizations."""

from __future__ import annotations

from typing import Optional

from agents import Agent
from agentz.tools.data_tools import create_visualization
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig


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
1. Use the create_visualization tool with file path, plot_type, optional columns/target_column
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
        output_type=ToolAgentOutput
    )
