"""Data Analysis Agent - Perform exploratory data analysis."""

from __future__ import annotations

from typing import Optional

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import analyze_data
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser


INSTRUCTIONS = f"""
You are an exploratory data analysis specialist. Your task is to analyze data patterns and relationships.

Steps:
1. Use the analyze_data tool (it automatically uses the currently loaded dataset)
   - If a target_column is mentioned in the task, pass it as a parameter
   - The tool will analyze the dataset that was previously loaded
2. The tool returns: distributions, correlations, outliers (IQR method), patterns, recommendations
3. Write a 3+ paragraph summary covering:
   - Key statistical insights (means, medians, distributions)
   - Important correlations (>0.7) and relationships
   - Outlier percentages and potential impact
   - Data patterns and anomalies identified
   - Preprocessing recommendations based on findings

Include specific numbers, correlation values, and percentages.

Output JSON only following this schema:
{ToolAgentOutput.model_json_schema()}
"""


@register_agent("data_analysis_agent", aliases=["data_analysis", "analysis"])
def create_data_analysis_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a data analysis agent.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for data analysis tasks
    """
    selected_model = cfg.llm.main_model

    return Agent(
        name="Data Analyzer",
        instructions=INSTRUCTIONS,
        tools=[analyze_data],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None
    )