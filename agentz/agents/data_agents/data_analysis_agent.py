"""Data Analysis Agent - Perform exploratory data analysis."""

from __future__ import annotations

from typing import Optional

from agents import Agent
from agentz.tools.data_tools import analyze_data
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig


def model_supports_structured_output(model: str) -> bool:
    """Check if model supports structured output."""
    structured_output_models = ["gpt-4", "gpt-3.5", "gemini", "claude"]
    return any(m in model.lower() for m in structured_output_models)


def create_type_parser(output_type):
    """Create a parser for the output type."""
    def parser(response):
        if isinstance(response, str):
            import json
            try:
                data = json.loads(response)
                return output_type(**data)
            except:
                return output_type(output=response, sources=[])
        return response
    return parser


INSTRUCTIONS = f"""
You are an exploratory data analysis specialist. Your task is to analyze data patterns and relationships.

Steps:
1. Use the analyze_data tool with the file path (and target_column if provided)
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
        output_type=ToolAgentOutput if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_structured_output(selected_model) else None
    )
