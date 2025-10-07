"""Data Loader Agent - Load and inspect datasets."""

from __future__ import annotations

from typing import Optional

from agents import Agent
from agentz.tools.data_tools import load_dataset
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig


INSTRUCTIONS = f"""
You are a data loading specialist. Your task is to load and inspect datasets.

Steps:
1. Use the load_dataset tool with the provided file path
2. The tool returns: shape, columns, dtypes, missing values, sample data, statistics, memory usage, duplicates
3. Write a 2-3 paragraph summary covering:
   - Dataset size and structure
   - Data types and columns
   - Data quality issues (missing values, duplicates)
   - Key statistics and initial observations

Include specific numbers and percentages in your summary.

Output JSON only following this schema:
{ToolAgentOutput.model_json_schema()}
"""


@register_agent("data_loader_agent", aliases=["data_loader"])
def create_data_loader_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a data loader agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for data loading tasks
    """
    selected_model = cfg.llm.main_model

    return Agent(
        name="Data Loader",
        instructions=INSTRUCTIONS,
        tools=[load_dataset],
        model=selected_model,
        output_type=ToolAgentOutput
    )
