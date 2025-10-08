"""Data Loader Agent - Load and inspect datasets."""

from __future__ import annotations

from typing import Optional

from agents import Agent
from agentz.tools.data_tools import load_dataset
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.mcp.servers.chrome_devtools import ChromeDevToolsMCP


INSTRUCTIONS = f"""
You are a chrome agent. Your task is to interact with the chrome browser following the instructions provided.
"""


@register_agent("chrome_agent", aliases=["chrome"])
def create_chrome_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a chrome agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for chrome tasks
    """
    selected_model = cfg.llm.main_model
    server = ChromeDevToolsMCP()

    return Agent(
        name="Chrome",
        instructions=INSTRUCTIONS,
        # tools=[chrome_devtools_mcp],
        mcp_servers=[server],
        model=selected_model,
        output_type=ToolAgentOutput
    )
