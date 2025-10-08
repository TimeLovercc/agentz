"""Chrome Agent - Interact with the chrome browser."""

from __future__ import annotations

from typing import Optional

# print("chrome_agent============")

from agentz.agents.base import ResearchAgent as Agent
from agentz.tools.data_tools import load_dataset
from agentz.agents.registry import register_agent, ToolAgentOutput
from agentz.configuration.base import BaseConfig
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils import create_type_parser
from agentz.mcp.servers.chrome_devtools.server import ChromeDevToolsMCP
from agents.mcp import MCPServer, MCPServerStdio


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
    # server = ChromeDevToolsMCP()

    server = MCPServerStdio(
        cache_tools_list=True,  # Cache the tools list, for demonstration
        params={"command": "npx", "args": ["-y", "chrome-devtools-mcp@latest"]},
    )
    server.connect()
    return Agent(
        name="Chrome",
        instructions=INSTRUCTIONS,
        # tools=[chrome_devtools_mcp],
        mcp_servers=[server],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None
    )