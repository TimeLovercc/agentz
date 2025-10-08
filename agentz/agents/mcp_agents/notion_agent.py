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
from agents.mcp import MCPServer, MCPServerStdio, MCPServerSse
from agents import HostedMCPTool
import asyncio


INSTRUCTIONS = f"""
You are a notion agent. Your task is to interact with the notion following the instructions provided.
"""

async def get_notion_server():
    async with MCPServerSse(
        name = "Notion",
        params = {
            "url": "https://mcp.notion.com/mcp"
        }
    ) as server:
        return server

@register_agent("notion_agent", aliases=["notion"])
def create_notion_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a notion agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for chrome tasks
    """
    selected_model = cfg.llm.main_model
    server = get_notion_server()

    return Agent(
        name="Notion",
        instructions=INSTRUCTIONS,
        mcp_servers=[server],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None
    )