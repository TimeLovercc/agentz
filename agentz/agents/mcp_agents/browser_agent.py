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
from agents.mcp import MCPServer, MCPServerStdio, MCPServerSse
from agents import HostedMCPTool
import asyncio
from agentz.context.behavior_profiles import behavior_profiles

async def get_browser_server():
    async with MCPServerStdio(
        name = "Browser",
        params = {
            "command": "npx",
            "args": ["-y", "@browsermcp/mcp@latest"]
        }
    ) as server:
        return server

@register_agent("browser_agent", aliases=["browser"])
def create_browser_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a browser agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for browser tasks
    """
    selected_model = cfg.llm.main_model
    spec = spec or {}
    server = get_browser_server()

    profile_name = spec.get("profile") or "browser_agent"
    profile = behavior_profiles.get_optional(profile_name) or behavior_profiles.get("browser_agent")

    instructions = spec.get("instructions", profile.render())
    agent_kwargs = profile.params_with(spec.get("params"))
    for reserved in ("name", "instructions", "tools", "model", "output_type", "output_parser", "mcp_servers"):
        agent_kwargs.pop(reserved, None)

    return Agent(
        name="Browser",
        instructions=instructions,
        mcp_servers=[server],
        model=selected_model,
        output_type=ToolAgentOutput if model_supports_json_and_tool_calls(selected_model) else None,
        output_parser=create_type_parser(ToolAgentOutput) if not model_supports_json_and_tool_calls(selected_model) else None,
        **agent_kwargs,
    )
