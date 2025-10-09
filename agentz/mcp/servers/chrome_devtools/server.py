import asyncio
import shutil

from agents import Agent, Runner, trace
from agents.mcp import MCPServer, MCPServerStdio


def ChromeDevToolsMCP():
    server = MCPServerStdio(
        cache_tools_list=True,  # Cache the tools list, for demonstration
        params={"command": "npx", "args": ["-y", "chrome-devtools-mcp@latest"]},
    ) 
    return server
      