from __future__ import annotations

from agentz.profiles.base import Profile


# Profile instance for browser agent
browser_profile = Profile(
    instructions="You are a browser agent. Your task is to interact with the browser MCP server following the instructions provided.",
    runtime_template="{instructions}",
    output_schema=None,
    tools=None,
    model=None
)
