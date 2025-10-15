from __future__ import annotations

from agentz.profiles.base import Profile


# Profile instance for notion agent
notion_profile = Profile(
    instructions="You are a notion agent. Your task is to interact with the notion MCP server following the instructions provided.",
    runtime_template="{instructions}",
    output_schema=None,
    tools=None,
    model=None
)
