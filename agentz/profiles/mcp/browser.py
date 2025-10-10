from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile


class InstructionsInput(BaseModel):
    """Input schema for instructions-based runtime template."""
    instructions: str = Field(description="The instructions to follow")


# Profile instance for browser agent
browser_profile = Profile(
    instructions="You are a browser agent. Your task is to interact with the browser MCP server following the instructions provided.",
    runtime_template="[[INSTRUCTIONS]]",
    output_schema=None,
    input_schema=InstructionsInput,
    tools=None,
    model=None
)
