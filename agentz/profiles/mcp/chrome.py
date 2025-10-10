from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile


class InstructionsInput(BaseModel):
    """Input schema for instructions-based runtime template."""
    instructions: str = Field(description="The instructions to follow")


# Profile instance for chrome agent
chrome_profile = Profile(
    instructions="You are a chrome agent. Your task is to interact with the chrome browser following the instructions provided.",
    runtime_template="[[INSTRUCTIONS]]",
    output_schema=None,
    input_schema=InstructionsInput,
    tools=None,
    model=None
)
