
from pydantic import BaseModel, Field


class DefaultAgentOutput(BaseModel):
    """Generic fallback output schema for agents without specialised designs."""

    content: str = Field(
        default="",
        description="Primary textual response produced by the agent.",
    )


__all__ = ["DefaultAgentOutput"]
