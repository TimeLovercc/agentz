"""Vanilla Chat Agent Profile - Simple conversational agent without tools."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile


class ChatInput(BaseModel):
    """Input schema for chat messages."""
    message: str = Field(description="The user's message")


class ChatOutput(BaseModel):
    """Output schema for chat responses."""
    response: str = Field(description="The assistant's response")


# Profile instance for vanilla chat agent
vanilla_chat_profile = Profile(
    instructions="""You are a helpful AI assistant. Engage in natural conversation with the user.

GUIDELINES:
- Be helpful, harmless, and honest
- Provide clear and concise responses
- If you don't know something, admit it
- Be conversational and friendly
- Stay on topic and address the user's questions directly

""",
    runtime_template="User: [[MESSAGE]]",
    output_schema=ChatOutput,
    input_schema=ChatInput,
    tools=None,  # No tools for vanilla chat
    model=None
)
