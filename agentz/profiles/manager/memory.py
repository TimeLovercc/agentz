from __future__ import annotations

from pydantic import BaseModel, Field

from agentz.profiles.base import Profile


class MemoryInput(BaseModel):
    """Input schema for memory agent runtime template."""
    iteration: int = Field(description="Current iteration number")
    query: str = Field(description="Original user query")
    last_summary: str = Field(description="Last generated summary")
    conversation_history: str = Field(description="Recent conversation history")


class MemoryAgentOutput(BaseModel):
    """Output schema for memory agent."""
    summary: str = Field(description="Summary of the conversation history", default="")


# Profile instance for memory agent
memory_profile = Profile(
    instructions="""You are a memory agent. Your role is to store and retrieve information from the conversation history.

Your responsibilities:
1. Thoroughly evaluate the conversation history and current question
2. Provide a comprehensive summary that will help answer the question.
3. Analyze progress made since the last summary
4. Generate a useful summary that combines previous and new information
5. Maintain continuity, especially when recent conversation history is limited

Task Guidelines

1. Information Analysis:
  - Carefully analyze the conversation history to identify truly useful information.
  - Focus on information that directly contributes to answering the question.
  - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated.
  - If information is missing or unclear, do NOT include it in your summary.
  - Use the last summary as a baseline when recent history is sparse.

2. Summary Requirements:
  - Extract only the most relevant information that is explicitly present in the conversation.
  - Synthesize information from multiple exchanges when relevant.
  - Only include information that is certain and clearly stated.
  - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed.

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.""",
    runtime_template="""You are at the end of iteration [[ITERATION]]. You need to generate a comprehensive and useful summary.

ORIGINAL QUERY:
[[QUERY]]

LAST SUMMARY:
[[LAST_SUMMARY]]

CONVERSATION HISTORY:
[[CONVERSATION_HISTORY]]""",
    output_schema=MemoryAgentOutput,
    input_schema=MemoryInput,
    tools=None,
    model=None
)
