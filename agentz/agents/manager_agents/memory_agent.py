# Memory_Compress_Prompt = """
# You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive summary that will help answer the question.

# The last summary serves as your starting point, marking the information landscape previously collected. Your role is to:
# - Analyze progress made since the last summary
# - Identify remaining information gaps
# - Generate a useful summary that combines previous and new information
# - Maintain continuity, especially when recent conversation history is limited

# Task Guidelines

# 1. Information Analysis:
#    - Carefully analyze the conversation history to identify truly useful information.
#    - Focus on information that directly contributes to answering the question.
#    - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated.
#    - If information is missing or unclear, do NOT include it in your summary.
#    - Use the last summary as a baseline when recent history is sparse.

# 2. Summary Requirements:
#    - Extract only the most relevant information that is explicitly present in the conversation.
#    - Synthesize information from multiple exchanges when relevant.
#    - Only include information that is certain and clearly stated.
#    - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed.

# 3. Output Format: Your response should be structured as follows:
# <summary>
# - Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question.]
# </summary>

# Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

# Question
# {{{question}}}

# Last Summary
# {{{last_summary}}}

# Conversation History
# {{{recent_history_messages}}}

# Please generate a comprehensive and useful summary. Note that you are not permitted to invoke tools during this process.
# """

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

from agentz.agents.base import ResearchAgent as Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent


class MemoryAgentOutput(BaseModel):
    """Plan containing multiple agent tasks to address knowledge gaps."""
    summary: str = Field(description="Summary of the conversation history", default="")

@register_agent("memory_agent", aliases=["memory"])
def create_memory_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a memory agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for memory
    """
    if spec is None:
        spec = get_agent_spec(cfg, "memory_agent")

    return Agent(
        name="Memory Agent",
        instructions=spec["instructions"],
        model=cfg.llm.main_model,
        output_type=MemoryAgentOutput,
        **spec.get("params", {})
    )
