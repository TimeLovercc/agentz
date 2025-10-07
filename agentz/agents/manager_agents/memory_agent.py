Memory_Compress_Prompt = """
You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive summary that will help answer the question.

The last summary serves as your starting point, marking the information landscape previously collected. Your role is to:
- Analyze progress made since the last summary
- Identify remaining information gaps
- Generate a useful summary that combines previous and new information
- Maintain continuity, especially when recent conversation history is limited

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

3. Output Format: Your response should be structured as follows:
<summary>
- Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question.]
</summary>

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

Question
{{{question}}}

Last Summary
{{{last_summary}}}

Conversation History
{{{recent_history_messages}}}

Please generate a comprehensive and useful summary. Note that you are not permitted to invoke tools during this process.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from loguru import logger

from agents import Agent
from agentz.configuration.base import BaseConfig, get_agent_spec
from agentz.agents.registry import register_agent


# def _always_true(result, ctx):
#     """Helper for emit rules - always returns True."""
#     return True


@register_agent("observe_agent", aliases=["observe"])
def create_memory_agent(cfg: BaseConfig, spec: Optional[dict] = None) -> Agent:
    """Create a memory agent using OpenAI Agents SDK.

    Args:
        cfg: Base configuration
        spec: Optional agent spec with {instructions, params}

    Returns:
        Agent instance configured for research observation
    """
    if spec is None:
        spec = get_agent_spec(cfg, "memory_agent")

    instructions = spec["instructions"]
    params = spec.get("params", {})

    agent = Agent(
        name="Memory Agent",
        instructions=instructions,
        model=cfg.llm.main_model,
        **params
    )

    # Add instruction template
    agent.instructions_template = """{header}

ORIGINAL QUERY:
{query}

{gap_block}

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
{history}
"""

    # Add prepare_instructions method
    def prepare_instructions(self, ctx: dict) -> str:
        header = f"Iteration {ctx['iteration']} • Phase: {ctx.get('phase', 'memory')}"
        gap_block = f"KNOWLEDGE GAP TO ADDRESS:\n{ctx['gap']}\n" if ctx.get("gap") else "No specific gap provided.\n"
        return self.instructions_template.format(
            header=header,
            query=ctx["query"],
            gap_block=gap_block,
            history=ctx["history"] or "No previous actions, findings or thoughts available.",
        )

    # Bind method to agent
    import types
    agent.prepare_instructions = types.MethodType(prepare_instructions, agent)

    # Add emit rules
    agent.emits: List[Dict[str, Any]] = [
        {"type": "thought", "source": "final_text", "when": _always_true},
    ]

    return agent