from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field

from agentz.profiles.base import Profile


class EvaluateOutput(BaseModel):
    """Output schema for evaluate agent."""
    research_complete: bool = Field(description="Boolean indicating if research is done")
    outstanding_gaps: List[str] = Field(description="List of specific gaps that still need addressing", default_factory=list)
    reasoning: str = Field(description="Clear explanation of the evaluation")


# Profile instance for evaluate agent
evaluate_profile = Profile(
    instructions="""You are a research evaluation agent. Analyze research progress and determine if goals have been met.

Your responsibilities:
1. Assess whether the research task has been completed
2. Identify any remaining knowledge gaps
3. Provide clear reasoning for your evaluation
4. Suggest specific next steps if research is incomplete

Evaluate the research state and provide structured output with:
- research_complete: boolean indicating if research is done
- outstanding_gaps: list of specific gaps that still need addressing
- reasoning: clear explanation of your evaluation""",
    runtime_template="""Current Iteration Number: {iteration}
Time Elapsed: {elapsed_minutes} minutes of maximum {max_minutes} minutes

ORIGINAL QUERY:
{query}

HISTORY OF ACTIONS, FINDINGS AND THOUGHTS:
{history}""",
    output_schema=EvaluateOutput,
    tools=None,
    model=None
)
