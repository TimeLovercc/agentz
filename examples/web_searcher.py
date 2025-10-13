"""Advanced usage of the DataScientistPipeline with custom manager agents.

This example demonstrates the new one-parameter API for DataScientistPipeline:
- Use 'config_path' in the dict to load a base YAML file
- Deep-merge additional config on top (dict wins over file)
- Agents can be provided as Agent objects or dicts with name/instructions
"""

from agents import Agent

from pipelines.web_searcher import WebSearcherPipeline

# DATA_PATH = "data/banana_quality.csv"
USER_PROMPT = (
    "Find the outstanding papers of ACL 2025, extract their title, author list, keywords, abstract, url in one sentence."
)

EVALUATE_AGENT = Agent(
    name="Focused Evaluator",
    instructions=(
        "Review research progress with an emphasis on data readiness, model risk, and "
        "missing validation steps. Respond using the EvaluateOutput schema in JSON."
    ),
)

ROUTING_AGENT = Agent(
    name="Priority Router",
    instructions=(
        "Analyse outstanding gaps and assign tasks to tool agents. Ensure preprocessing "
        "and evaluation steps are scheduled before model retraining. Reply as JSON that "
        "conforms to AgentSelectionPlan."
    ),
)

OBSERVE_AGENT = Agent(
    name="Insight Observer",
    instructions=(
        "Summarise iteration progress, highlight risks, and note pending follow-up items "
        "before the next iteration begins."
    ),
)

WRITER_AGENT = Agent(
    name="Narrative Writer",
    instructions=(
        "Compile research results into a cohesive markdown report including context, "
        "analysis, and recommended follow-ups."
    ),
)

# New API: load base from YAML, patch with inline config
# - 'config_path' specifies the base YAML to load
# - Other keys in the dict override/merge with the base config
# - Agents can be Agent objects or dicts with name/instructions
pipeline = WebSearcherPipeline({
    "config_path": "pipelines/configs/web_searcher.yaml",
    "user_prompt": USER_PROMPT,
    "agents": {
        "evaluate_agent": EVALUATE_AGENT,
        "routing_agent": ROUTING_AGENT,
        "observe_agent": OBSERVE_AGENT,
        "writer_agent": WRITER_AGENT,
    }
})

pipeline.run_sync()
