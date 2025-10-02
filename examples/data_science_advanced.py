"""Advanced usage of the DataScientistPipeline with typed configuration."""

from agents import Agent

from agentz.agents.base import DefaultAgentOutput
from pipelines.data_scientist import DataScientistPipeline

DATA_PATH = "data/banana_quality.csv"
USER_PROMPT = (
    "Build a model to classify banana quality as good or bad based on their numerical "
    "information about bananas of different quality (size, weight, sweetness, softness, "
    "harvest time, ripeness, and acidity). We have uploaded the entire dataset for you "
    "here in the banana_quality.csv file."
)

EVALUATE_AGENT = Agent(
    name="Focused Evaluator",
    instructions=(
        "Review research progress with an emphasis on data readiness, model risk, and "
        "missing validation steps. Respond using the KnowledgeGapOutput schema in JSON."
    ),
    output_type=DefaultAgentOutput,
)

ROUTING_AGENT = Agent(
    name="Priority Router",
    instructions=(
        "Analyse outstanding gaps and assign tasks to tool agents. Ensure preprocessing "
        "and evaluation steps are scheduled before model retraining. Reply as JSON that "
        "conforms to AgentSelectionPlan."
    ),
    output_type=DefaultAgentOutput,
)

OBSERVE_AGENT = Agent(
    name="Insight Observer",
    instructions=(
        "Summarise iteration progress, highlight risks, and note pending follow-up items "
        "before the next iteration begins."
    ),
    output_type=DefaultAgentOutput,
)

WRITER_AGENT = Agent(
    name="Narrative Writer",
    instructions=(
        "Compile research results into a cohesive markdown report including context, "
        "analysis, and recommended follow-ups."
    ),
    output_type=DefaultAgentOutput,
)

pipeline = DataScientistPipeline(
    data_path=DATA_PATH,
    user_prompt=USER_PROMPT,
    agents=[EVALUATE_AGENT, ROUTING_AGENT, OBSERVE_AGENT, WRITER_AGENT],
)

pipeline.run_sync()
