"""Advanced usage of the DataScientistPipeline with typed configuration."""

from agents import Agent

from pipelines.data_scientist import DataScientistPipeline
from agentz.agents.base import DefaultAgentOutput
from agentz.agents.manager_agents import observe_agent, writer_agent

DATA_PATH = "data/banana_quality.csv"
USER_PROMPT = (
    "Build a model to classify banana quality as good or bad based on their numerical "
    "information about bananas of different quality (size, weight, sweetness, softness, "
    "harvest time, ripeness, and acidity). We have uploaded the entire dataset for you "
    "here in the banana_quality.csv file."
)
evaluate_agent=Agent(
    name="Focused Evaluator",
    instructions=(
        "Review research progress with an emphasis on data readiness, model risk, and "
        "missing validation steps. Respond using the KnowledgeGapOutput schema in JSON."
    ),
    output_type=DefaultAgentOutput,
)

routing_agent=Agent(
    name="Priority Router",
    instructions=(
        "Analyse outstanding gaps and assign tasks to tool agents. Ensure preprocessing "
        "and evaluation steps are scheduled before model retraining. Reply as JSON that "
        "conforms to AgentSelectionPlan."
    ),
    output_type=DefaultAgentOutput,
)
    
pipeline = DataScientistPipeline(
    agents= [evaluate_agent, routing_agent, observe_agent, writer_agent],
)

pipeline.run_sync()
