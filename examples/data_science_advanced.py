"""Advanced usage of the DataScientistPipeline with typed configuration."""

from agents import Agent

from agentz.agents.manager_agents.evaluate_agent import KnowledgeGapOutput
from agentz.agents.manager_agents.routing_agent import AgentSelectionPlan
from agentz.configuration import DataScienceConfig
from pipelines.data_scientist import DataScientistPipeline

DATA_PATH = "data/banana_quality.csv"
USER_PROMPT = (
    "Build a model to classify banana quality as good or bad based on their numerical "
    "information about bananas of different quality (size, weight, sweetness, softness, "
    "harvest time, ripeness, and acidity). We have uploaded the entire dataset for you "
    "here in the banana_quality.csv file."
)
CONFIG_PATH = "pipelines/configs/data_science.yaml"


def create_focus_evaluator(llm_config):
    """Specialised evaluation agent emphasising data quality checks."""

    return Agent(
        name="Focused Evaluator",
        instructions=(
            "Review research progress with an emphasis on data readiness, model risk, and "
            "missing validation steps. Respond using the KnowledgeGapOutput schema in JSON."
        ),
        output_type=KnowledgeGapOutput,
        model=llm_config.main_model,
    )


def create_priority_router(llm_config):
    """Routing agent that prioritises data cleansing before modelling."""

    return Agent(
        name="Priority Router",
        instructions=(
            "Analyse outstanding gaps and assign tasks to tool agents. Ensure preprocessing "
            "and evaluation steps are scheduled before model retraining. Reply as JSON that "
            "conforms to AgentSelectionPlan."
        ),
        output_type=AgentSelectionPlan,
        model=llm_config.main_model,
    )


base_config = DataScienceConfig.from_path(CONFIG_PATH)
custom_config = (
    base_config
    .model_copy(
        update={
            "pipeline": {**base_config.pipeline, "max_iterations": 4, "verbose": True}
        }
    )
    .with_manager_agents(
        evaluate_agent=create_focus_evaluator,
        routing_agent=create_priority_router,
    )
)

pipeline = DataScientistPipeline(
    config=custom_config,
    data_path=DATA_PATH,
    user_prompt=USER_PROMPT,
)

pipeline.run_sync()
