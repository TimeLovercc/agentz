from pipelines.data_scientist import DataScientistPipeline
from agentz.configuration.data_science import DataScienceConfig

DATA_PATH = "data/banana_quality.csv"
USER_PROMPT = (
    "Build a model to classify banana quality as good or bad based on their numerical "
    "information about bananas of different quality (size, weight, sweetness, softness, "
    "harvest time, ripeness, and acidity). We have uploaded the entire dataset for you "
    "here in the banana_quality.csv file."
)

config = DataScienceConfig.from_path("pipelines/configs/data_science.yaml")

pipe = DataScientistPipeline(
    config=config,
    data_path=DATA_PATH,
    user_prompt=USER_PROMPT,
)

pipe.run_sync()
