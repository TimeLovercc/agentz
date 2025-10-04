from agentz.configuration.base import BaseConfig
from pipelines.data_scientist import DataScientistPipeline


ds_config = BaseConfig(
    data_path="data/banana_quality.csv",
    user_prompt=(
        "Build a model to classify banana quality as good or bad based on their numerical "
        "information about bananas of different quality (size, weight, sweetness, softness, "
        "harvest time, ripeness, and acidity). We have uploaded the entire dataset for you "
        "here in the banana_quality.csv file."
    )
)

pipe = DataScientistPipeline(ds_config)

pipe.run_sync()
