from pipelines.data_scientist import DataScientistPipeline
from agentz.configuration.data_science import DataScienceConfig

config = DataScienceConfig.from_path("pipelines/configs/data_science.yaml")

pipe = DataScientistPipeline(
    config=config,
)

pipe.run_sync()
