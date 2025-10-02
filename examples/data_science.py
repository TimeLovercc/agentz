from pipelines.data_scientist import DataScientistPipeline

# Load the default configuration file and start the pipeline using the one-parameter API.
pipe = DataScientistPipeline("pipelines/configs/data_science.yaml")

pipe.run_sync()
