from pipelines.data_scientist_memory import DataScientistMemoryPipeline

# Load the default configuration file and start the pipeline using the one-parameter API.
pipe = DataScientistMemoryPipeline("pipelines/configs/data_science_memory.yaml")

pipe.run_sync()
