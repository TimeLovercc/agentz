from pipelines.data_scientist import DataScientistPipeline, DataScienceQuery

# Load the default configuration file and start the pipeline using the one-parameter API.
pipe = DataScientistPipeline("pipelines/configs/data_science.yaml")

query = DataScienceQuery(
    prompt="Analyze the dataset and build a predictive model",
    data_path="/Users/zhimeng/workspaces/agentz/data/banana_quality.csv"
)

pipe.run_sync(query)
