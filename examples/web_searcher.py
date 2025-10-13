from pipelines.web_searcher import WebSearcherPipeline, WebSearchQuery

# Load the default configuration file and start the pipeline using the one-parameter API.
pipe = WebSearcherPipeline("pipelines/configs/web_searcher.yaml")

query = WebSearchQuery(
    prompt="Find the outstanding papers of ACL 2025, extract their title, author list, keywords, abstract, url in one sentence."
)

pipe.run_sync(query)
