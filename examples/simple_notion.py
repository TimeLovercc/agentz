"""Simple two-agent pipeline example.

This example demonstrates a minimal pipeline with:
- Routing agent that analyzes the query and creates tasks
- Single data analysis tool agent that executes the task

Usage:
    python examples/simple.py
"""

from pipelines.simple_notion import SimpleNotionPipeline


# Example 1: Using config file
def run_with_config():
    """Run pipeline using YAML configuration."""
    pipeline = SimpleNotionPipeline("pipelines/configs/simple_notion.yaml")
    result = pipeline.run_sync()
    print("\n=== Result ===")
    print(result)
    return result


# Example 2: Using config override
def run_with_override():
    """Run pipeline with configuration override."""
    pipeline = SimpleNotionPipeline({
        "config_path": "pipelines/configs/simple_notion.yaml",
        "data": {
            "prompt": "Open www.google.com and search for 'openai agents sdk'. Return the title of the first result."
        }
    })
    result = pipeline.run_sync()
    print("\n=== Analysis Result ===")
    print(result)
    return result


if __name__ == "__main__":
    # Run the simple pipeline
    run_with_config()
