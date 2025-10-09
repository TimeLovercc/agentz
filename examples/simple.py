"""Simple two-agent pipeline example.

This example demonstrates a minimal pipeline with:
- Routing agent that analyzes the query and creates tasks
- Single data analysis tool agent that executes the task

Usage:
    python examples/simple.py
"""

from pipelines.simple import SimplePipeline


# Example 1: Using config file
def run_with_config():
    """Run pipeline using YAML configuration."""
    pipeline = SimplePipeline("pipelines/configs/simple.yaml")
    result = pipeline.run_sync()
    print("\n=== Analysis Result ===")
    print(result)
    return result


# Example 2: Using config override
def run_with_override():
    """Run pipeline with configuration override."""
    pipeline = SimplePipeline({
        "config_path": "pipelines/configs/simple.yaml",
        "data": {
            "path": "data/banana_quality.csv",
            "prompt": "Analyze correlations in the banana dataset"
        }
    })
    result = pipeline.run_sync()
    print("\n=== Analysis Result ===")
    print(result)
    return result


if __name__ == "__main__":
    # Run the simple pipeline
    run_with_config()
