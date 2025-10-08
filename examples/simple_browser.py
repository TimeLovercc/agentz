"""Simple two-agent pipeline example.

This example demonstrates a minimal pipeline with:
- Routing agent that analyzes the query and creates tasks
- Single data analysis tool agent that executes the task

Usage:
    python examples/simple.py
"""

from pipelines.simple_browser import SimpleBrowserPipeline


# Example 1: Using config file
def run_with_config():
    """Run pipeline using YAML configuration."""
    pipeline = SimpleBrowserPipeline("pipelines/configs/simple_browser.yaml")
    result = pipeline.run_sync()
    print("\n=== Result ===")
    print(result)
    return result


# Example 2: Using config override
def run_with_override():
    """Run pipeline with configuration override."""
    pipeline = SimpleBrowserPipeline({
        "config_path": "pipelines/configs/simple_browser.yaml",
        "data": {
            "prompt": "Open google.com and search for which LPL team loses to Vietnam in Bo3. Return the name of that team."
        }
    })
    result = pipeline.run_sync()
    print("\n=== Analysis Result ===")
    print(result)
    return result


if __name__ == "__main__":
    # Run the simple pipeline
    run_with_config()
