"""Simple two-agent pipeline example.

This example demonstrates a minimal pipeline with:
- Routing agent that analyzes the query and creates tasks
- Single web searcher tool agent that executes the task

Usage:
    python examples/simple.py
"""

import json

from agentz.profiles.base import load_all_profiles
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
            "prompt": "Find the outstanding papers of ACL 2025, extract their title, author list, keywords, abstract, url in one sentence."
        }
    })
    result = pipeline.run_sync()
    print("\n=== Analysis Result ===")
    print(result)
    return result


def test_web_searcher_profile():
    """Smoke test for the web_searcher profile's runtime template and schema."""
    profiles = load_all_profiles()
    web_searcher = profiles["web_searcher"]

    # Validate the input schema round-trip
    task_input = web_searcher.input_schema(task="Find recent breakthroughs in reinforcement learning.")

    # Render runtime template with sample data
    runtime_payload = web_searcher.render(TASK=task_input.task)

    # Expand the output schema placeholder for human inspection
    instructions = web_searcher.instructions
    if "[[OUTPUT_SCHEMA]]" in instructions and web_searcher.output_schema:
        schema_json = json.dumps(web_searcher.output_schema.model_json_schema(), indent=2)
        instructions = instructions.replace("[[OUTPUT_SCHEMA]]", schema_json)

    print("\n=== Web Searcher Profile Test ===")
    print("Input schema dump:", task_input.model_dump())
    print("\nResolved instructions:\n", instructions)
    print("\nRuntime template payload:\n", runtime_payload)
    tool_names = [getattr(tool, "name", repr(tool)) for tool in (web_searcher.tools or [])]
    print("Registered tools:", tool_names)

    return {
        "task_input": task_input,
        "runtime_payload": runtime_payload,
        "instructions": instructions,
        "tools": tool_names,
    }


if __name__ == "__main__":
    # Run the simple pipeline
    test_web_searcher_profile()
    run_with_config()
