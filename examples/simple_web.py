"""Example script to run the SimpleWebPipeline for quick debugging.

Usage:
    python examples/simple_web.py
"""

from pipelines.simple_web import SimpleWebPipeline, WebSearchDebugQuery


def main() -> None:
    pipeline = SimpleWebPipeline("pipelines/configs/web_searcher.yaml")

    query = WebSearchDebugQuery(
        prompt="Find recent breakthroughs in reinforcement learning and cite sources."
    )

    pipeline.run_sync(query)


if __name__ == "__main__":
    main()
