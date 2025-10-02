<div align="center">

# AgentZ: Agent from Zero

**A Research-Oriented Multi-Agent System Platform**

</div>

AgentZ is a minimal, extensible codebase for multi-agent systems research. Build intelligent agent workflows with minimal code while achieving strong baseline performance. The platform enables autonomous reasoning, experience learning, and dynamic tool creation - providing both a comparative baseline and production-ready foundation for multi-agent research.

## Features

- **🎯 Minimal Implementation** - Build new systems with just a few lines of code
- **🔄 Stateful Workflows** - Persistent memory and object management throughout agent lifecycle
- **📚 Experience Learning** - Agents improve over time through memory-based reasoning
- **🛠️ Dynamic Tool Creation** - Agents can generate and use custom tools on-demand
- **🧠 Autonomous Reasoning** - Built-in cognitive capabilities for complex multi-step tasks
- **⚙️ Config-Driven** - Easily modify behavior through configuration files

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

### Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv
```

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for more options.

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/agentz.git
cd agentz

# Sync dependencies
uv sync
```

## Quick Start

```python
from pipelines.data_scientist import DataScientistPipeline

pipe = DataScientistPipeline(
    data_path="data/banana_quality.csv",
    user_prompt="Build a model to classify banana quality as good or bad based on their numerical information about bananas of different quality (size, weight, sweetness, softness, harvest time, ripeness, and acidity). We have uploaded the entire dataset for you here in the banana_quality.csv file.",
    model="gemini-2.5-flash",
    config_file="agentx/configs/gemini.json",
)

pipe.run_sync()
```

## Building Your Own System

### 1. Create a Custom Pipeline

Inherit from `BasePipeline` to create your own agent workflow:

```python
from pipelines.base import BasePipeline

class MyCustomPipeline(BasePipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add your custom initialization

    async def run(self):
        # Implement your workflow logic
        pass
```

### 2. Add Custom Agents

Implement your agents following the standard interface:

```python
from agents import Agent

def create_my_agent(config):
    return Agent(
        name="my_agent",
        instructions="Your agent instructions here",
        model=config.main_model
    )
```

### 3. Configure & Run

Create a config file and run your pipeline:

```python
pipe = MyCustomPipeline(
    data_path="your_data.csv",
    user_prompt="Your task description",
    provider="gemini",
    model="gemini-2.5-flash"
)

pipe.run_sync()
```

## Architecture

```
agentx/
├── pipelines/
│   ├── base.py              # Base pipeline with config management
│   └── data_scientist.py    # Reference implementation
├── src/
│   ├── agents/
│   │   ├── manager_agents/  # Orchestration agents
│   │   └── tool_agents/     # Task-specific agents
│   ├── llm/
│   │   └── llm_setup.py     # LLM configuration
│   ├── memory/
│   │   └── global_memory.py # Persistent memory system
│   └── tools/               # Agent tools and utilities
├── examples/
│   └── data_science.py      # Example workflows
└── data/                    # Sample datasets
```

## Benchmarks

AgentZ has been verified on several benchmarks for multi-agent research:

- **Data Science Tasks**: State-of-the-art performance on automated ML pipelines
- **Complex Reasoning**: Competitive results on multi-step reasoning benchmarks
- **Tool Usage**: High accuracy in dynamic tool selection and execution

*Detailed benchmark results and comparisons coming soon.*

## Roadmap

- [x] Persistence Process - Stateful agent workflows
- [x] Experience Learning - Memory-based reasoning
- [x] Tool Design - Dynamic tool creation
- [ ] Workflow RAG - Retrieval-augmented generation for complex workflows
- [ ] MCPs - Model Context Protocol support for enhanced agent communication

## Key Design Principles

1. **Minimal Core** - Keep the base system simple and extensible
2. **Intelligent Defaults** - Provide strong baseline implementations
3. **Research-First** - Design for experimentation and comparison
4. **Modular Architecture** - Easy to swap components and test variations
5. **Production-Ready** - Scale from research prototypes to deployed systems

## Use Cases

- **Multi-Agent Research** - Baseline for comparing agent architectures
- **Automated Data Science** - End-to-end ML pipeline automation
- **Complex Task Decomposition** - Break down and solve multi-step problems
- **Tool-Using Agents** - Research on dynamic tool creation and usage
- **Agent Memory Systems** - Study persistence and experience learning

## Citation

If you use AgentZ in your research, please cite:

```bibtex
@software{agentz2025,
  title={AgentZ: A Research-Oriented Multi-Agent System Platform},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/agentz}
}
```

## Contributing

We welcome contributions! AgentZ is designed to be a community resource for multi-agent research. Please open an issue or submit a pull request.

## License

[Your License Here]

## Acknowledgements

AgentZ is built with inspiration from the multi-agent systems research community. We thank the developers of various LLM frameworks and tools that make this work possible.

---

<div align="center">

**AgentZ**: Building intelligent agents from zero to hero 🚀

</div>
