# AgentZ: Agent from Zero

**A Research-Oriented Multi-Agent System Platform**

AgentZ is a minimal, extensible codebase designed for multi-agent systems research. It provides a comparative baseline and powerful foundation for building intelligent agent workflows with minimal code.

## Why AgentZ?

### 🎯 Simple
- **Minimal Pipeline Implementation**: Build new systems with just a few lines of code
- **Easy Agent Integration**: Add your own agents without modifying core architecture
- **Config-Driven**: Change behavior through simple configuration files

### 🧠 Intelligent
- **Design Agent**: Automatically handles prompt engineering and architecture design
- **Self-Optimizing**: Intelligent agents learn and improve from interactions
- **Autonomous Reasoning**: Built-in cognitive capabilities for complex tasks

### ⚡ Easy-to-Use
- **Powerful Base Implementation**: Strong default performance out-of-the-box
- **Minimal Modifications Needed**: Achieve good results without extensive tuning
- **Clean API**: Intuitive interfaces for common research tasks

## Core Features

### 🔄 Persistence Process
Keep and modify objects in memory throughout the agent lifecycle, enabling stateful workflows and continuous learning.

### 📚 Experience Learning
Agents learn from past experiences, improving performance over time through memory-based reasoning and pattern recognition.

### 🛠️ Tool Design
Easily integrate new tools through code generation capabilities. Agents can dynamically create and use custom tools as needed.

### 🚀 Coming Soon
- **Workflow RAG**: Retrieval-augmented generation for complex workflows
- **MCPs**: Model Context Protocol support for enhanced agent communication

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

## Installation

```bash
pip install -r requirements.txt
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

## Key Design Principles

1. **Minimal Core**: Keep the base system simple and extensible
2. **Intelligent Defaults**: Provide strong baseline implementations
3. **Research-First**: Design for experimentation and comparison
4. **Modular Architecture**: Easy to swap components and test variations
5. **Production-Ready**: Scale from research prototypes to deployed systems

## Use Cases

- **Multi-Agent Research**: Baseline for comparing agent architectures
- **Automated Data Science**: End-to-end ML pipeline automation
- **Complex Task Decomposition**: Break down and solve multi-step problems
- **Tool-Using Agents**: Research on dynamic tool creation and usage
- **Agent Memory Systems**: Study persistence and experience learning

## Contributing

We welcome contributions! AgentZ is designed to be a community resource for multi-agent research.

## License

[Your License Here]

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

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact [your contact info].

---

**AgentZ**: Building intelligent agents from zero to hero 🚀
