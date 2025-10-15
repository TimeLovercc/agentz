<div align="center">

# AgentZ: Agent from Zero

**A Context-Central Multi-Agent System Platform**

</div>

AgentZ is a context-central multi-agent systems framework. AgentZ focuses on efficiently managing the context of each agent, binds all agents through centralized context engineering. Context-central design philosophy significantly improves the reusage of key components and eases the development and maintainence of scaled multi-agent system.

## Features

- **🎯 Context-Central Architecture** - All agents share and interact through centralized context state
- **🔄 Component Reusability** - Unified context design enables easy reuse of agents, tools, and flows
- **📚 Declarative Flows** - Define complex multi-agent workflows through structured, declarative specifications
- **🛠️ Stateful Execution** - Persistent conversation state tracks all agent interactions and tool results
- **🧠 Structured IO Contracts** - Type-safe communication between agents via Pydantic models
- **⚙️ Scalable Development** - Simplified maintenance and extension of multi-agent systems

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

pipe = DataScientistPipeline("pipelines/configs/data_science.yaml")

pipe.run_sync()
```

## Building Your Own System

### 1. Create a Custom Pipeline

Inherit from `BasePipeline` to create your own agent workflow:

```python
from pipelines.base import BasePipeline


class MyCustomPipeline(BasePipeline):
    DEFAULT_CONFIG_PATH = "pipelines/configs/my_pipeline.yaml"

    def __init__(self, config=None):
        super().__init__(config)
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

AgentZ is organised around a **central conversation state** and a set of declarative
flow specifications that describe how agents collaborate. The main
components you will interact with are:

- **`pipelines/`** – High level orchestration that wires agents together.
- **`agentz/agents/`** – Capability definitions for manager agents and tool agents.
- **`agentz/flow/`** – Flow primitives (`FlowRunner`, `FlowNode`, `IterationFlow`) that
  execute declarative pipelines.
- **`agentz/memory/`** – Structured state management (`ConversationState`,
  `ToolExecutionResult`, global memory helpers).
- **`examples/`** – Example scripts showing end-to-end usage.

```
agentz/
├── pipelines/
│   ├── base.py               # Base pipeline with config management & helpers
│   ├── flow_runner.py        # Declarative flow executor utilities
│   └── data_scientist.py     # Reference research pipeline
├── agentz/
│   ├── agents/
│   │   ├── manager_agents/   # Observe, evaluate, routing, writer agents
│   │   └── tool_agents/      # Specialised tool executors
│   ├── flow/                 # Flow node definitions and runtime objects
│   ├── memory/               # Conversation state & persistence utilities
│   ├── llm/                  # LLM adapters and setup helpers
│   └── tools/                # Built-in tools
└── examples/
    └── data_science.py       # Example workflows
```

### Declarative Pipeline Flow

The reference `DataScientistPipeline` models an entire research workflow using
three building blocks:

1. **Central ConversationState** – A shared store that captures every field any
   agent might read or write (iteration metadata, gaps, observations, tool
   results, timing, final report, etc.). Each loop creates a new
   `IterationRecord`, enabling partial updates and clean tracking of tool
   outcomes.
2. **Structured IO Contracts** – Each agent step declares the Pydantic model it
   expects and produces (for example `KnowledgeGapOutput` or
   `AgentSelectionPlan`). Input builders map slices of `ConversationState` into
   those models and output handlers merge the validated results back into the
   central state.
3. **Declarative FlowRunner** – The pipeline defines an `IterationFlow` of
   `FlowNode`s such as observe → evaluate → route → tools. Loop and termination
   logic are expressed with predicates (`loop_condition`, `condition`), so the
   executor can stop when evaluation marks `state.complete` or constraints are
   reached. Finalisation steps (like the writer agent) run after the iteration
   loop using the same structured IO.

Because the flow is declarative and all state is centralised, extending the
pipeline is as simple as adding a new node, output field, or tool capability—no
custom `run()` logic is required beyond sequencing the flow runner.

## Benchmarks

AgentZ's context-central design has been validated on multiple research benchmarks:

- **Data Science Tasks**: Efficient context sharing enables streamlined automated ML pipelines
- **Complex Reasoning**: Centralized state tracking improves multi-step reasoning coordination
- **Scalability**: Reduced overhead through component reuse in large-scale multi-agent systems

*Detailed benchmark results and comparisons coming soon.*

## Roadmap

- [x] Persistence Process - Stateful agent workflows
- [x] Experience Learning - Memory-based reasoning
- [x] Tool Design - Dynamic tool creation
- [ ] Workflow RAG - Retrieval-augmented generation for complex workflows
- [ ] MCPs - Model Context Protocol support for enhanced agent communication

## Key Design Principles

1. **Context-Central** - All agents communicate through shared, centralized context state
2. **Component Reusability** - Unified context engineering maximizes code reuse across agents
3. **Declarative Over Imperative** - Define workflows through structured specifications, not manual orchestration
4. **Structured State Management** - Type-safe IO contracts ensure reliable agent coordination
5. **Scalable by Design** - Simplified development and maintenance of large multi-agent systems

## Use Cases

- **Multi-Agent Research** - Study context-sharing patterns and agent coordination strategies
- **Automated Data Science** - Build reusable, stateful ML pipeline automation systems
- **Complex Workflow Orchestration** - Design scalable multi-step agent collaborations
- **Enterprise AI Systems** - Develop maintainable, large-scale agent deployments
- **Agent Architecture Comparison** - Benchmark different approaches with consistent context management

## Citation

If you use AgentZ in your research, please cite:

```bibtex
@software{agentz2025,
  title={AgentZ: A Context-Central Multi-Agent Systems Framework},
  author={Zhimeng Guo, Hangfan Zhang, Minghao Chen},
  year={2025},
  url={https://https://github.com/TimeLovercc/agentz}
}
```

## Contributing

We welcome contributions! AgentZ is designed to be a community resource for multi-agent research. Please open an issue or submit a pull request.


## Acknowledgements

AgentZ's context-central design is inspired by the multi-agent systems research community and best practices in distributed state management. We thank the developers of LLM frameworks and orchestration tools that informed this architecture.

---

<div align="center">

**AgentZ**: Building intelligent agents from zero to hero 🚀

</div>
