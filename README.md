# Felix Framework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## Overview

Felix is a Python multi-agent AI framework that leverages helical geometry for adaptive agent progression, enabling dynamic, scalable AI interactions. The framework models agent behaviors and communications along helical structures, allowing for continuous evolution and optimization of AI tasks through a hub-spoke communication model combined with helical progression.

Key features include helical progression from exploration (top_radius=3.0) to synthesis (bottom_radius=0.5), dynamic agent spawning based on confidence thresholds (0.75), role-specialized agents (Research/Analysis/Synthesis/Critic), efficient hub-spoke messaging (O(N) complexity), persistent memory with SQLite storage and abstractive compression (target_length=100, ~0.3 ratio), local LLM integration via LMStudioClient with token budgeting (base=2048, strict mode), and linear pipelines with chunking (chunk=512).

Felix validates three key hypotheses: H1 (helical progression enhances agent adaptation by 20% workload distribution), H2 (hub-spoke communication optimizes resource allocation by 15% efficiency), and H3 (memory compression reduces latency by 25% attention focus). The framework supports up to 133 agents and is designed for applications like autonomous drone swarms, personalized AI assistants, and scalable chatbots.

For detailed structure, see [index.md](index.md).

## Features

- **Helical Progression**: Agents evolve along spiral paths from broad exploration to focused synthesis
- **Role-Specialized Agents**: Research, Analysis, Synthesis, and Critic agents with position-aware behavior
- **Hub-Spoke Communication**: O(N) efficient messaging vs O(N²) mesh networks
- **Token-Budgeted LLM Calls**: Local LM Studio integration with adaptive budgeting
- **Context Compression**: Abstractive memory reduction for sustained performance
- **Linear Pipelines**: Sequential processing with configurable chunking
- **Scalability**: Supports teams from 5 to 133 agents
- **Hypothesis Validation**: Comprehensive benchmarking for H1-H3 gains

## Quick Start

### Prerequisites
- Python 3.8+
- Git

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd felix
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # .venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install openai httpx numpy scipy
   ```

4. (Optional) Set up LM Studio for local LLM inference.

### Run Example
```bash
python exp/example_workflow.py "Evaluate Python for modern software development"
```

This runs a complete workflow with mock LLM responses, demonstrating agent spawning, task processing, memory storage, and results generation.

## Installation

### Detailed Setup Steps
1. Ensure Python 3.8+ is installed on your system.
2. Clone the repository and navigate to the project directory.
3. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install required dependencies:
   ```bash
   pip install openai httpx numpy scipy
   ```
   Additional packages (sqlite3, asyncio) are typically included with Python.
5. (Optional) Download and configure LM Studio server for local LLM integration.
6. Databases auto-initialize on first run (felix_memory.db, felix_knowledge.db).

For full operational details and troubleshooting, see [User Manual](USER_MANUAL.md).

## Usage

### High-Level Workflow
Felix operates through script-driven execution. Configure parameters via YAML files based on [optimal_parameters.md](exp/optimal_parameters.md), then run workflows using example scripts.

### Basic Custom Workflow
```python
from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory
from src.agents.specialized_agents import ResearchAgent, AnalysisAgent, SynthesisAgent

# Initialize components
helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
central_post = CentralPost(max_agents=10, enable_metrics=True, enable_memory=True)
agent_factory = AgentFactory(helix, llm_client)

# Create and register agents
research_agent = agent_factory.create_research_agent(domain="technical")
analysis_agent = agent_factory.create_analysis_agent()
synthesis_agent = agent_factory.create_synthesis_agent()

for agent in [research_agent, analysis_agent, synthesis_agent]:
    central_post.register_agent(agent)

# Process tasks (see workflow steps below)
```

### Workflow Steps
1. **Initialization**: Set up HelixGeometry, CentralPost, and AgentFactory
2. **Agent Spawning**: Create role-specialized agents with spawn time ranges
3. **Registration**: Register agents with CentralPost for message routing
4. **Task Processing**: Agents process tasks with position-aware LLM prompting
5. **Communication**: Share results via hub-spoke messaging
6. **Memory Storage**: Store insights in KnowledgeStore with compression
7. **Dynamic Adaptation**: Monitor confidence and spawn additional agents as needed
8. **Result Generation**: Synthesize final outputs through helical progression

For detailed workflow patterns, see [Workflow Steps](exp/workflow_steps.md).

### Benchmarking
Run comprehensive benchmarks to validate H1-H3 hypotheses:
```bash
cd exp
python benchmark_felix.py
```
Outputs metrics to benchmark_results.csv, including expected gains of 20% (H1), 15% (H2), and 25% (H3).

## Architecture

Felix follows a modular architecture with clear component interactions:

- **Core**: Helical geometry algorithms for agent positioning and adaptation
- **Agents**: Specialized agent classes with LLM integration and dynamic spawning
- **Communication**: Hub-spoke messaging system for efficient inter-agent coordination
- **Memory**: Persistent storage with compression for knowledge retention
- **LLM**: Local model integration with token budgeting
- **Pipeline**: Linear processing with chunking support

Components interact through CentralPost as the coordination hub, with agents progressing along helical paths while sharing results and adapting based on confidence monitoring.

For detailed architecture diagrams and data flows, see [Component Interactions](exp/component_interactions.md).

## Benchmarks

Felix includes comprehensive benchmarking via `exp/benchmark_felix.py`, which tests components and validates hypotheses:

- **H1 (20% gain)**: Helical progression enhances workload distribution through adaptive agent behavior
- **H2 (15% gain)**: Hub-spoke communication optimizes resource allocation vs mesh networks
- **H3 (25% gain)**: Memory compression reduces latency while maintaining attention focus

Run benchmarks from the exp/ directory to generate CSV metrics. Results demonstrate functional verification with expected performance improvements across all hypotheses.

## GUI Interface

A Tkinter GUI is available in `src/gui/` for interactive control of Felix components. See [`src/gui/README.md`](src/gui/README.md) for details. Run with:

```bash
python -m src.gui.main
```

## Documentation

- **[User Manual](USER_MANUAL.md)**: Complete setup, configuration, and operational guide
- **[Optimal Parameters](exp/optimal_parameters.md)**: Parameter tuning and trade-off analysis
- **[Workflow Steps](exp/workflow_steps.md)**: Detailed workflow breakdown and patterns
- **[Component Interactions](exp/component_interactions.md)**: Architecture and data flow diagrams
- **[Index](index.md)**: Framework overview and project structure

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make changes with comprehensive tests
4. Run benchmarks to ensure H1-H3 validation
5. Submit a pull request

For bugs or feature requests, please open an issue with detailed reproduction steps.

## License

MIT License - see LICENSE file for details.