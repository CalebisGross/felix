# Felix Framework Index

## Overview

The Felix Framework is a Python-based multi-agent AI system designed to leverage helical geometry for agent progression and adaptation. It enables dynamic, scalable AI interactions by modeling agent behaviors and communications along helical structures, allowing for continuous evolution and optimization of AI tasks. The framework integrates local LLM clients, persistent memory systems, and modular pipelines to support complex, adaptive workflows in multi-agent environments.

At its core, Felix employs a hub-spoke communication model combined with helical progression to facilitate agent spawning, role specialization, and task execution. This architecture promotes resilience and efficiency in handling diverse AI challenges, from prompt optimization to knowledge compression. By incorporating token budgeting and context-aware memory, the system ensures sustainable performance across varying computational constraints.

Felix is structured around key modules that handle agents, communication, core geometry, LLM integration, memory management, and pipeline processing. This modular design allows for flexible deployment and extension, making it suitable for applications requiring autonomous agent coordination and adaptive learning.

## Project Structure

```
felix/
├── .venv/                    # Virtual environment for Python dependencies
└── src/
    ├── __init__.py           # Package initialization
    ├── agents/               # Agent management and specialization
    │   ├── agent.py          # Base agent class and utilities
    │   ├── dynamic_spawning.py # Dynamic agent creation logic
    │   ├── llm_agent.py      # LLM-integrated agent implementations
    │   ├── prompt_optimization.py # Prompt refinement strategies
    │   └── specialized_agents.py # Role-specific agent definitions
    ├── communication/        # Inter-agent communication protocols
    │   ├── __init__.py       # Communication module init
    │   ├── central_post.py   # Central hub for message routing
    │   ├── mesh.py           # Mesh network communication
    │   └── spoke.py          # Spoke-based message handling
    ├── core/                 # Core helical geometry and utilities
    │   ├── __init__.py       # Core module init
    │   └── helix_geometry.py # Helical progression algorithms
    ├── llm/                  # LLM client and token management
    │   ├── __init__.py       # LLM module init
    │   ├── lm_studio_client.py # Local LM Studio integration
    │   ├── multi_server_client.py # Multi-server LLM support
    │   └── token_budget.py   # Token usage tracking and limits
    ├── memory/               # Persistent storage and compression
    │   ├── __init__.py       # Memory module init
    │   ├── context_compression.py # Context data compression
    │   ├── knowledge_store.py # Long-term knowledge storage
    │   └── task_memory.py    # Task-specific memory handling
    └── pipeline/             # Linear processing and chunking
        ├── __init__.py       # Pipeline module init
        ├── chunking.py       # Data chunking strategies
        └── linear_pipeline.py # Linear baseline pipelines
```

## Key Components

### Agents Module
- [`Agent`](src/agents/agent.py): Base class defining common agent behaviors and interfaces for multi-agent systems.
- [`DynamicSpawning`](src/agents/dynamic_spawning.py): Handles runtime creation and management of new agents based on system demands.
- [`LLMAgent`](src/agents/llm_agent.py): Integrates LLM capabilities into agent roles for intelligent decision-making.
- [`PromptOptimization`](src/agents/prompt_optimization.py): Refines prompts dynamically to improve LLM response quality and relevance.
- [`SpecializedAgents`](src/agents/specialized_agents.py): Defines role-specific agents for targeted tasks within the framework.

### Communication Module
- [`CentralPost`](src/communication/central_post.py): Acts as the central hub for routing messages between agents in a hub-spoke model.
- [`Mesh`](src/communication/mesh.py): Implements peer-to-peer communication networks for decentralized agent interactions.
- [`Spoke`](src/communication/spoke.py): Manages individual agent connections and message handling in the spoke layer.

### Core Module
- [`HelixGeometry`](src/core/helix_geometry.py): Provides algorithms for helical progression and geometric adaptation in agent behaviors.

### LLM Module
- [`LMStudioClient`](src/llm/lm_studio_client.py): Client for integrating with local LM Studio LLM services.
- [`MultiServerClient`](src/llm/multi_server_client.py): Supports connections to multiple LLM servers for distributed processing.
- [`TokenBudget`](src/llm/token_budget.py): Manages token usage limits and budgeting for efficient LLM interactions.

### Memory Module
- [`ContextCompression`](src/memory/context_compression.py): Compresses contextual data to optimize memory usage and retrieval.
- [`KnowledgeStore`](src/memory/knowledge_store.py): Persistent storage system for long-term knowledge retention across sessions.
- [`TaskMemory`](src/memory/task_memory.py): Handles short-term memory for task-specific data and state management.

### Pipeline Module
- [`Chunking`](src/pipeline/chunking.py): Strategies for breaking down large data into manageable chunks for processing.
- [`LinearPipeline`](src/pipeline/linear_pipeline.py): Implements linear baseline pipelines for sequential task execution.

## Hypotheses and Use Cases

### H1: Helical Progression Enhances Agent Adaptation
Helical geometry allows agents to evolve continuously along spiral paths, improving adaptability in dynamic environments like real-time strategy games or adaptive tutoring systems.

### H2: Multi-Agent Communication Optimizes Resource Allocation
The hub-spoke model with mesh extensions enables efficient resource sharing, applicable to distributed computing tasks such as load balancing in cloud services or collaborative problem-solving in research simulations.

### H3: Memory Compression Reduces Latency in AI Workflows
Context compression and knowledge stores minimize data retrieval times, ideal for high-throughput applications like automated content generation or real-time analytics in financial trading.

Example applications include autonomous drone swarms for environmental monitoring, personalized AI assistants in education, and scalable chatbots for customer service.

## Dependencies and Setup

Felix requires Python 3.8+ and the following key dependencies (install via pip in the provided .venv virtual environment):
- `openai` or similar for LLM integrations
- `numpy` and `scipy` for geometric computations
- `sqlite3` or `redis` for memory persistence
- `asyncio` for asynchronous agent operations

To set up: Activate the .venv environment (`source .venv/bin/activate` on Linux) and install dependencies from requirements.txt (if present) or manually via pip.

## Interactions and Workflow

Agents spawn dynamically via the core helix geometry, communicating through the central post or mesh networks. LLM agents optimize prompts and manage tokens, while memory modules compress and store knowledge. Pipelines process tasks linearly or via chunking, feeding results back into the helical progression loop for continuous adaptation. This interconnected workflow ensures scalable, adaptive AI behavior across modules.