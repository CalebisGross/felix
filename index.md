# Felix Framework Index

## Overview

The Felix Framework is a Python-based multi-agent AI system designed to leverage helical geometry for agent progression and adaptation. It enables dynamic, scalable AI interactions by modeling agent behaviors and communications along helical structures, allowing for continuous evolution and optimization of AI tasks. The framework integrates local LLM clients, persistent memory systems, autonomous knowledge brain for document learning, and modular pipelines to support complex, adaptive workflows in multi-agent environments.

At its core, Felix employs a hub-spoke communication model combined with helical progression to facilitate agent spawning, role specialization, and task execution. The autonomous knowledge brain system extends this architecture by enabling document ingestion, agentic comprehension, knowledge graph construction, and semantic retrieval with meta-learning. This architecture promotes resilience and efficiency in handling diverse AI challenges, from prompt optimization to knowledge compression and continuous learning from documents. By incorporating token budgeting, context-aware memory, and 3-tier embeddings, the system ensures sustainable performance across varying computational constraints.

Felix is structured around key modules that handle agents, communication, core geometry, LLM integration, memory management, knowledge brain, and pipeline processing. This modular design allows for flexible deployment and extension, making it suitable for applications requiring autonomous agent coordination, adaptive learning, and continuous knowledge acquisition from documents.

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
    │   ├── central_post.py   # Central hub for message routing with agent awareness
    │   ├── mesh.py           # Mesh network communication
    │   └── spoke.py          # Spoke-based message handling
    ├── core/                 # Core helical geometry and utilities
    │   ├── __init__.py       # Core module init
    │   └── helix_geometry.py # Helical progression algorithms
    ├── knowledge/            # Autonomous knowledge brain system
    │   ├── __init__.py       # Knowledge module init
    │   ├── document_ingest.py # Multi-format document reading and semantic chunking
    │   ├── embeddings.py     # 3-tier embedding system (LM Studio/TF-IDF/FTS5)
    │   ├── comprehension.py  # Agentic document comprehension engine
    │   ├── graph_builder.py  # Knowledge graph construction and relationship discovery
    │   ├── knowledge_daemon.py # Autonomous background processing daemon
    │   ├── retrieval.py      # Semantic search with meta-learning boost
    │   └── workflow_integration.py # Bridge between knowledge brain and workflows
    ├── gui/                  # Graphical user interface components
    │   ├── __init__.py       # GUI module init
    │   ├── main.py           # Main GUI application
    │   ├── themes.py         # Dark/light theme management
    │   ├── workflow_history_frame.py # Workflow history browser
    │   ├── knowledge_brain.py # Knowledge Brain tab with 4 sub-tabs (Overview, Documents, Concepts, Activity)
    │   └── [other GUI components] # Dashboard, workflows, memory, agents, approvals, terminal, prompts, learning tabs
    ├── llm/                  # LLM client and token management
    │   ├── __init__.py       # LLM module init
    │   ├── lm_studio_client.py # Local LM Studio integration with streaming
    │   ├── multi_server_client.py # Multi-server LLM support
    │   ├── token_budget.py   # Token usage tracking and limits
    │   └── web_search_client.py # DuckDuckGo/SearxNG web search
    ├── memory/               # Persistent storage and compression
    │   ├── __init__.py       # Memory module init
    │   ├── context_compression.py # Context data compression
    │   ├── knowledge_store.py # Long-term knowledge storage
    │   ├── task_memory.py    # Task-specific memory handling
    │   └── workflow_history.py # Workflow execution tracking
    ├── pipeline/             # Linear processing and chunking
    │   ├── __init__.py       # Pipeline module init
    │   ├── chunking.py       # Data chunking strategies
    │   └── linear_pipeline.py # Linear baseline pipelines
    ├── utils/                # Utility functions and formatting
    │   ├── __init__.py       # Utils module init
    │   └── markdown_formatter.py # Markdown result formatting
    └── workflows/            # Workflow orchestration and integration
        ├── __init__.py       # Workflows module init
        ├── context_builder.py # Collaborative context management
        ├── felix_workflow.py # Main workflow with web search integration
        └── truth_assessment.py # Workflow output validation
```

## Key Components

### Agents Module
- [`Agent`](src/agents/agent.py): Base class defining common agent behaviors and interfaces for multi-agent systems.
- [`DynamicSpawning`](src/agents/dynamic_spawning.py): Handles runtime creation and management of new agents based on system demands.
- [`LLMAgent`](src/agents/llm_agent.py): Integrates LLM capabilities into agent roles for intelligent decision-making.
- [`PromptOptimization`](src/agents/prompt_optimization.py): Refines prompts dynamically to improve LLM response quality and relevance.
- [`SpecializedAgents`](src/agents/specialized_agents.py): Defines role-specific agents for targeted tasks within the framework.

### Communication Module
- [`CentralPost`](src/communication/central_post.py): Acts as the central hub for routing messages between agents in a hub-spoke model with agent awareness and phase-based tracking.
- [`AgentRegistry`](src/communication/central_post.py): Phase-based agent tracking system enabling team state queries, peer discovery, and collaboration coordination.
- [`Mesh`](src/communication/mesh.py): Implements peer-to-peer communication networks for decentralized agent interactions.
- [`Spoke`](src/communication/spoke.py): Manages individual agent connections and message handling in the spoke layer.

### Core Module
- [`HelixGeometry`](src/core/helix_geometry.py): Provides algorithms for helical progression and geometric adaptation in agent behaviors.

### LLM Module
- [`LMStudioClient`](src/llm/lm_studio_client.py): Client for integrating with local LM Studio LLM services with incremental token streaming support.
- [`MultiServerClient`](src/llm/multi_server_client.py): Supports connections to multiple LLM servers for distributed processing.
- [`TokenBudget`](src/llm/token_budget.py): Manages token usage limits and budgeting for efficient LLM interactions.
- [`WebSearchClient`](src/llm/web_search_client.py): Integrates DuckDuckGo and SearxNG search with result caching, domain filtering, and page content fetching.

### Memory Module
- [`ContextCompression`](src/memory/context_compression.py): Compresses contextual data to optimize memory usage and retrieval.
- [`KnowledgeStore`](src/memory/knowledge_store.py): Persistent storage system for long-term knowledge retention across sessions.
- [`TaskMemory`](src/memory/task_memory.py): Handles short-term memory for task-specific data and state management.
- [`WorkflowHistory`](src/memory/workflow_history.py): Tracks and persists complete workflow executions with searchable metadata, confidence scores, and performance metrics.

### Pipeline Module
- [`Chunking`](src/pipeline/chunking.py): Strategies for breaking down large data into manageable chunks for processing.
- [`LinearPipeline`](src/pipeline/linear_pipeline.py): Implements linear baseline pipelines for sequential task execution.

### Knowledge Module
- [`DocumentReader`](src/knowledge/document_ingest.py): Multi-format document reading (PDF, TXT, MD, code) with semantic chunking that respects paragraph and section boundaries.
- [`EmbeddingProvider`](src/knowledge/embeddings.py): 3-tier embedding system with automatic fallback - LM Studio (768-dim) → TF-IDF (pure Python) → FTS5 (SQLite BM25).
- [`KnowledgeComprehensionEngine`](src/knowledge/comprehension.py): Agentic document understanding using Research, Analysis, and Critic agents to extract concepts, entities, and relationships.
- [`KnowledgeGraphBuilder`](src/knowledge/graph_builder.py): Discovers bidirectional relationships via explicit mentions, embedding similarity (0.75 threshold), and co-occurrence (5-chunk window).
- [`KnowledgeDaemon`](src/knowledge/knowledge_daemon.py): Autonomous background processor running 3 concurrent modes - batch processing, continuous refinement (hourly), and file watching with watchdog.
- [`KnowledgeRetriever`](src/knowledge/retrieval.py): Semantic search with meta-learning boost that tracks which knowledge proves useful for which workflows.
- [`WorkflowIntegration`](src/knowledge/workflow_integration.py): Bridge connecting knowledge brain to Felix workflows for automatic context augmentation.

### GUI Module
- [`themes`](src/gui/themes.py): Dark and light theme management with ColorScheme definitions and ThemeManager for dynamic switching.
- [`workflow_history_frame`](src/gui/workflow_history_frame.py): Interactive browser for viewing, searching, and filtering past workflow executions.
- [`knowledge_brain`](src/gui/knowledge_brain.py): Knowledge Brain tab with 4 sub-tabs - Overview (daemon control and statistics), Documents (browse ingested documents), Concepts (search knowledge by domain), and Activity (real-time processing log).
- Other components: Dashboard, Workflows tab with web search integration, Memory browser, Agents manager, Approvals browser, Terminal monitor, Prompts editor, and Learning systems.

### Utils Module
- [`MarkdownFormatter`](src/utils/markdown_formatter.py): Professional markdown formatting for synthesis results with agent metrics, performance summaries, and detailed reports.

### Workflows Module
- [`FelixWorkflow`](src/workflows/felix_workflow.py): Main workflow orchestration with web search integration and task complexity classification.
- [`ContextBuilder`](src/workflows/context_builder.py): Collaborative context management for multi-agent coordination.
- [`TruthAssessment`](src/workflows/truth_assessment.py): Framework for validating and assessing workflow outputs.

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
- `httpx` for async HTTP requests
- `numpy` and `scipy` for geometric computations
- `ddgs` for DuckDuckGo search functionality
- `beautifulsoup4` and `lxml` for web page parsing
- `sqlite3` for memory persistence (typically included with Python)
- `asyncio` for asynchronous agent operations (typically included with Python)
- `tkinter` for GUI (typically included with Python)

Optional dependencies for Knowledge Brain system:
- `PyPDF2` for PDF document reading (system works without it, falls back to text extraction)
- `watchdog` for file system monitoring (daemon can still run batch/refinement modes without it)

To set up: Activate the .venv environment (`source .venv/bin/activate` on Linux) and install dependencies:
```bash
pip install openai httpx numpy scipy ddgs beautifulsoup4 lxml
# Optional for Knowledge Brain:
pip install PyPDF2 watchdog
```

## Interactions and Workflow

Agents spawn dynamically via the core helix geometry, communicating through the central post with phase-based awareness and peer discovery. LLM agents optimize prompts, manage tokens, and leverage web search for real-time information gathering. Memory modules compress and store knowledge, while workflow history tracks complete execution records with searchable metadata.

The workflow system integrates web search capabilities, allowing agents to augment their knowledge with current information from DuckDuckGo or SearxNG. Results are cached per-task and filtered by domain to ensure quality. Pipelines process tasks linearly or via chunking, with synthesis results formatted as professional markdown reports including agent metrics and performance summaries.

The autonomous knowledge brain extends Felix's capabilities by continuously learning from documents. The system monitors watch directories for new documents, uses agentic comprehension (Research, Analysis, Critic agents) to understand content, builds a knowledge graph with bidirectional relationships, and makes knowledge retrievable via 3-tier embeddings (LM Studio → TF-IDF → FTS5). Meta-learning tracks which knowledge proves useful for which workflows, boosting retrieval relevance over time. The daemon runs indefinitely with concurrent batch processing, hourly refinement, and optional file watching, requiring zero external dependencies through intelligent fallback architecture.

The GUI provides interactive control with dark/light themes, allowing users to monitor workflows, browse history, manage memory, spawn agents, control the knowledge brain daemon, and view learned concepts. All workflow executions are persisted with comprehensive metadata for later analysis and retrieval. This interconnected workflow ensures scalable, adaptive AI behavior with continuous learning and optimization across all modules.