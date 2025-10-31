# Felix Framework

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)

## Overview

Felix is a Python multi-agent AI framework that leverages helical geometry for adaptive agent progression, enabling dynamic, scalable AI interactions. The framework models agent behaviors and communications along helical structures, allowing for continuous evolution and optimization of AI tasks through a hub-spoke communication model combined with helical progression.

Key features include helical progression from exploration (top_radius=3.0) to synthesis (bottom_radius=0.5), dynamic agent spawning based on confidence thresholds (0.80), role-specialized agents (Research/Analysis/Critic), smart CentralPost synthesis (adaptive temp 0.2-0.4, tokens 1500-3000), agent awareness with phase-based coordination, efficient hub-spoke messaging (O(N) complexity), autonomous knowledge brain for document learning with 3-tier embeddings and knowledge graph construction, web search integration (DuckDuckGo/SearxNG) with caching, workflow history tracking, system autonomy with SYSTEM_ACTION_NEEDED pattern detection, three-tier trust system (SAFE/REVIEW/BLOCKED) with configurable trust rules, approval workflow with 5 decision types (Approve Once, Always Exact, Always Command, Always Path, Deny), workflow pausing via threading.Event synchronization, command history tracking in felix_system_actions.db with deduplication, approval history browser in GUI, persistent memory with SQLite storage and abstractive compression (target_length=100, ~0.3 ratio), local LLM integration via LMStudioClient with incremental token streaming and token budgeting (base=2048, strict mode), markdown result formatting, dark mode GUI theme support, and linear pipelines with chunking (chunk=512).

Felix validates three key hypotheses: H1 (helical progression enhances agent adaptation by 20% workload distribution), H2 (hub-spoke communication optimizes resource allocation by 15% efficiency), and H3 (memory compression reduces latency by 25% attention focus). The framework supports up to 50 agents and is designed for applications like autonomous drone swarms, personalized AI assistants, and scalable chatbots.

For detailed structure, see [index.md](index.md).

## Features

- **Helical Progression**: Agents evolve along spiral paths from broad exploration to focused synthesis
- **Role-Specialized Agents**: Research, Analysis, and Critic agents with position-aware behavior
- **Agent Awareness System**: Phase-based coordination with team state queries and peer discovery
- **Smart CentralPost**: Hub performs intelligent final synthesis with adaptive parameters
- **Hub-Spoke Communication**: O(N) efficient messaging vs O(N²) mesh networks
- **Autonomous Knowledge Brain**: Document ingestion, agentic comprehension, and semantic retrieval with continuous learning
- **3-Tier Embeddings**: LM Studio → TF-IDF → FTS5 with automatic fallback for zero external dependencies
- **Knowledge Graph**: Relationship discovery via explicit mentions, embedding similarity, and co-occurrence analysis
- **Meta-Learning**: Tracks which knowledge proves useful for which workflows to improve retrieval relevance
- **Agentic RAG**: Agents actively comprehend documents using Research, Analysis, and Critic roles, not just chunking
- **Web Search Integration**: DuckDuckGo and SearxNG providers with result caching and domain filtering
- **Workflow History**: Persistent tracking of all workflow executions with searchable database
- **Token-Budgeted LLM Calls**: Local LM Studio integration with adaptive budgeting and incremental streaming
- **Context Compression**: Abstractive memory reduction for sustained performance
- **Markdown Export**: Professional formatting of synthesis results with agent metrics
- **Dark Mode GUI**: Theme support with persistent user preferences
- **Linear Pipelines**: Sequential processing with configurable chunking
- **Scalability**: Supports teams from 5 to 50 agents
- **Hypothesis Validation**: Comprehensive benchmarking for H1-H3 gains
- **System Autonomy**: Agents request command execution via SYSTEM_ACTION_NEEDED pattern detection
- **Three-Tier Trust System**: Commands classified as SAFE (auto-execute), REVIEW (require approval), or BLOCKED (never execute)
- **Approval Workflow**: Interactive approval dialogs with 5 decision types and workflow-scoped rules
- **Command History**: Persistent tracking of all system actions with success/failure status
- **Intelligent Command Generation**: Agents trained to check state before modifications, use proper shell quoting, and avoid redundant operations

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
   pip install openai httpx numpy scipy ddgs beautifulsoup4 lxml
   ```

4. (Optional) Set up LM Studio for local LLM inference.

### Run Example

See [QUICKSTART.md](QUICKSTART.md) for quick start examples using the CLI, GUI, or Python scripts.

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
   pip install openai httpx numpy scipy ddgs beautifulsoup4 lxml
   ```
   Additional packages (sqlite3, asyncio) are typically included with Python.
   - `ddgs`: DuckDuckGo search integration
   - `beautifulsoup4` & `lxml`: Web page parsing for search results
5. (Optional) Download and configure LM Studio server for local LLM integration.
6. Databases auto-initialize on first run (felix_memory.db, felix_knowledge.db).

For full operational details and troubleshooting, see [User Manual](USER_MANUAL.md).

## Usage

### High-Level Workflow
Felix operates through script-driven execution. Configure parameters via YAML files (see [CONFIGURATION.md](docs/CONFIGURATION.md)), then run workflows using the CLI, GUI, or Python scripts.

### Basic Custom Workflow
```python
from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory
from src.agents.specialized_agents import ResearchAgent, AnalysisAgent, CriticAgent

# Initialize components (CentralPost now handles synthesis)
helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
central_post = CentralPost(max_agents=10, enable_metrics=True, enable_memory=True, llm_client=llm_client)
agent_factory = AgentFactory(helix, llm_client)

# Create and register agents
research_agent = agent_factory.create_research_agent(domain="technical")
analysis_agent = agent_factory.create_analysis_agent()
critic_agent = agent_factory.create_critic_agent()

for agent in [research_agent, analysis_agent, critic_agent]:
    central_post.register_agent(agent)

# When consensus reached, CentralPost synthesizes final output
synthesis_result = central_post.synthesize_agent_outputs(task_description="Your task")

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

For detailed workflow patterns, see [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md).

### Testing & Benchmarking
Run comprehensive tests to validate H1-H3 hypotheses:
```bash
pytest tests/
```
Tests include unit tests, integration tests, and hypothesis validation demonstrating expected gains of 20% (H1), 15% (H2), and 25% (H3). See [tests/README.md](tests/README.md) for details.

## Command-Line Interface

Felix provides a comprehensive CLI for running workflows, checking status, and managing the system without requiring the GUI. This makes Felix ideal for CI/CD integration, remote deployments, and automated workflows.

### Available Commands

```bash
# Run a workflow
python -m src.cli run "Your task here"
python -m src.cli run "Explain quantum computing" --output result.md
python -m src.cli run "Design a REST API" --max-steps 10 --web-search

# Check system status (providers, databases, knowledge)
python -m src.cli status

# Test LLM connection and provider health
python -m src.cli test-connection

# Launch GUI from command line
python -m src.cli gui

# Initialize/reset databases
python -m src.cli init
```

### CLI Options

- `--output, -o FILE`: Save results to file (supports .txt, .md, .json formats)
- `--max-steps N`: Maximum workflow steps (default: 10)
- `--web-search`: Enable web search integration for the workflow
- `--config PATH`: LLM configuration file (default: config/llm.yaml)
- `--verbose, -v`: Enable verbose output with full stack traces

### Use Cases

- **CI/CD Integration**: Run Felix workflows in automated pipelines
- **Remote Servers**: Execute tasks on headless servers without X11/display
- **Scripting**: Integrate Felix into bash scripts and automation tools
- **Quick Testing**: Rapidly test configurations and providers
- **Batch Processing**: Process multiple tasks sequentially

For complete CLI documentation, see [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md).

## Architecture

Felix follows a modular architecture with clear component interactions:

- **Core**: Helical geometry algorithms for agent positioning and adaptation
- **Agents**: Specialized agent classes with LLM integration and dynamic spawning
- **Communication**: Hub-spoke messaging system (O(N) complexity) with specialized coordinators:
  - `CentralPost`: Main coordination hub delegating to specialized subsystems
  - `SynthesisEngine`: Intelligent synthesis of agent outputs
  - `WebSearchCoordinator`: Confidence-based web search triggering
  - `SystemCommandManager`: Command execution with trust and approval workflows
  - `StreamingCoordinator`: Real-time token streaming
  - `MemoryFacade`: Unified memory access layer
  - `PerformanceMonitor`: Metrics tracking and analysis
- **Memory**: Persistent storage with compression for knowledge retention and meta-learning
- **LLM**: Multi-provider architecture with automatic fallback (LM Studio, Anthropic, Gemini)
- **Knowledge Brain**: Autonomous document learning with 3-tier embeddings and knowledge graphs
- **Pipeline**: Linear processing with chunking support

Components interact through CentralPost as the coordination hub, with agents progressing along helical paths while sharing results and adapting based on confidence monitoring. The coordinator architecture enables efficient delegation of specialized concerns like synthesis, streaming, and command execution.

For detailed architecture diagrams and data flows, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Testing & Validation

Felix includes comprehensive testing to validate architectural hypotheses:

- **H1 (20% gain)**: Helical progression enhances workload distribution through adaptive agent behavior
- **H2 (15% gain)**: Hub-spoke communication optimizes resource allocation vs mesh networks
- **H3 (25% gain)**: Memory compression reduces latency while maintaining attention focus

Run validation tests with `pytest tests/` to verify expected performance improvements. See [tests/README.md](tests/README.md) for detailed test descriptions and results.

## GUI Interface

A Tkinter GUI is available in `src/gui/` for interactive control of Felix components with dark mode support. Features include:

- **Eight Tabs**: Dashboard, Workflows (with web search and approval polling), Memory, Agents, Approvals (pending and history), Terminal (command execution monitoring), Prompts, Learning, and Knowledge Brain
- **Knowledge Brain Tab**: Autonomous document learning with 4 sub-tabs:
  - Overview: Daemon control, status, and statistics
  - Documents: Browse ingested documents with status filtering
  - Concepts: Search and explore extracted knowledge by domain
  - Activity: Real-time processing log with auto-refresh
- **Dark/Light Themes**: Toggle between themes with persistent preferences
- **Terminal Tab**: Real-time command execution monitoring with:
  - Active Commands panel showing currently executing commands with live output streaming
  - Command History browser with filtering by status, search, and agent
  - Detailed execution views with full stdout/stderr, timing, and environment info
- **Workflow History Browser**: Search, filter, and view past workflow executions
- **Markdown Export**: Save synthesis results as formatted markdown files
- **Approval System**: Real-time approval dialogs during workflow execution with 5 decision types
- **Command Tracking**: View pending approvals and browse approval history with decision outcomes
- **Real-time Monitoring**: Track workflow execution and agent activity

See [`src/gui/README.md`](src/gui/README.md) for details. Run with:

```bash
python -m src.gui.main
```

## System Autonomy

Felix agents can request system command execution through the `SYSTEM_ACTION_NEEDED:` pattern. Commands are automatically classified using a three-tier trust system:

- **SAFE**: Read-only operations (ls, pwd, date, pip list) execute immediately
- **REVIEW**: State-modifying operations (mkdir, pip install, file writes) require user approval
- **BLOCKED**: Dangerous operations (rm -rf, credential access) are never executed

### Approval Workflow

When agents request REVIEW-level commands:
1. Workflow pauses via threading.Event synchronization
2. Approval dialog appears with command details, risk assessment, and context
3. User chooses from 5 decision types:
   - **Approve Once**: Execute this specific command once
   - **Always - Exact Match**: Auto-approve this exact command for current workflow
   - **Always - Command Type**: Auto-approve all commands of this type (e.g., all mkdir)
   - **Always - Path Pattern**: Auto-approve commands matching path pattern
   - **Deny**: Reject and continue workflow without executing

4. Decision recorded in approval history
5. Workflow resumes with command result

### Intelligent Command Generation

Agents are trained to:
- Check system state before modifications (test -d, test -f)
- Use proper shell quoting (double quotes for apostrophes)
- Avoid redundant operations (check before mkdir)
- Consider data preservation (append vs overwrite)

Configuration via `config/trust_rules.yaml`. Command history persisted in `felix_system_actions.db`.

## Knowledge Brain Configuration

The Knowledge Brain system enables autonomous document learning and semantic retrieval. Configure via the Settings tab in the GUI:

### Core Settings
- **Enable Knowledge Brain**: Toggle the autonomous learning system (default: disabled)
- **Watch Directories**: Directories to monitor for documents (one per line, e.g., `./knowledge_sources`, `./docs`)
- **Embedding Mode**: Embedding tier selection
  - `auto`: Automatically select best available (LM Studio → TF-IDF → FTS5)
  - `lm_studio`: Use LM Studio embeddings only
  - `tfidf`: Use TF-IDF embeddings only
  - `fts5`: Use SQLite FTS5 full-text search only

### Behavior Settings
- **Auto-Augment Workflows**: Automatically inject relevant knowledge into workflow context
- **Daemon Enabled**: Enable background processing daemon
- **Refinement Interval**: Hours between knowledge graph refinement cycles (default: 1 hour)
- **Processing Threads**: Number of concurrent document processing threads (default: 2)
- **Max Memory**: Maximum memory for processing in MB (default: 512)
- **Chunk Size**: Characters per document chunk (default: 1000)
- **Chunk Overlap**: Character overlap between chunks (default: 200)

### How It Works
1. **Document Ingestion**: Monitors watch directories for PDFs, markdown, text files, and code
2. **Agentic Comprehension**: Uses Felix agents (Research, Analysis, Critic) to understand content
3. **Knowledge Extraction**: Extracts concepts, entities, and relationships
4. **Graph Building**: Discovers connections via explicit mentions, similarity, and co-occurrence
5. **Semantic Search**: Makes knowledge retrievable via 3-tier embedding system
6. **Meta-Learning**: Tracks which knowledge helps which workflows over time
7. **Continuous Learning**: Runs indefinitely with batch processing, refinement, and file watching

The system requires zero external dependencies thanks to the tiered fallback architecture. Enable it, point it at your document directories, and let it build your knowledge brain autonomously.

## Documentation

- **[Quick Start Guide](QUICKSTART.md)**: Get up and running in 10 minutes
- **[CLI Guide](docs/CLI_GUIDE.md)**: Complete command-line interface reference
- **[Configuration Reference](docs/CONFIGURATION.md)**: Complete configuration and tuning guide
- **[Architecture Overview](docs/ARCHITECTURE.md)**: System architecture and design patterns
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Development and extension guide
- **[LLM Provider Guide](docs/LLM_PROVIDER_GUIDE.md)**: Multi-provider setup and custom providers
- **[Coordinator Architecture](docs/COORDINATOR_ARCHITECTURE.md)**: Detailed coordinator documentation
- **[User Manual](USER_MANUAL.md)**: Complete setup, configuration, and operational guide
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