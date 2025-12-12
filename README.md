# Felix Framework

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production--Ready-blue)

## Designed for Air-Gapped Environments

Felix is built offline-first, not offline-as-afterthought:

- **Zero external dependencies** - No vector databases or cloud services required
- **Graceful degradation** - 3-tier fallback (LM Studio -> TF-IDF -> SQLite FTS5)
- **Safe autonomy** - Trust system (SAFE/REVIEW/BLOCKED) for agent command execution

Other frameworks can work offline with configuration. Felix works offline out of the box.

**Perfect for:** Defense contractors, government agencies, healthcare (HIPAA), finance (SOX), any organization requiring complete data isolation.

---

## Overview

Felix is a Python multi-agent AI framework that leverages helical geometry for adaptive agent progression, enabling dynamic, scalable AI interactions. The framework models agent behaviors and communications along helical structures, allowing for continuous evolution and optimization of AI tasks through a hub-spoke communication model combined with helical progression.

**Key architectural features:**

- **FelixAgent**: Unified identity layer that routes requests based on complexity - direct inference for simple queries, full multi-agent workflow for complex tasks
- **Helical Progression**: Agents evolve from exploration (wide radius) to synthesis (narrow) along a 3D spiral path
- **Hub-Spoke Communication**: O(N) efficient messaging through CentralPost vs O(N^2) mesh networks
- **3-Tier Embeddings**: LM Studio -> TF-IDF -> FTS5 with automatic fallback for zero external dependencies
- **Modern GUI**: CustomTkinter-based interface with Claude-style chat, 11 tabs, and responsive design

Felix supports Python 3.12, 3.13, and 3.14 including free-threaded builds.

For detailed structure, see [index.md](index.md).

## Features

### Core Architecture

- **Helical Progression**: Agents evolve along spiral paths from broad exploration to focused synthesis
- **FelixAgent Identity Layer**: Unified persona that routes between direct inference and full workflows
- **Role-Specialized Agents**: Research, Analysis, Critic, and System agents with position-aware behavior
- **Agent Awareness System**: Phase-based coordination with team state queries and peer discovery
- **Smart CentralPost**: Hub performs intelligent final synthesis with adaptive parameters
- **Hub-Spoke Communication**: O(N) efficient messaging vs O(N^2) mesh networks

### Knowledge System

- **Autonomous Knowledge Brain**: Document ingestion, agentic comprehension, and semantic retrieval
- **3-Tier Embeddings**: LM Studio -> TF-IDF -> FTS5 with automatic fallback
- **Knowledge Graph**: Relationship discovery via explicit mentions, embedding similarity, and co-occurrence
- **Strategic Comprehension**: Efficient document processing with outline-based chunk prioritization
- **Gap Tracking**: Identifies missing knowledge and directs learning
- **Coverage Analysis**: Measures knowledge completeness across domains
- **Meta-Learning**: Tracks which knowledge proves useful for which workflows

### LLM Integration

- **Multi-Provider Architecture**: LM Studio, Anthropic, Gemini with automatic fallback
- **Token-Budgeted Calls**: Adaptive budgeting and incremental streaming
- **Context Compression**: Abstractive memory reduction for sustained performance
- **Prompt Pipeline**: Configurable prompt chain with failure recovery

### Autonomy Features

- **Three-Tier Trust System**: SAFE (auto-execute), REVIEW (approval), BLOCKED (never execute)
- **Approval Workflow**: Interactive dialogs with 5 decision types
- **Command History**: Persistent tracking with success/failure status
- **Intelligent Command Generation**: State checking, proper quoting, redundancy avoidance

### User Interfaces

- **Modern CustomTkinter GUI**: 11 tabs including Chat with Claude-style interface
- **Original Tkinter GUI**: Classic interface for compatibility
- **CLI**: Full-featured command-line interface for automation
- **REST API**: FastAPI-based endpoints for integration
- **Chat CLI**: Conversational interface with session management

## Quick Start

### Prerequisites

- Python 3.12+
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
   pip install -e .
   # OR manually:
   pip install openai httpx pyyaml numpy scipy ddgs beautifulsoup4 lxml
   ```

4. (Optional) Set up LM Studio for local LLM inference.

### Run Example

See [QUICKSTART.md](QUICKSTART.md) for quick start examples using the CLI, GUI, or Python scripts.

## Installation

### Option 1: Docker (Recommended for Production)

```bash
# Clone repository
git clone <repository-url>
cd felix

# Start with Docker Compose
docker-compose up -d

# Access API at http://localhost:8000
# View API docs at http://localhost:8000/docs
```

**Air-gapped networks:**
- Build image offline: `docker build -t felix-framework .`
- Transfer image to isolated network
- Run without internet access: `docker run -p 8000:8000 felix-framework`

### Option 2: pip Install

```bash
pip install felix-framework

# Run CLI
felix run "Your task here"

# Start API server
felix api --port 8000
```

### Option 3: From Source (Development)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd felix
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -e .  # Installs from pyproject.toml
   ```

4. (Optional) Install additional features:
   ```bash
   pip install -e .[api]       # REST API support (FastAPI, uvicorn)
   pip install -e .[knowledge] # PDF reading, file watching
   pip install -e .[dev]       # Testing & linting (pytest, ruff, mypy)
   pip install -e .[all]       # Everything
   ```

5. (Optional) Set up local LLM (LM Studio recommended)

6. Databases auto-initialize on first run

### Deployment Options

- **Development:** Python virtual environment (Option 3)
- **Production:** Docker (Option 1)
- **CI/CD:** Docker + automated builds
- **Air-Gapped:** Docker image transferred offline
- **Quick Testing:** pip install (Option 2)

For full operational details, see [User Manual](USER_MANUAL.md).

## Usage

### Running Felix

```bash
# Modern GUI (recommended)
python -m src.gui_ctk

# Original Tkinter GUI
python -m src.gui

# CLI workflow
python -m src.cli run "Your task here"

# REST API
uvicorn src.api.main:app --port 8000
```

### Basic Custom Workflow

```python
from src.core.helix import HelixGeometry
from src.communication.central_post import CentralPost
from src.agents.specialized_agents import ResearchAgent, AnalysisAgent, CriticAgent

# Initialize components
helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
central_post = CentralPost(max_agents=10, enable_metrics=True, enable_memory=True, llm_client=llm_client)

# Create and register agents
for agent in [ResearchAgent(), AnalysisAgent(), CriticAgent()]:
    central_post.register_agent(agent)

# Synthesize final output
synthesis_result = central_post.synthesize_agent_outputs(task_description="Your task")
```

### Workflow Steps

1. **Initialization**: Set up HelixGeometry, CentralPost, and agents
2. **Agent Spawning**: Create role-specialized agents with spawn time ranges
3. **Registration**: Register agents with CentralPost for message routing
4. **Task Processing**: Agents process tasks with position-aware LLM prompting
5. **Communication**: Share results via hub-spoke messaging
6. **Memory Storage**: Store insights in KnowledgeStore with compression
7. **Dynamic Adaptation**: Monitor confidence and spawn additional agents as needed
8. **Result Generation**: Synthesize final outputs through helical progression

For detailed workflow patterns, see [DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md).

### Testing

```bash
pytest tests/                    # All tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests
pytest tests/ -v --cov=src       # With coverage
```

## Command-Line Interface

Felix provides a comprehensive CLI for running workflows, checking status, and managing the system.

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

For complete CLI documentation, see [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md).

## Architecture

Felix follows a modular architecture with clear component interactions:

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/agents/` | Agent creation, spawning, specialization (Research, Analysis, Critic, System, FelixAgent) |
| `src/communication/` | Hub-spoke messaging, synthesis engine, coordinators |
| `src/core/` | Helical geometry algorithms |
| `src/llm/` | Multi-provider LLM clients with fallback |
| `src/memory/` | Persistent storage, context compression, workflow history |
| `src/knowledge/` | Autonomous knowledge brain, graph building, strategic comprehension |
| `src/prompts/` | Prompt management and pipeline system |
| `src/workflows/` | High-level task orchestration |
| `src/gui_ctk/` | Modern CustomTkinter GUI (11 tabs) |
| `src/gui/` | Original Tkinter GUI |
| `src/api/` | FastAPI REST endpoints |

### Coordinator Architecture

- **CentralPost**: Main coordination hub delegating to specialized subsystems
- **SynthesisEngine**: Intelligent synthesis of agent outputs
- **WebSearchCoordinator**: Confidence-based web search triggering
- **SystemCommandManager**: Command execution with trust and approval workflows
- **StreamingCoordinator**: Real-time token streaming
- **MemoryFacade**: Unified memory access layer
- **PerformanceMonitor**: Metrics tracking and analysis

For detailed architecture diagrams and data flows, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## GUI Interface

### Modern CustomTkinter GUI (Recommended)

The modern GUI in `src/gui_ctk/` provides a premium user experience:

```bash
python -m src.gui_ctk
```

**11 Tabs:**

- **Dashboard**: System overview with status cards and quick actions
- **Chat**: Claude-style conversational interface with:
  - Message bubbles with thinking views
  - Session management and history sidebar
  - Action bubbles for system command execution
  - Keyboard shortcuts (Enter to send, Shift+Enter for newlines)
- **Workflows**: Run and monitor multi-agent workflows with web search
- **Memory**: Browse and manage persistent memory storage
- **Agents**: View active agents and their helical positions
- **Approvals**: Pending approvals and decision history
- **Terminal**: Real-time command execution monitoring
- **Prompts**: Manage and edit prompt templates
- **Learning**: Task pattern learning and meta-cognition
- **Knowledge Brain**: Document ingestion and knowledge graph management
- **Settings**: Configure LLM providers, trust rules, and preferences

**Premium Components:**

- Responsive grid layouts
- Skeleton loaders for async operations
- Enhanced entry fields with validation
- Resizable separators
- Themed treeviews
- Status cards with animations

### Original Tkinter GUI

The classic GUI in `src/gui/` remains available:

```bash
python -m src.gui
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

Configuration via `config/trust_rules.yaml`. Command history persisted in `felix_system_actions.db`.

## Knowledge Brain

The Knowledge Brain system enables autonomous document learning and semantic retrieval.

### Core Features

- **Document Ingestion**: PDFs, markdown, text files, and code
- **Strategic Comprehension**: Outline-based prioritization for efficient processing
- **3-Tier Embeddings**: Automatic fallback (LM Studio -> TF-IDF -> FTS5)
- **Knowledge Graph**: Entity and relationship extraction
- **Gap Tracking**: Identifies missing knowledge
- **Coverage Analysis**: Measures completeness across domains
- **Meta-Learning**: Improves retrieval based on workflow feedback

### Configuration

Configure via the Settings tab or `config/llm.yaml`:

- **Watch Directories**: Directories to monitor for documents
- **Embedding Mode**: `auto`, `lm_studio`, `tfidf`, or `fts5`
- **Auto-Augment Workflows**: Inject relevant knowledge into workflow context
- **Processing Settings**: Chunk size, overlap, threads, memory limits

### How It Works

1. **Document Ingestion**: Monitors watch directories for new documents
2. **Strategic Analysis**: Creates document outlines, prioritizes high-value chunks
3. **Agentic Comprehension**: Uses Felix agents to understand content
4. **Knowledge Extraction**: Extracts concepts, entities, and relationships
5. **Graph Building**: Discovers connections via similarity and co-occurrence
6. **Semantic Search**: Makes knowledge retrievable via embedding system
7. **Continuous Learning**: Refines knowledge graph over time

## Documentation

- **[Quick Start Guide](QUICKSTART.md)**: Get up and running quickly
- **[CLI Guide](docs/CLI_GUIDE.md)**: Complete command-line interface reference
- **[Chat CLI Guide](docs/CHAT_CLI_GUIDE.md)**: Conversational CLI with sessions
- **[Configuration Reference](docs/CONFIGURATION.md)**: Complete configuration and tuning guide
- **[Architecture Overview](docs/ARCHITECTURE.md)**: System architecture and design patterns
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Development and extension guide
- **[LLM Provider Guide](docs/LLM_PROVIDER_GUIDE.md)**: Multi-provider setup and custom providers
- **[Coordinator Architecture](docs/COORDINATOR_ARCHITECTURE.md)**: Detailed coordinator documentation
- **[Knowledge Brain API](docs/KNOWLEDGE_BRAIN_API.md)**: Knowledge system integration
- **[Plugin API](docs/PLUGIN_API.md)**: Extending Felix with plugins
- **[REST API](docs/API_QUICKSTART.md)**: FastAPI endpoint documentation
- **[User Manual](USER_MANUAL.md)**: Complete setup, configuration, and operational guide
- **[Index](index.md)**: Framework overview and project structure

## Configuration Files

- `config/llm.yaml` - LLM provider selection and API keys
- `config/prompts.yaml` - Agent prompt templates
- `config/trust_rules.yaml` - Security rules for command execution
- `config/task_complexity_patterns.yaml` - Task classification patterns
- `config/chat_system_prompt.md` - FelixAgent identity configuration

## Databases (auto-created)

SQLite databases are created automatically in the project root:

- `felix_knowledge.db` - Knowledge brain storage
- `felix_memory.db` - Long-term memory
- `felix_workflow_history.db` - Workflow execution history
- `felix_task_memory.db` - Task pattern learning
- `felix_system_actions.db` - Command execution history

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make changes with comprehensive tests
4. Run linting: `ruff check src/` and `mypy src/`
5. Run tests: `pytest tests/`
6. Submit a pull request

For bugs or feature requests, please open an issue with detailed reproduction steps.

## License

MIT License - see LICENSE file for details.
