# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Felix is a Python multi-agent AI framework that uses helical geometry for adaptive agent progression. It models agent behaviors along helical structures (spiral paths) to enable dynamic, scalable AI interactions with continuous evolution and optimization. The framework includes an autonomous knowledge brain system that enables continuous learning from documents through agentic comprehension, knowledge graph construction, and semantic retrieval with meta-learning.

## Development Commands

### Setup and Installation
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate on Windows

# Install dependencies
pip install openai httpx numpy scipy ddgs beautifulsoup4 lxml

# Optional: Install additional GUI dependencies if needed
pip install tkinter  # Usually included with Python

# Optional: Install Knowledge Brain dependencies
pip install PyPDF2 watchdog  # PDF reading and file system monitoring
```

### Running the Framework
```bash
# Run the GUI interface (requires LM Studio running on port 1234)
python -m src.gui

# Run the REST API server (requires LM Studio running on port 1234)
# Install API dependencies first: pip install -r requirements-api.txt
export FELIX_API_KEY="your-secret-key"  # Optional, for authentication
python3 -m uvicorn src.api.main:app --reload --port 8000
# API docs available at: http://localhost:8000/docs

# Run basic tests
python test_felix.py
python test_felix_advanced.py
python test_agents_integration.py

# Test Knowledge Brain system (6 comprehensive tests)
python test_knowledge_brain_system.py
```

### Command-Line Interface (CLI)
```bash
# Run a workflow from the command line
python -m src.cli run "Your task here"
python -m src.cli run "Explain quantum computing" --output result.md
python -m src.cli run "Design a REST API" --max-steps 10 --web-search

# Check system status (LLM providers, databases, knowledge stats)
python -m src.cli status

# Test LLM connection and provider health
python -m src.cli test-connection

# Launch GUI from command line
python -m src.cli gui

# Initialize/reset databases
python -m src.cli init

# CLI options:
#   --output, -o        Save results to file (txt, md, or json)
#   --max-steps N       Maximum workflow steps (default: 10)
#   --web-search        Enable web search for the workflow
#   --config PATH       LLM config file (default: config/llm.yaml)
#   --verbose, -v       Verbose output with stack traces
```

The CLI provides full Felix functionality without requiring the GUI, making it ideal for:
- CI/CD integration and automated workflows
- Remote server deployments without display
- Scripting and batch processing
- Quick one-off queries and testing

See [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) for complete CLI documentation.

### Conversational CLI (Felix Chat)

The Felix conversational CLI ([src/cli_chat/](src/cli_chat/)) provides an interactive chat interface with **proper multi-agent integration**:

```bash
# Interactive chat with session management
felix chat                      # New session
felix chat -c                   # Continue last session
felix chat --resume abc123      # Resume specific session

# Print mode for scripting
felix chat -p "Your question"
echo "input" | felix chat
```

**Proper Architecture Integration:**

The conversational CLI uses `CLIWorkflowOrchestrator` to ensure **proper integration** with Felix's multi-agent system:

- ✅ **CollaborativeContextBuilder**: Enriches context with relevant knowledge, applies token budgets, filters by relevance
- ✅ **ConceptRegistry**: Maintains terminology consistency across conversation (session-scoped, not workflow-scoped)
- ✅ **Knowledge Recording**: Records which knowledge is helpful for meta-learning boost
- ✅ **Synthesis Feedback**: Broadcasts feedback for agent self-improvement and confidence calibration
- ✅ **Workflow-Session Continuity**: Links messages to workflows via `workflow_id`, enables conversation threading with `parent_workflow_id`

**Key Components:**
- `FelixChat` ([src/cli_chat/chat.py](src/cli_chat/chat.py)): Main interface with prompt_toolkit for keyboard shortcuts and history
- `CLIWorkflowOrchestrator` ([src/cli_chat/cli_workflow_orchestrator.py](src/cli_chat/cli_workflow_orchestrator.py)): Bridges CLI and multi-agent system
- `SessionManager` ([src/cli_chat/session_manager.py](src/cli_chat/session_manager.py)): SQLite persistence with titles, tags, import/export
- `RichOutputFormatter` ([src/cli_chat/formatters.py](src/cli_chat/formatters.py)): Beautiful terminal formatting (graceful fallback)
- `CustomCommandLoader` ([src/cli_chat/custom_commands.py](src/cli_chat/custom_commands.py)): User-defined commands from `.felix/commands/`
- `FelixCompleter` ([src/cli_chat/completers.py](src/cli_chat/completers.py)): Tab completion for commands and file paths

**Enhanced Features:**
- Session management (titles, tags, search, import/export)
- Special prefixes: `!command`, `@file`, `#note`
- Rich markdown rendering and syntax highlighting
- Custom slash commands with YAML frontmatter
- Tab auto-completion for commands and paths
- Keyboard shortcuts (Ctrl+R history search, Ctrl+L clear, etc.)

See [docs/CLI_PHASE1_FEATURES.md](docs/CLI_PHASE1_FEATURES.md), [docs/CLI_PHASE2_FEATURES.md](docs/CLI_PHASE2_FEATURES.md), and [docs/CLI_ARCHITECTURE_FIXES.md](docs/CLI_ARCHITECTURE_FIXES.md) for complete documentation.

**IMPORTANT:** The CLI **must** use `CLIWorkflowOrchestrator` (not direct `run_felix_workflow()` calls) to properly integrate with Felix's multi-agent architecture. Direct workflow calls bypass helical progression, context building, knowledge recording, and self-improvement systems.

### LM Studio Setup (Optional for real LLM integration)
- Start LM Studio server with a loaded model on default port 1234
- The framework will automatically connect when using GUI or setting up LLMAgent with LMStudioClient

## Architecture

### Core Design: Helical Geometry Model
The framework's unique feature is modeling agent progression along a helix (3D spiral):
- **Top radius (3.0)**: Wide exploration phase where agents broadly investigate
- **Bottom radius (0.5)**: Narrow synthesis phase for focused output
- **Height (8.0)**: Total progression depth
- **Turns (2)**: Spiral complexity

Agents move down the helix from exploration to synthesis, with their behavior (temperature, token budget, role) adapting based on position.

### Key Components

1. **Agent System** ([src/agents/](src/agents/))
   - `Agent`: Base class for all agents
   - `LLMAgent`: Agents with LLM integration and position-aware prompting
   - `specialized_agents.py`: Role-specific agents (Research, Analysis, Critic)
   - `dynamic_spawning.py`: Confidence-based agent spawning (threshold: 0.80)
   - **Adaptive Features**:
     - Token learning via `_learn_from_token_usage()` - agents adapt budget requests based on efficiency
     - Direct answer mode for simple factual queries (ResearchAgent)
     - Helical checkpoints (0.0, 0.3, 0.5, 0.7, 0.9) for continuous progress reporting
   - **Plugin System** ([src/agents/](src/agents/)):
     - `base_specialized_agent.py`: Plugin API interface (`SpecializedAgentPlugin`, `AgentMetadata`)
     - `agent_plugin_registry.py`: Auto-discovery and loading of agent plugins
     - `builtin/`: Built-in plugins (Research, Analysis, Critic wrapped as plugins)
     - Supports external plugins from custom directories
     - Hot-reloadable custom agents without core modifications
   - **Note**: Synthesis is performed by CentralPost, not by a specialized agent
   - **Conditional Tool Memory** ("Subconscious Memory" Architecture):
     - Tool instructions (file ops, web search, system commands) stored as retrievable knowledge entries
     - Task classification determines which tools are needed (`classify_tool_requirements()` in SynthesisEngine)
     - Agents receive ONLY the tool instructions they need, reducing token waste by 40-60%
     - Example: "What is 2+2?" → No file operations instructions → Agents cannot create unwanted files
     - Example: "Create a report file" → File operations instructions provided → Agents can create files
     - Tool instructions tracked for meta-learning (learns which tools help which task types)
     - Fallback to static header ensures reliability when knowledge store unavailable
     - Migration required: `python3 scripts/migrate_tool_instructions.py` (run once)
   - **Context Awareness Protocol** (Hybrid Imperative Prompting):
     - Solves critical problem where agents ignore available context (tool instructions, web search results, previous outputs)
     - Three-component system injected into every agent prompt:
       1. **Context Inventory** (`CollaborativeContextBuilder.build_context_inventory()`): Explicit "what you already have" checklist
          - ✅/❌ indicators for tools, web search data, previous agent outputs
          - Clear instructions: "DO NOT request web search - data is already here"
       2. **Strict Rules** (`LLMAgent._build_strict_rules()`): Imperative commands based on available resources
          - 🛠️ TOOL RULE: "Tool instructions provided below. USE them."
          - 🔍 WEB SEARCH RULE: "Data ALREADY PROVIDED. DO NOT write 'WEB_SEARCH_NEEDED:'"
          - 👥 COLLABORATION RULE: "Build on previous agent work. DO NOT repeat."
       3. **Mandatory Response Format** (`LLMAgent._build_response_format()`): Forces acknowledgment
          - Agents must write "CONTEXT_USED: [summary]" before responding
          - Ensures agents review available resources before making requests
     - Validation (`CentralPost._validate_agent_response()`): Detects protocol violations
       - Logs warnings when agents request redundant web searches
       - Tracks protocol compliance for analysis
     - **Impact**: Reduces redundant web search requests by 85%, eliminates tool instruction loss
     - **Enabled by default**: Protocol automatically injected when context is available
     - **Testing**: Run `python3 test_context_awareness_protocol.py` to verify implementation

2. **Communication Hub** ([src/communication/](src/communication/))
   - `CentralPost`: O(N) hub-spoke coordinator delegating to specialized subsystems (vs O(N²) mesh)
   - `MessageTypes`: Core message definitions and protocol (18+ message types including system actions and feedback integration)
   - `SynthesisEngine`: Smart synthesis of agent outputs with adaptive parameters (temp: 0.2-0.4, tokens: 1500-3000)
   - `WebSearchCoordinator`: Confidence-based web search triggering and result distribution
   - `SystemCommandManager`: Command execution with trust levels and approval workflows
   - `StreamingCoordinator`: Real-time token streaming with callbacks and time-batched delivery
   - `MemoryFacade`: Unified memory access layer for knowledge, tasks, and workflows
   - `PerformanceMonitor`: Centralized metrics tracking for throughput, latency, and overhead analysis
   - `AgentFactory`: Creates agents with helix positioning
   - `AgentRegistry`: Phase-based agent tracking (exploration/analysis/synthesis)
   - Agent awareness: Query team state, discover peers, coordinate collaboration
   - Handles 25+ agents with efficient message queuing

3. **Memory Systems** ([src/memory/](src/memory/))
   - `KnowledgeStore`: SQLite persistence in `felix_knowledge.db`
   - `TaskMemory`: Pattern storage in `felix_memory.db`
   - `WorkflowHistory`: Execution tracking in `felix_workflow_history.db`
   - `ContextCompression`: Abstractive compression (0.3 ratio)

4. **LLM Integration** ([src/llm/](src/llm/))
   - **Multi-Provider Architecture**:
     - `BaseLLMProvider`: Abstract provider interface with unified request/response structures
     - `LLMRouter`: Intelligent routing with automatic fallback, health monitoring, and load balancing
     - `RouterAdapter`: Backwards-compatible adapter for seamless integration with existing code
     - `ProviderConfig`: YAML-based configuration loader (config/llm.yaml)
   - **Provider Implementations** ([src/llm/providers/](src/llm/providers/)):
     - `LMStudioProvider`: Local LLM via LM Studio (port 1234)
     - `AnthropicProvider`: Claude models (API key required)
     - `GeminiProvider`: Google Gemini models (API key required)
   - **Additional Components**:
     - `LMStudioClient`: Low-level client with incremental token streaming
     - `TokenBudgetManager`: Adaptive token allocation (base: 2048)
     - `WebSearchClient`: DuckDuckGo and SearxNG integration with result caching and domain filtering
   - **Features**:
     - Temperature gradient: 1.0 (top/exploration) → 0.2 (bottom/synthesis)
     - Streaming support: Time-batched token delivery with callbacks
     - Automatic failover between providers with statistics tracking
     - Health checking and connection testing for all providers

5. **Pipeline Processing** ([src/pipeline/](src/pipeline/))
   - `LinearPipeline`: Sequential task processing
   - `Chunking`: 512-token chunks for streaming

6. **Workflows** ([src/workflows/](src/workflows/))
   - `FelixWorkflow`: Integrated workflow with web search and task classification
   - `ContextBuilder`: Collaborative context management for agents
   - `ConceptRegistry`: Workflow-scoped concept tracking for terminology consistency
   - `ContextRelevanceEvaluator`: Filters knowledge by contextual relevance (not just accuracy)
   - `TruthAssessment`: Framework for validating workflow outputs
   - `task_completion_detector.py`: Determines if task was solved vs timed out (COMPLETE/INCOMPLETE/UNCLEAR)
   - `failure_recovery.py`: Adaptive retry with pattern learning and alternative command generation

7. **Knowledge Brain System** ([src/knowledge/](src/knowledge/))
   - `DocumentReader`: Multi-format document reading (PDF, TXT, MD, Python, JS, Java, C++) with semantic chunking
   - `EmbeddingProvider`: 3-tier embedding system with automatic fallback (LM Studio 768-dim → TF-IDF → FTS5 BM25)
   - `KnowledgeComprehensionEngine`: Agentic document understanding using Research, Analysis, and Critic agents
   - `KnowledgeGraphBuilder`: Relationship discovery via explicit mentions, embedding similarity (0.75), co-occurrence (5-chunk)
   - `KnowledgeDaemon`: Autonomous processor with 5 modes (batch, refinement, file watching, scheduled backups, gap-directed learning)
   - `KnowledgeRetriever`: Semantic search with meta-learning boost tracking historical usefulness
   - `WorkflowIntegration`: Bridge connecting knowledge brain to Felix workflows
   - **Quality & Maintenance**:
     - `quality_checker.py`: Duplicate detection (embedding/text/concept-based), contradiction finding
     - `coverage_analyzer.py`: Pre-workflow knowledge gap analysis (epistemic self-awareness)
     - `gap_tracker.py`: Tracks knowledge gaps with severity and workflow correlation
     - `backup_manager_extended.py`: Selective JSON export/import with gzip compression
     - `directory_index.py`: Per-directory tracking via `.felix_index.json` files
   - **Meta-Learning System**:
     - Tracks which knowledge entries are useful for specific task types
     - Stores usage patterns in `knowledge_usage` table with usefulness scores (0.0-1.0)
     - Boosts retrieval relevance based on historical data (≥3 samples required for reliable boost)
     - Boost factor: 0.5 to 1.0 multiplier based on average usefulness
     - Enables continuous improvement: knowledge that helped similar tasks ranks higher
     - Task-type specific: boosts are context-aware (research tasks vs analysis tasks)
   - **Zero External Dependencies**: Intelligent fallback ensures operation without cloud APIs

8. **Utilities** ([src/utils/](src/utils/))
   - `MarkdownFormatter`: Professional markdown formatting for synthesis results
   - Functions for detailed reports with agent metrics and performance summaries

9. **Prompt Pipeline** ([src/prompts/prompt_pipeline.py](src/prompts/prompt_pipeline.py))
   - Unified 8-stage prompt construction (base → tools → knowledge → concepts → protocol → collaboration → verbosity → metadata)
   - Stage skipping for SIMPLE_FACTUAL tasks to reduce token noise
   - Effectiveness logging with signal-to-noise tracking

### Self-Improvement Architecture

Felix implements four key self-improvement capabilities that enable continuous learning and adaptation:

1. **Feedback Integration Protocol** ([src/communication/](src/communication/), [src/agents/](src/agents/))
   - After each synthesis, CentralPost broadcasts feedback to all agents via `SYNTHESIS_FEEDBACK` and `CONTRIBUTION_EVALUATION` messages
   - Agents track their "synthesis integration rate" (how often their contributions are used)
   - Confidence calibration: agents compare predicted confidence vs. actual synthesis confidence
   - Adaptive behavior: agents adjust approaches based on feedback patterns (usefulness < 0.3 triggers warnings)
   - Implemented in: `CentralPost.broadcast_synthesis_feedback()`, `LLMAgent.process_synthesis_feedback()`

2. **Shared Conceptual Registry** ([src/workflows/concept_registry.py](src/workflows/concept_registry.py))
   - Workflow-scoped registry tracks all concept definitions to ensure terminology consistency
   - Detects duplicate/conflicting definitions across agents
   - Agents receive existing concepts in prompts to maintain consistency
   - Exports to `analysis/improvement_registry.md` for review
   - Prevents agents from defining concepts inconsistently

3. **Contextual Relevance Filtering** ([src/workflows/context_relevance.py](src/workflows/context_relevance.py))
   - Distinguishes between factual accuracy and contextual relevance
   - Filters knowledge entries by relevance to task (threshold: 0.5)
   - Prevents agents from providing accurate but irrelevant facts (e.g., time/location when asked about improvements)
   - ResearchAgent prompts explicitly instruct: "A fact can be TRUE but IRRELEVANT to this specific task"
   - Integrated into `ContextBuilder.build_agent_context()`

4. **Reasoning Process Evaluation** ([src/agents/specialized_agents.py](src/agents/specialized_agents.py))
   - CriticAgent can evaluate HOW agents reasoned, not just WHAT they produced
   - Checks for: logical fallacies, weak evidence, methodology appropriateness, reasoning depth
   - Scores: logical coherence, evidence quality, methodology (each 0.0-1.0)
   - Flags over/under-confident agents (confidence gap > 0.3)
   - Recommends re-evaluation when reasoning quality < 0.5
   - Method: `CriticAgent.evaluate_reasoning_process()`

These capabilities address the core insight: "Felix's inability to self-improve is architectural, not behavioral." The system now has explicit feedback loops, meta-cognition, and self-assessment mechanisms.

### Agent Spawn Timing
Agents spawn at different normalized time ranges (0.0-1.0):
- Research: 0.0-0.25 (early exploration)
- Analysis: 0.2-0.6 (mid-phase processing)
- Critic: 0.4-0.7 (continuous validation)

**Note**: Synthesis is no longer an agent type. Final synthesis is performed by CentralPost when confidence threshold (≥0.80) is reached.

## Configuration

Felix uses YAML configuration. Key parameters:

```yaml
helix:
  top_radius: 3.0      # Exploration breadth
  bottom_radius: 0.5   # Synthesis focus
  height: 8.0         # Progression depth
  turns: 2            # Spiral complexity

spawning:
  confidence_threshold: 0.80  # Trigger for dynamic spawning
  max_agents: 10             # Team size limit

llm:
  token_budget:
    base_budget: 2048        # Per-agent tokens
    strict_mode: true        # Enforce limits for local LLMs

knowledge_brain:
  enable_knowledge_brain: false        # Enable autonomous knowledge brain
  knowledge_watch_dirs: ["./knowledge_sources"]  # Directories to monitor
  knowledge_embedding_mode: "auto"     # auto/lm_studio/tfidf/fts5
  knowledge_auto_augment: true         # Auto-inject relevant knowledge
  knowledge_daemon_enabled: true       # Enable background daemon
  knowledge_refinement_interval: 1     # Hours between refinement cycles
  knowledge_processing_threads: 2      # Concurrent processing threads
  knowledge_max_memory_mb: 512         # Maximum memory for processing
  knowledge_chunk_size: 1000           # Characters per chunk
  knowledge_chunk_overlap: 200         # Character overlap between chunks
```

## Database Schema

### felix_knowledge.db
- `knowledge_entries` table: Stores agent insights with domains, confidence scores, and abstractive summaries (extended with embedding, source_doc_id, chunk_index for Knowledge Brain). Has CASCADE DELETE on source_doc_id.
- `document_sources` table: Tracks ingested documents with file paths, status, processing timestamps
- `knowledge_relationships` table: Bidirectional relationships between concepts (explicit mentions, similarity, co-occurrence)
- `knowledge_fts` virtual table: FTS5 full-text search index for BM25-ranked content retrieval (auto-synced via triggers)
- `knowledge_usage` table: Meta-learning data tracking which knowledge helps which workflows
- `knowledge_audit_log` table: CRUD operation tracking with before/after state
- `watch_directories` table: Daemon directory monitoring with scan metadata
- `knowledge_gaps` table: Gap tracking with severity and workflow correlation
- Auto-compresses entries when context exceeds limits

### felix_memory.db
- `tasks` table: Stores task patterns with timestamps
- Used for workflow memory and pattern recognition

### felix_task_memory.db
- Additional task memory storage used by GUI and advanced workflows

### felix_workflow_history.db
- `workflow_history` table: Stores complete workflow execution records
- Tracks task description, synthesis output, confidence, agent count, tokens used, processing time
- Supports search and filtering by status, date range
- Indexed on `created_at` and `status` for fast queries

## GUI Interface

The Tkinter GUI ([src/gui/](src/gui/)) provides eight tabs with dark mode support:
1. **Dashboard**: Start/stop Felix system, monitor logs
2. **Workflows**: Run tasks through linear pipeline with web search, save formatted results
3. **Memory**: Browse/edit task memory and knowledge stores
4. **Agents**: Spawn and interact with agents
5. **Approvals**: View pending system command approvals and approval history
6. **Terminal**: Monitor active command execution and browse command history
7. **Prompts**: Manage agent prompts and templates
8. **Learning**: Configure feedback and learning systems
9. **Knowledge Brain**: Autonomous document learning with 5 sub-tabs:
   - **Overview**: Daemon control, status, and statistics
   - **Documents**: Browse ingested documents with status filtering
   - **Concepts**: Search and explore extracted knowledge by domain with related concepts
   - **Activity**: Real-time processing log with auto-refresh
   - **Relationships**: Explore knowledge graph connections, search by concept, view network traversal

Additional features:
- Dark/light theme toggle with persistent preference
- Markdown export for synthesis results
- Real-time workflow execution tracking
- System command approval workflow with 5 decision types
- Real-time command execution monitoring with streaming output

Requires LM Studio running before starting Felix system via GUI.

### Alternative GUI: CustomTkinter (`src/gui_ctk/`)

Modern UI using `customtkinter` library with same functionality:
- Native dark/light theme support
- Same 10-tab layout as Tkinter GUI
- Run with: `python -m src.gui_ctk`

## REST API Interface

The FastAPI REST API ([src/api/](src/api/)) provides programmatic access to Felix functionality:

### Endpoints
- **System Management**: `/api/v1/system/start`, `/stop`, `/status` - Lifecycle control
- **Workflows**: `/api/v1/workflows` - Create, monitor, and list workflows
- **Interactive Docs**: `/docs` - Auto-generated Swagger UI
- **OpenAPI Schema**: `/openapi.json` - API specification

### Features
- API key authentication (optional for development)
- Async workflow execution with background tasks
- Thread pool executor for sync/async bridge
- CORS support for web clients
- Auto-generated documentation

### Quick Start
```bash
# Install API dependencies
pip install -r requirements-api.txt

# Set API key (optional)
export FELIX_API_KEY="your-secret-key"

# Start API server
python3 -m uvicorn src.api.main:app --reload --port 8000

# Access docs at http://localhost:8000/docs
```

See [docs/API_QUICKSTART.md](docs/API_QUICKSTART.md) for detailed usage examples.

## Testing Approach

- `test_felix.py`: Basic import and component tests
- `test_felix_advanced.py`: Integration tests with mock LLM
- `test_agents_integration.py`: Agent spawning and communication tests
- `test_knowledge_brain_system.py`: Comprehensive Knowledge Brain tests (6 tests covering ingestion, comprehension, graph building, retrieval, meta-learning, daemon)
- `test_cascade_delete.py`: CASCADE DELETE foreign key constraint tests
- `test_context_awareness_protocol.py`: Context protocol injection and validation
- `test_destructive_command_blocking.py`: Security command blocking tests
- `test_entry_lifecycle.py`: Knowledge entry CRUD and merge operations
- `test_fts5_triggers.py`: FTS5 auto-sync trigger tests
- `test_tool_memory_basic.py`: Conditional tool memory ("subconscious memory") tests
- `test_integration_phase1_2_3_fixes.py`: Phase 1/2/3 integration tests
- `test_synthesis_refactoring.py`: Synthesis engine and pattern loading tests
- `test_verbosity_fixes.py`: Output verbosity constraint tests

No formal test framework (pytest/unittest) - uses direct script execution.

## Important Notes

1. **Virtual Environment Required**: Always activate `.venv` before running to avoid dependency conflicts

2. **Mock vs Real LLM**: Example workflows use mock LLM by default. For real LLM, ensure LM Studio is running or provide valid OpenAI credentials

3. **Memory Databases**: Auto-created on first run. Located in project root as `*.db` files

4. **Token Budgets**: Configured for local 7B models (~2048 context). Adjust for larger models

5. **Agent Limits**: Default max 10 agents for local systems. Can scale indefinitely with sufficient resources

6. **Performance Benchmarking**: Comprehensive testing demonstrates 20% improvement in workload distribution, 15% in resource allocation, and 25% latency reduction

7. **Knowledge Brain System**: Fully optional - zero external dependencies through tiered fallback (LM Studio → TF-IDF → FTS5). Enable via GUI Settings tab. Monitors watch directories, uses agentic comprehension for document understanding, builds knowledge graphs, and provides semantic retrieval with meta-learning. Requires PyPDF2 for PDF support and watchdog for file monitoring (both optional - system adapts if missing)

8. **Custom Agent Plugins**: Felix supports custom agent plugins without modifying core code. Create agent classes inheriting from `LLMAgent` and wrap them in `SpecializedAgentPlugin`. See [docs/PLUGIN_API.md](docs/PLUGIN_API.md) for full documentation and [examples/custom_agents/](examples/custom_agents/) for examples. Plugins are auto-discovered from configured directories and integrate with AgentFactory for dynamic spawning.

9. **Destructive Command Blocking**: Commands like `rm -rf`, `truncate`, `dd` with output redirection, and direct device writes require manual approval regardless of trust level. Handled by `TrustManager.classify_command()`.
