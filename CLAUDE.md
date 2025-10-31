# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Felix is a Python multi-agent AI framework that uses helical geometry for adaptive agent progression. It models agent behaviors along helical structures (spiral paths) to enable dynamic, scalable AI interactions with continuous evolution and optimization. The framework includes an autonomous knowledge brain system that enables continuous learning from documents through agentic comprehension, knowledge graph construction, and semantic retrieval with meta-learning.

The framework implements three core hypotheses:
- **H1**: Helical progression enhances agent adaptation (20% workload distribution improvement)
- **H2**: Hub-spoke communication optimizes resource allocation (15% efficiency gain)
- **H3**: Memory compression reduces latency (25% attention focus improvement)

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

# Run basic tests
python test_felix.py
python test_felix_advanced.py
python test_agents_integration.py

# Test Knowledge Brain system (6 comprehensive tests)
python test_knowledge_brain_system.py
```

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
   - **Note**: Synthesis is performed by CentralPost, not by a specialized agent

2. **Communication Hub** ([src/communication/central_post.py](src/communication/central_post.py))
   - `CentralPost`: O(N) hub-spoke message routing (vs O(N²) mesh)
   - `CentralPost Synthesis`: Smart hub performs final synthesis of all agent outputs
   - `AgentFactory`: Creates agents with helix positioning
   - `AgentRegistry`: Phase-based agent tracking (exploration/analysis/synthesis)
   - Agent awareness: Query team state, discover peers, coordinate collaboration
   - Handles up to 133 agents with efficient message queuing
   - Adaptive synthesis: temperature (0.2-0.4) and tokens (1500-3000) based on consensus

3. **Memory Systems** ([src/memory/](src/memory/))
   - `KnowledgeStore`: SQLite persistence in `felix_knowledge.db`
   - `TaskMemory`: Pattern storage in `felix_memory.db`
   - `WorkflowHistory`: Execution tracking in `felix_workflow_history.db`
   - `ContextCompression`: Abstractive compression (0.3 ratio)

4. **LLM Integration** ([src/llm/](src/llm/))
   - `LMStudioClient`: Local LLM via LM Studio (port 1234) with incremental token streaming
   - `TokenBudgetManager`: Adaptive token allocation (base: 2048)
   - `WebSearchClient`: DuckDuckGo and SearxNG integration with result caching and domain filtering
   - Temperature gradient: 1.0 (top/exploration) → 0.2 (bottom/synthesis)
   - Streaming support: Time-batched token delivery with callbacks

5. **Pipeline Processing** ([src/pipeline/](src/pipeline/))
   - `LinearPipeline`: Sequential task processing
   - `Chunking`: 512-token chunks for streaming

6. **Workflows** ([src/workflows/](src/workflows/))
   - `FelixWorkflow`: Integrated workflow with web search and task classification
   - `ContextBuilder`: Collaborative context management for agents
   - `TruthAssessment`: Framework for validating workflow outputs

7. **Knowledge Brain System** ([src/knowledge/](src/knowledge/))
   - `DocumentReader`: Multi-format document reading (PDF, TXT, MD, Python, JS, Java, C++) with semantic chunking
   - `EmbeddingProvider`: 3-tier embedding system with automatic fallback (LM Studio 768-dim → TF-IDF → FTS5 BM25)
   - `KnowledgeComprehensionEngine`: Agentic document understanding using Research, Analysis, and Critic agents
   - `KnowledgeGraphBuilder`: Relationship discovery via explicit mentions, embedding similarity (0.75), co-occurrence (5-chunk)
   - `KnowledgeDaemon`: Autonomous processor with 3 concurrent modes (batch, hourly refinement, file watching)
   - `KnowledgeRetriever`: Semantic search with meta-learning boost tracking historical usefulness
   - `WorkflowIntegration`: Bridge connecting knowledge brain to Felix workflows
   - **Zero External Dependencies**: Intelligent fallback ensures operation without cloud APIs

8. **Utilities** ([src/utils/](src/utils/))
   - `MarkdownFormatter`: Professional markdown formatting for synthesis results
   - Functions for detailed reports with agent metrics and performance summaries

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
- `knowledge_entries` table: Stores agent insights with domains, confidence scores, and abstractive summaries (extended with embedding, source_doc_id, chunk_index for Knowledge Brain)
- `document_sources` table: Tracks ingested documents with file paths, status, processing timestamps
- `knowledge_relationships` table: Bidirectional relationships between concepts (explicit mentions, similarity, co-occurrence)
- `knowledge_fts` virtual table: FTS5 full-text search index for BM25-ranked content retrieval
- `knowledge_usage` table: Meta-learning data tracking which knowledge helps which workflows
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

## Testing Approach

- `test_felix.py`: Basic import and component tests
- `test_felix_advanced.py`: Integration tests with mock LLM
- `test_agents_integration.py`: Agent spawning and communication tests
- `test_knowledge_brain_system.py`: Comprehensive Knowledge Brain tests (6 tests covering ingestion, comprehension, graph building, retrieval, meta-learning, daemon)

No formal test framework (pytest/unittest) - uses direct script execution.

## Important Notes

1. **Virtual Environment Required**: Always activate `.venv` before running to avoid dependency conflicts

2. **Mock vs Real LLM**: Example workflows use mock LLM by default. For real LLM, ensure LM Studio is running or provide valid OpenAI credentials

3. **Memory Databases**: Auto-created on first run. Located in project root as `*.db` files

4. **Token Budgets**: Configured for local 7B models (~2048 context). Adjust for larger models

5. **Agent Limits**: Default max 10 agents for local systems. Can scale to 133 with sufficient resources

6. **Hypothesis Validation**: Run benchmarks to verify expected gains (H1: 20%, H2: 15%, H3: 25%)

7. **Knowledge Brain System**: Fully optional - zero external dependencies through tiered fallback (LM Studio → TF-IDF → FTS5). Enable via GUI Settings tab. Monitors watch directories, uses agentic comprehension for document understanding, builds knowledge graphs, and provides semantic retrieval with meta-learning. Requires PyPDF2 for PDF support and watchdog for file monitoring (both optional - system adapts if missing)