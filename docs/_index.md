# Felix Framework - Source Code Index

## Overview

Felix is a Python multi-agent AI framework that uses **helical geometry** for adaptive agent progression. It models agent behaviors along helical structures (3D spiral paths) to enable dynamic, scalable AI interactions with continuous evolution and optimization.

The framework implements three core hypotheses:
- **H1**: Helical progression enhances agent adaptation (20% workload distribution improvement)
- **H2**: Hub-spoke communication optimizes resource allocation (15% efficiency gain)
- **H3**: Memory compression reduces latency (25% attention focus improvement)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GUI Interface (Tkinter)                 │
│    8 tabs: Dashboard, Workflows, Memory, Agents,            │
│    Approvals, Terminal, Learning, Knowledge Brain           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│                    Felix System                             │
│          (Central coordinator for all components)           │
└─────┬──────┬──────┬──────┬──────┬──────┬──────┬───────────┘
      │      │      │      │      │      │      │
      ▼      ▼      ▼      ▼      ▼      ▼      ▼
    Agents Comm  Memory  Know  LLM  Workflow Learning
                        Brain
```

## Module Directory

### Core Systems

#### [agents/](../src/agents/_index.md)
**Agent System & Spawning**
- Base agent classes and lifecycle management
- LLM-enabled agents with helix position awareness
- Specialized role-based agents (Research, Analysis, Critic)
- Dynamic confidence-based agent spawning
- Plugin architecture for extensibility

**Key Classes**: `Agent`, `LLMAgent`, `ResearchAgent`, `AnalysisAgent`, `CriticAgent`, `DynamicSpawning`

---

#### [communication/](../src/communication/_index.md)
**Hub-Spoke Communication**
- O(N) hub-spoke message routing vs O(N²) mesh
- CentralPost coordinator for agent messaging
- Specialized coordinators (synthesis, web search, system commands, streaming)
- Agent awareness and team coordination
- Adaptive synthesis based on consensus

**Key Classes**: `CentralPost`, `AgentRegistry`, `AgentFactory`, `SynthesisEngine`, `WebSearchCoordinator`

---

#### [core/](../src/core/_index.md)
**Helix Geometry Foundation**
- Parametric 3D helix modeling
- Agent positioning and behavior adaptation
- Temperature gradient (exploration → synthesis)
- Mathematical foundation for framework

**Key Classes**: `HelixGeometry`

---

### Intelligence & Learning

#### [workflows/](../src/workflows/_index.md)
**Task Orchestration**
- High-level multi-agent workflow coordination
- Web search and knowledge integration
- Context building and truth assessment
- Workflow history tracking

**Key Functions**: `run_felix_workflow()`, `assess_answer_confidence()`

---

#### [knowledge/](../src/knowledge/_index.md)
**Autonomous Knowledge Brain**
- Multi-format document ingestion (PDF, TXT, MD, code)
- Agentic comprehension using multi-agent analysis
- Three-tier embedding system (LM Studio → TF-IDF → FTS5)
- Knowledge graph construction and semantic retrieval
- Meta-learning optimization
- Autonomous daemon with batch/refinement/watch modes

**Key Classes**: `DocumentReader`, `KnowledgeComprehensionEngine`, `EmbeddingProvider`, `KnowledgeGraphBuilder`, `KnowledgeRetriever`, `KnowledgeDaemon`

---

#### [learning/](../src/learning/_index.md)
**Adaptive Learning Systems**
- Pattern recognition from workflow history
- Multi-strategy recommendation engine
- Dynamic threshold optimization
- Confidence calibration
- Continuous framework improvement

**Key Classes**: `PatternLearner`, `RecommendationEngine`, `ThresholdLearner`, `ConfidenceCalibrator`

---

#### [feedback/](../src/feedback/_index.md)
**User Feedback Collection**
- Structured feedback on workflows and knowledge
- Pattern recognition in feedback
- Integration with learning systems
- Actionable improvement identification

**Key Classes**: `FeedbackManager`, `FeedbackIntegrator`

---

### Infrastructure

#### [memory/](../src/memory/_index.md)
**Multi-Layer Persistence**
- Knowledge store with semantic search
- Task memory and pattern storage
- Workflow history tracking
- Context compression
- Agent performance tracking

**Key Classes**: `KnowledgeStore`, `TaskMemory`, `WorkflowHistory`, `ContextCompressor`

**Databases**: `felix_knowledge.db`, `felix_memory.db`, `felix_task_memory.db`, `felix_workflow_history.db`

---

#### [llm/](../src/llm/_index.md)
**Language Model Integration**
- Local LLM via LM Studio (default port 1234)
- Multi-provider routing (LM Studio, Anthropic, Gemini)
- Adaptive token budget management
- Web search integration (DuckDuckGo, SearxNG)
- Load balancing across servers
- Streaming token delivery

**Key Classes**: `LMStudioClient`, `LLMRouter`, `TokenBudgetManager`, `WebSearchClient`

---

#### [execution/](../src/execution/_index.md)
**Safe Command Execution**
- System command execution with sandboxing
- Multi-stage approval workflow
- Trust level assessment and management
- Command history tracking
- Error categorization

**Key Classes**: `SystemExecutor`, `ApprovalManager`, `TrustManager`, `CommandHistory`

---

#### [migration/](../src/migration/_index.md)
**Database Schema Evolution**
- Versioned migrations with rollback support
- Automatic backups before changes
- Transaction safety (ACID guarantees)
- Separate migrations for task/knowledge/system schemas

**Key Classes**: `Migration`, `MigrationManager`, `BackupManager`

---

### User Interface

#### [gui/](../src/gui/_index.md)
**Tkinter GUI Interface**
- 8 main tabs for system interaction
- Dark/light theme support
- Real-time log streaming
- Workflow execution and history
- Memory browsing and editing
- Agent spawning and interaction
- Command approval workflow
- Terminal monitoring with streaming output
- Knowledge Brain control (5 sub-tabs)

**Key Classes**: `MainApp`, `FelixSystem`, various Frame classes

---

### Utilities

#### [prompts/](../src/prompts/_index.md)
**Prompt Management**
- Centralized prompt template storage
- Variable substitution
- Version control for prompts
- A/B testing support
- Role-specific prompt optimization

**Key Classes**: `PromptManager`

---

#### [utils/](../src/utils/_index.md)
**Markdown Formatting**
- Professional synthesis output formatting
- Detailed reports with agent metrics
- Workflow statistics presentation
- File export utilities

**Key Functions**: `format_synthesis_markdown()`, `format_synthesis_markdown_detailed()`, `save_markdown_to_file()`

---

#### [pipeline/](../src/pipeline/_index.md)
**Data Processing Pipelines**
- Progressive token streaming
- Chunked content processing
- Rolling summarization
- Latency reduction through early processing

**Key Classes**: `ProgressiveProcessor`, `ContentSummarizer`, `ChunkedResult`

---

## Component Interactions

### Agent Workflow
```
1. User submits task via GUI/CLI
2. FelixSystem initializes CentralPost and AgentFactory
3. AgentFactory creates agents positioned on helix
4. Agents process task with position-aware behavior
5. DynamicSpawning monitors confidence, spawns if needed
6. CentralPost routes messages (O(N) hub-spoke)
7. SynthesisEngine combines results when confident
8. Results saved to WorkflowHistory and KnowledgeStore
9. GUI displays formatted markdown output
```

### Knowledge Brain Workflow
```
1. Documents placed in watch directories
2. KnowledgeDaemon detects new files
3. DocumentReader ingests and chunks content
4. KnowledgeComprehensionEngine spawns Research/Analysis/Critic agents
5. Agents extract concepts, entities, relationships
6. KnowledgeGraphBuilder creates bidirectional links
7. EmbeddingProvider generates vectors (3-tier fallback)
8. KnowledgeStore persists with FTS5 index
9. KnowledgeRetriever provides semantic search
10. WorkflowIntegration auto-injects relevant knowledge
11. Meta-learning tracks usefulness
```

### Learning Cycle
```
1. Workflow executes with specific configuration
2. WorkflowHistory records metrics
3. User provides feedback via GUI
4. FeedbackManager stores structured feedback
5. PatternLearner identifies recurring patterns
6. ThresholdLearner optimizes spawning thresholds
7. ConfidenceCalibrator adjusts agent confidence
8. RecommendationEngine suggests improvements
9. Future workflows benefit from learned optimizations
```

## Key Design Patterns

### 1. Helical Progression
Agents move from wide exploration (top) to narrow synthesis (bottom) along 3D spiral, adapting temperature, token budget, and role based on position.

### 2. Hub-Spoke Communication
Single CentralPost hub routes all messages (O(N) complexity) instead of mesh topology (O(N²)), enabling efficient scaling to 25+ agents.

### 3. Three-Tier Fallback
Knowledge Brain operates without external dependencies through LM Studio → TF-IDF → FTS5 BM25 fallback chain.

### 4. Dynamic Spawning
Agents spawn automatically when team confidence < 0.80, with spawn timing determined by role and normalized time.

### 5. Meta-Learning
System tracks what knowledge helps which workflows, which patterns succeed, and which thresholds optimize outcomes.

### 6. Progressive Processing
LLM outputs stream in chunks, enabling early validation and parallel processing to reduce latency.

### 7. Component Coordinators
Specialized coordinators (SynthesisEngine, WebSearchCoordinator, SystemCommandManager, StreamingCoordinator) extracted from CentralPost for clean separation.

## Configuration

See [CLAUDE.md](../CLAUDE.md) for complete configuration reference.

Key parameters:
- **Helix**: `top_radius: 3.0`, `bottom_radius: 0.5`, `height: 8.0`, `turns: 2`
- **Spawning**: `confidence_threshold: 0.80`, `max_agents: 10`
- **LLM**: `base_budget: 2048`, `strict_mode: true`
- **Knowledge Brain**: `embedding_mode: auto`, `chunk_size: 1000`, `chunk_overlap: 200`

## Getting Started

### Prerequisites
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install openai httpx numpy scipy ddgs beautifulsoup4 lxml

# Optional: Knowledge Brain
pip install PyPDF2 watchdog
```

### Running Felix
```bash
# GUI interface (requires LM Studio)
python -m src.gui

# CLI
python -m src.cli run "Analyze quantum computing trends"
```

### LM Studio Setup
1. Download and install LM Studio
2. Load a model (7B recommended for local)
3. Start server on default port 1234
4. Felix auto-connects when system starts

## Testing

```bash
# Basic tests
python test_felix.py
python test_felix_advanced.py

# Agent integration
python test_agents_integration.py

# Knowledge Brain (6 comprehensive tests)
python test_knowledge_brain_system.py
```

## Development Guidelines

1. **Always activate venv**: `source .venv/bin/activate`
2. **Follow module structure**: Keep related functionality together
3. **Update migrations**: Create migration for schema changes
4. **Document classes**: Include purpose, key methods, usage examples
5. **Test before committing**: Run relevant test files
6. **Update indices**: Keep _index.md files current when adding features

## Additional Resources

- **Project Instructions**: [CLAUDE.md](../CLAUDE.md)
- **Quick Start**: [QUICKSTART.md](../QUICKSTART.md)
- **README**: [README.md](../README.md)
- **Example Scripts**: [examples/](../examples/)

## Questions or Issues?

Refer to module-specific indices above for detailed documentation on each component.
