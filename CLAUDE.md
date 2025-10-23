# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Felix is a Python multi-agent AI framework that uses helical geometry for adaptive agent progression. It models agent behaviors along helical structures (spiral paths) to enable dynamic, scalable AI interactions with continuous evolution and optimization.

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
```

### Running the Framework
```bash
# Run example workflow with mock LLM (no external server required)
python exp/example_workflow.py "Your task description here"

# Run comprehensive benchmarks to validate H1-H3 hypotheses
cd exp && python benchmark_felix.py

# Run the GUI interface (requires LM Studio running on port 1234)
python -m src.gui

# Run basic tests
python test_felix.py
python test_felix_advanced.py
python test_agents_integration.py
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

7. **Utilities** ([src/utils/](src/utils/))
   - `MarkdownFormatter`: Professional markdown formatting for synthesis results
   - Functions for detailed reports with agent metrics and performance summaries

### Agent Spawn Timing
Agents spawn at different normalized time ranges (0.0-1.0):
- Research: 0.0-0.25 (early exploration)
- Analysis: 0.2-0.6 (mid-phase processing)
- Critic: 0.4-0.7 (continuous validation)

**Note**: Synthesis is no longer an agent type. Final synthesis is performed by CentralPost when confidence threshold (≥0.80) is reached.

## Configuration

Felix uses YAML configuration (see [exp/optimal_parameters.md](exp/optimal_parameters.md) for tuning). Key parameters:

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
```

## Database Schema

### felix_knowledge.db
- `knowledge` table: Stores agent insights with domains, confidence scores, and abstractive summaries
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

The Tkinter GUI ([src/gui/](src/gui/)) provides five tabs with dark mode support:
1. **Dashboard**: Start/stop Felix system, monitor logs
2. **Workflows**: Run tasks through linear pipeline with web search, save formatted results
3. **Memory**: Browse/edit task memory and knowledge stores
4. **Agents**: Spawn and interact with agents
5. **Workflow History**: Browse past executions, view details, search and filter results

Additional features:
- Dark/light theme toggle with persistent preference
- Markdown export for synthesis results
- Real-time workflow execution tracking

Requires LM Studio running before starting Felix system via GUI.

## Testing Approach

- `test_felix.py`: Basic import and component tests
- `test_felix_advanced.py`: Integration tests with mock LLM
- `test_agents_integration.py`: Agent spawning and communication tests
- `exp/benchmark_felix.py`: Performance validation for H1-H3 hypotheses

No formal test framework (pytest/unittest) - uses direct script execution.

## Important Notes

1. **Virtual Environment Required**: Always activate `.venv` before running to avoid dependency conflicts

2. **Mock vs Real LLM**: Example workflows use mock LLM by default. For real LLM, ensure LM Studio is running or provide valid OpenAI credentials

3. **Memory Databases**: Auto-created on first run. Located in project root as `*.db` files

4. **Token Budgets**: Configured for local 7B models (~2048 context). Adjust for larger models

5. **Agent Limits**: Default max 10 agents for local systems. Can scale to 133 with sufficient resources

6. **Hypothesis Validation**: Run benchmarks to verify expected gains (H1: 20%, H2: 15%, H3: 25%)