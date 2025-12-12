# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Felix is a production-ready, air-gapped multi-agent AI framework designed for organizations requiring complete data isolation. It works entirely offline with zero external dependencies using a 3-tier fallback system (LM Studio → TF-IDF → SQLite FTS5).

## Build & Run Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install with optional features
pip install -e .[all]          # Everything
pip install -e .[api]          # REST API only
pip install -e .[knowledge]    # PDF/watchdog
pip install -e .[dev]          # Testing & linting

# Run application
python -m src.cli run "Your task here"    # CLI workflow
python -m src.gui_ctk                      # Modern CustomTkinter GUI
python -m src.gui                          # Original Tkinter GUI
uvicorn src.api.main:app --port 8000       # REST API

# Testing
pytest tests/                              # All tests
pytest tests/unit/                         # Unit tests only
pytest tests/integration/                  # Integration tests
pytest tests/ -v --cov=src                 # With coverage

# Linting (configured in pyproject.toml)
ruff check src/
mypy src/
```

## Architecture

### Helical Geometry Model
Agents progress from exploration (wide radius) to synthesis (narrow) along a 3D spiral path. Agent behavior adapts based on helix position:
- **Top**: Exploration phase - high temperature (1.0), broad search
- **Middle**: Analysis phase - moderate settings
- **Bottom**: Synthesis phase - low temperature (0.2), focused output

Core implementation: `src/core/helix.py`

### Hub-Spoke Communication (O(N) complexity)
All messages route through `CentralPost` instead of direct agent-to-agent (O(N²)):
- `src/communication/central_post.py` - Message orchestration hub
- `src/communication/agent_registry.py` - Agent lifecycle tracking
- `src/communication/synthesis_engine.py` - Intelligent output synthesis
- `src/communication/web_search_coordinator.py` - Confidence-based web search

### Three-Tier Embeddings Fallback
Knowledge retrieval degrades gracefully:
1. LM Studio embeddings (best quality, requires local LLM)
2. TF-IDF fallback (fast, no dependencies)
3. SQLite FTS5 (full-text search, always available)

Implementation: `src/knowledge/embeddings.py`

### Trust System for Command Execution
```
SAFE commands → Auto-execute (ls, pwd, date)
REVIEW commands → User approval required
BLOCKED commands → Never execute (rm -rf, credential access)
```

Configuration: `config/trust_rules.yaml`

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/agents/` | Agent creation, spawning, specialization (Research, Analysis, Critic, System) |
| `src/communication/` | Hub-spoke messaging and coordination |
| `src/core/` | Helical geometry algorithms |
| `src/llm/` | Multi-provider LLM clients (LM Studio, Anthropic, Gemini) with fallback |
| `src/memory/` | Persistent storage, context compression, workflow history |
| `src/knowledge/` | Autonomous knowledge brain, graph building, document comprehension |
| `src/workflows/` | High-level task orchestration |
| `src/gui_ctk/` | Modern CustomTkinter GUI (9 tabs) |
| `src/api/` | FastAPI REST endpoints |

## Entry Points

- **CLI**: `src/cli.py::main()` → `python -m src.cli`
- **GUI (Modern)**: `src/gui_ctk/__main__.py` → `python -m src.gui_ctk`
- **REST API**: `src/api/main.py` → `uvicorn src.api.main:app`
- **Package**: `felix` command (defined in pyproject.toml)

## Configuration Files

- `config/llm.yaml` - LLM provider selection and API keys
- `config/prompts.yaml` - Agent prompt templates
- `config/trust_rules.yaml` - Security rules for command execution
- `config/task_complexity_patterns.yaml` - Task classification patterns

## Databases (auto-created)

SQLite databases are created automatically in the project root:
- `felix_knowledge.db` - Knowledge brain storage
- `felix_memory.db` - Long-term memory
- `felix_workflow_history.db` - Workflow execution history
- `felix_task_memory.db` - Task pattern learning

## Python Version

Requires Python 3.12+. Supports 3.12, 3.13, 3.14 including free-threaded builds.
