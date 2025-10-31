# Felix Framework - Quick Start Guide

Get up and running with Felix in under 10 minutes!

## What is Felix?

Felix is a multi-agent AI framework that uses **helical geometry** for adaptive agent progression. It enables multiple AI agents to work together on complex tasks, with built-in knowledge management and multi-provider LLM support.

**Key Features:**
- 🔄 Dynamic agent spawning based on task complexity
- 🤖 Support for LM Studio (local), Anthropic Claude, and Google Gemini
- 🧠 Autonomous knowledge brain with document learning
- 🎯 Hub-spoke communication (O(N) vs O(N²))
- 📊 Built-in GUI and command-line interface

---

## Installation (5 minutes)

### Prerequisites
- Python 3.9 or higher
- (Optional) LM Studio running locally for free LLM inference

### Step 1: Clone and Setup

```bash
# Clone the repository
cd /path/to/felix

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Initialize Databases

```bash
# Felix creates databases automatically on first run
# But you can initialize them explicitly:
python -c "from src.migration.version_manager import VersionManager; VersionManager().initialize_databases()"
```

---

## First Run (5 minutes)

### Option A: Using the GUI (Easiest)

```bash
# Make sure LM Studio is running (or configure cloud provider)
python -m src.gui
```

**Steps in GUI:**
1. Click "Start Felix System" on Dashboard tab
2. Go to "Workflows" tab
3. Enter a task: "Explain quantum computing for beginners"
4. Click "Run Workflow"
5. Watch agents work and see results!

### Option B: Using Python Script

Create `my_first_workflow.py`:

```python
from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory
from src.llm.router_adapter import create_router_adapter
from src.workflows.felix_workflow import execute_linear_workflow_optimized

# Initialize components
helix = HelixGeometry(top_radius=3.0, bottom_radius=0.5, height=8.0, turns=2)
llm_client = create_router_adapter()  # Uses config/llm.yaml

central_post = CentralPost(helix)
agent_factory = AgentFactory(central_post, helix, llm_client)

# Run a workflow
task = "Explain quantum computing for beginners"
result = execute_linear_workflow_optimized(
    task_input=task,
    felix_system=type('obj', (object,), {
        'helix': helix,
        'central_post': central_post,
        'agent_factory': agent_factory,
        'lm_client': llm_client,
        'config': type('obj', (object,), {
            'workflow_max_steps': 10,
            'enable_web_search': False
        })()
    })(),
    max_steps_override=5
)

print("\n=== RESULT ===")
print(result.get("centralpost_synthesis", {}).get("synthesis_content", "No result"))
```

Run it:

```bash
python my_first_workflow.py
```

### Option C: Using the Command Line (Fastest)

Felix includes a built-in CLI for quick workflows without writing code:

```bash
# Basic workflow
python -m src.cli run "Explain quantum computing for beginners"

# Save to file
python -m src.cli run "Write a Python sorting algorithm" --output algorithm.md

# Enable web search
python -m src.cli run "Latest AI trends in 2024" --web-search

# Check system status
python -m src.cli status

# Test LLM connection
python -m src.cli test-connection
```

**CLI Advantages:**
- ⚡ No code writing required
- 🚀 Perfect for CI/CD pipelines
- 🖥️  Works on headless servers
- 📁 Direct file output (.txt, .md, .json)

---

## Configuration

### Using LM Studio (Local, Free)

Default configuration works out of the box:

```yaml
# config/llm.yaml (default)
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "local-model"
```

**Requirements:**
1. Download [LM Studio](https://lmstudio.ai/)
2. Load a model (e.g., Mistral 7B, Llama 2)
3. Start local server (port 1234)

### Using Anthropic Claude

Edit `config/llm.yaml`:

```yaml
primary:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"  # Set environment variable
  model: "claude-3-5-sonnet-20241022"
  timeout: 120

fallbacks:
  - type: "lm_studio"  # Fallback to local if Claude fails
    base_url: "http://localhost:1234/v1"
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### Using Google Gemini

Edit `config/llm.yaml`:

```yaml
primary:
  type: "gemini"
  api_key: "${GEMINI_API_KEY}"
  model: "gemini-1.5-flash-latest"  # Fast and cheap!
  timeout: 120
```

Set your API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

---

## Common Tasks

### Task 1: Run a Simple Query

```python
from src.gui.felix_system import FelixSystem

# Create system
system = FelixSystem()
system.start_system()

# Run workflow
result = system.run_workflow("What is the capital of France?")
print(result)
```

**Expected:** 1-2 agents spawn, quick answer from web search.

### Task 2: Run a Complex Analysis

```python
result = system.run_workflow(
    "Compare and contrast the architectural patterns "
    "of microservices vs monolithic applications"
)
```

**Expected:** 5-8 agents spawn (Research, Analysis, Critic), detailed response.

### Task 3: Enable Knowledge Brain

Edit `config/llm.yaml`:

```yaml
knowledge_brain:
  enable_knowledge_brain: true
  knowledge_watch_dirs: ["./documents"]
  knowledge_auto_augment: true
```

Create `documents/` folder and add PDFs/text files. Felix will automatically:
- Read and comprehend documents
- Build knowledge graph
- Inject relevant knowledge into workflows

### Task 4: Use Web Search

```python
result = system.run_workflow(
    "What are the latest developments in AI as of October 2025?",
    enable_web_search=True
)
```

Felix will search DuckDuckGo and incorporate current information.

---

## Understanding Agent Spawning

Felix intelligently spawns agents based on task complexity:

| Query Type | Complexity | Agents Spawned | Example |
|------------|------------|----------------|---------|
| Simple factual | 0.1-0.3 | 1-2 agents | "What time is it?" |
| Medium analysis | 0.4-0.6 | 3-5 agents | "Explain machine learning" |
| Complex synthesis | 0.7-1.0 | 6-10 agents | "Design a distributed system" |

**Recent improvements:**
- ✅ Reduced spawning from 20 → 2-3 for simple queries
- ✅ Added 30-second cooldown between spawns
- ✅ Smart complexity detection

---

## Troubleshooting

### "Failed to connect to LLM provider"

**LM Studio:**
- Ensure LM Studio is running
- Ensure a model is loaded
- Check port 1234 is accessible

**Cloud providers:**
- Verify API key is set correctly
- Check `config/llm.yaml` syntax
- Try fallback to LM Studio

### "Database locked"

```bash
# Check for other Felix processes
ps aux | grep felix

# Kill if necessary
pkill -f felix

# Or just restart
```

### "Too many agents spawning"

This should be fixed in the latest version. If still happening:

1. Check `src/agents/dynamic_spawning.py` line 571 (should be multiplier of 3, not 10)
2. Verify config: `max_agents: 10` in your workflow
3. Update to latest code

### "Import errors"

```bash
# Make sure you're in virtual environment
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.9+
```

---

## Next Steps

### Learn More
- Read [USER_MANUAL.md](USER_MANUAL.md) for comprehensive documentation
- Check [CLAUDE.md](CLAUDE.md) for development guide
- Explore [examples/](examples/) for code samples

### Try Examples
```bash
# Simple workflow
python examples/basic/hello_felix.py

# Multi-agent coordination
python examples/basic/multi_agent_team.py

# Knowledge brain integration
python examples/advanced/knowledge_brain.py
```

### Run Tests
```bash
# Verify your installation
pytest tests/unit -v

# Run with coverage
pytest tests/unit --cov=src
```

### Join Community
- GitHub Issues: Report bugs or request features
- Discussions: Ask questions and share use cases

---

## Quick Reference

### Start GUI
```bash
python -m src.gui
```

### Run Workflow (CLI)
```bash
python -c "from src.workflows import felix_workflow; print(felix_workflow.execute_simple('Your task here'))"
```

### Check System Status
```python
from src.gui.felix_system import FelixSystem
system = FelixSystem()
print(system.get_status())
```

### View Databases
```bash
# Knowledge entries
sqlite3 felix_knowledge.db "SELECT COUNT(*) FROM knowledge_entries;"

# Workflow history
sqlite3 felix_workflow_history.db "SELECT COUNT(*) FROM workflow_history;"
```

---

## Cost Estimates (Cloud Providers)

**Anthropic Claude 3.5 Sonnet:**
- $3.00 per 1M input tokens
- $15.00 per 1M output tokens
- Typical workflow: ~10,000 tokens = $0.10

**Google Gemini 1.5 Flash:**
- $0.075 per 1M input tokens
- $0.30 per 1M output tokens
- Typical workflow: ~10,000 tokens = $0.002 (very cheap!)

**LM Studio (Local):**
- Free! But requires decent hardware (8GB+ RAM recommended)

---

## Success!

You're now ready to use Felix! Try these workflows:

```
"Explain quantum entanglement to a 10-year-old"
"Write a haiku about artificial intelligence"
"Compare the pros and cons of functional vs object-oriented programming"
"Design a REST API for a todo application"
```

**Happy building! 🚀**
