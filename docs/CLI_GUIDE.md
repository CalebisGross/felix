# Felix CLI Guide

Complete reference for the Felix command-line interface.

## Overview

The Felix CLI provides full framework functionality without requiring the GUI, making it ideal for:
- CI/CD integration and automated workflows
- Remote server deployments (headless/no display)
- Scripting and batch processing
- Quick testing and experimentation
- Integration with other tools

## Installation

The CLI is included with Felix. No additional installation required.

```bash
# Ensure Felix is installed
cd /path/to/felix
source .venv/bin/activate

# Verify CLI is available
python -m src.cli --version
```

## Commands

### `run` - Execute Workflows

Run a Felix workflow from the command line.

**Syntax:**
```bash
python -m src.cli run "TASK_DESCRIPTION" [OPTIONS]
```

**Options:**
- `--output, -o FILE` - Save results to file (.txt, .md, or .json)
- `--max-steps N` - Maximum workflow steps (default: 10)
- `--web-search` - Enable web search integration
- `--config PATH` - LLM config file (default: config/llm.yaml)
- `--verbose, -v` - Enable verbose output with stack traces

**Examples:**

```bash
# Basic workflow
python -m src.cli run "Explain quantum computing"

# Save to markdown file
python -m src.cli run "Design a REST API" --output design.md

# Complex workflow with web search
python -m src.cli run "Latest AI trends in 2024" \
  --max-steps 15 \
  --web-search \
  --output trends.md

# Save as JSON for parsing
python -m src.cli run "Analyze this algorithm" --output result.json

# Use custom LLM config
python -m src.cli run "Summarize machine learning" \
  --config config/custom_llm.yaml
```

**Output:**
- Displays workflow progress with agent information
- Shows final synthesis result
- Prints confidence score and agent count
- Saves to file if `--output` specified

**Exit Codes:**
- `0` - Success
- `1` - Workflow or initialization error
- `130` - Interrupted by user (Ctrl+C)

---

### `status` - System Status

Check Felix system status including LLM providers, databases, and knowledge statistics.

**Syntax:**
```bash
python -m src.cli status [OPTIONS]
```

**Options:**
- `--config PATH` - LLM config file (default: config/llm.yaml)

**Example:**
```bash
python -m src.cli status
```

**Output Sections:**

1. **LLM Providers** - Connection status for all configured providers
   - ✓ Provider connected
   - ✗ Provider failed
   - Router statistics (if available)

2. **Databases** - Existence and size of Felix databases
   - felix_knowledge.db
   - felix_workflow_history.db
   - felix_memory.db
   - felix_task_memory.db
   - felix_system_actions.db

3. **Knowledge** - Knowledge brain statistics
   - Number of entries
   - Number of documents

**Example Output:**
```
Felix System Status
============================================================

🔌 LLM Providers:
  ✓ lm_studio
  ✓ anthropic
  ✗ gemini

📊 Router Statistics:
  Total requests: 42
  Success rate: 95.2%

💾 Databases:
  ✓ felix_knowledge.db (2.3 MB)
  ✓ felix_workflow_history.db (1.1 MB)
  ✓ felix_memory.db (0.5 MB)
  ✓ felix_task_memory.db (0.3 MB)
  ✓ felix_system_actions.db (0.2 MB)

📚 Knowledge:
  Entries: 156
  Documents: 12

============================================================
```

---

### `test-connection` - Test LLM Connection

Test connectivity to LLM providers and check health status.

**Syntax:**
```bash
python -m src.cli test-connection [OPTIONS]
```

**Options:**
- `--config PATH` - LLM config file (default: config/llm.yaml)
- `--verbose, -v` - Show detailed error information

**Example:**
```bash
python -m src.cli test-connection
```

**Output:**
- Shows primary provider
- Tests primary provider connection
- Tests all configured providers
- Displays success/failure for each

**Example Output:**
```
Testing LLM Connection
============================================================
Initializing router...
Primary provider: lm_studio

Testing primary provider...
✓ Primary provider connected

Testing all providers...
  ✓ lm_studio
  ✓ anthropic
  ✗ gemini

============================================================
```

---

### `gui` - Launch GUI

Launch the Felix GUI from the command line.

**Syntax:**
```bash
python -m src.cli gui
```

**Notes:**
- Requires display/X11 (not available on headless servers)
- Equivalent to `python -m src.gui`

**Example:**
```bash
python -m src.cli gui
```

---

### `init` - Initialize Databases

Initialize or reset Felix databases.

**Syntax:**
```bash
python -m src.cli init
```

**Warning:** This will run database migrations and initialize schema. Use with caution on existing databases.

**Example:**
```bash
python -m src.cli init
```

**Output:**
```
Initializing Felix databases...
✓ Databases initialized
```

---

## Configuration

The CLI uses the same configuration files as the rest of Felix.

### LLM Configuration

Default location: `config/llm.yaml`

```yaml
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
  model: "local-model"

fallbacks:
  - type: "anthropic"
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-3-5-sonnet-20241022"
```

Specify custom config:
```bash
python -m src.cli run "Task" --config path/to/llm.yaml
```

### Environment Variables

Set API keys via environment variables:

```bash
export ANTHROPIC_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"
```

---

## Use Cases

### CI/CD Integration

Run Felix workflows in GitHub Actions, GitLab CI, etc:

```yaml
# .github/workflows/felix.yml
name: Felix Workflow
on: [push]
jobs:
  run-felix:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Felix
        run: |
          source .venv/bin/activate
          python -m src.cli run "Analyze code quality" \
            --output report.md \
            --max-steps 10
      - name: Upload report
        uses: actions/upload-artifact@v2
        with:
          name: felix-report
          path: report.md
```

### Batch Processing

Process multiple tasks in a loop:

```bash
#!/bin/bash
# process_tasks.sh

TASKS=(
  "Explain machine learning"
  "Summarize deep learning"
  "Describe neural networks"
)

for i in "${!TASKS[@]}"; do
  echo "Processing task $((i+1))/${#TASKS[@]}"
  python -m src.cli run "${TASKS[$i]}" \
    --output "output_$i.md" \
    --max-steps 8
done
```

### Remote Server Deployment

Run on headless servers:

```bash
# SSH into server
ssh user@server

# Run Felix workflow
cd /path/to/felix
source .venv/bin/activate
python -m src.cli run "Server analysis task" \
  --output /var/log/felix/analysis.md
```

### Cron Jobs

Schedule periodic Felix tasks:

```cron
# Run Felix analysis daily at 2 AM
0 2 * * * cd /path/to/felix && source .venv/bin/activate && python -m src.cli run "Daily summary" --output /logs/daily_$(date +\%Y\%m\%d).md
```

---

## Troubleshooting

### "No module named 'src'"

Ensure you're in the Felix project directory:
```bash
cd /path/to/felix
python -m src.cli run "Task"
```

### "Could not initialize router"

1. Check LLM configuration exists:
   ```bash
   ls config/llm.yaml
   ```

2. Test provider connectivity:
   ```bash
   python -m src.cli test-connection --verbose
   ```

3. Verify provider is running (for LM Studio):
   - Open LM Studio
   - Load a model
   - Start server on port 1234

### "Initialization failed"

Database initialization issue. Try:
```bash
# Check database permissions
ls -la *.db

# Reinitialize
python -m src.cli init
```

### Verbose Mode

Use `--verbose` for detailed error information:
```bash
python -m src.cli run "Task" --verbose
python -m src.cli test-connection --verbose
```

---

## Tips and Best Practices

1. **Start Small**: Begin with short max-steps (5-8) for testing
2. **Use JSON Output**: Parse results programmatically with `.json` output
3. **Check Status First**: Run `status` before workflows to verify system health
4. **Web Search**: Only enable when needed (adds latency and complexity)
5. **Verbose Mode**: Use for debugging, disable for production
6. **Config Files**: Use separate configs for different environments
7. **Output Directory**: Create dedicated directory for CLI outputs
8. **Error Handling**: Wrap CLI calls in try/catch for automation

Example wrapper script:
```bash
#!/bin/bash
set -e  # Exit on error

OUTPUT_DIR="./felix_outputs"
mkdir -p "$OUTPUT_DIR"

if python -m src.cli run "$1" --output "$OUTPUT_DIR/result.md" --max-steps 10; then
    echo "✓ Workflow succeeded"
    exit 0
else
    echo "✗ Workflow failed"
    exit 1
fi
```

---

## Comparison: CLI vs GUI vs API

| Feature | CLI | GUI | REST API |
|---------|-----|-----|----------|
| Workflow execution | ✓ | ✓ | ✓ |
| No display required | ✓ | ✗ | ✓ |
| Automation friendly | ✓ | ✗ | ✓ |
| Interactive | ✗ | ✓ | ✗ |
| Real-time monitoring | Limited | ✓ | ✓ (WebSocket) |
| Knowledge Brain UI | ✗ | ✓ | ✓ |
| System status | ✓ | ✓ | ✓ |
| Approvals | ✗ | ✓ | ✓ |
| File output | ✓ | ✓ | ✓ |

**Choose CLI when:**
- Running on servers without display
- Automating workflows
- Scripting/batch processing
- Quick one-off tasks

**Choose GUI when:**
- Interactive exploration
- Visual monitoring
- Approval workflows
- Knowledge Brain management

**Choose API when:**
- Integration with other services
- Programmatic access
- WebSocket streaming needed
- Building custom interfaces

---

## See Also

- [QUICKSTART.md](../QUICKSTART.md) - Getting started guide
- [API_QUICKSTART.md](API_QUICKSTART.md) - REST API documentation
- [CLAUDE.md](../CLAUDE.md) - Complete project documentation
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration reference
