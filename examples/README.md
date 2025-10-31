# Felix Examples

This directory contains practical examples demonstrating Felix's capabilities.

## Basic Examples

### hello_felix.py
**Simplest possible workflow**
- Minimal setup (< 50 lines)
- Single task execution
- Shows basic Felix architecture

```bash
python examples/basic/hello_felix.py
```

### multi_provider.py
**Multi-provider LLM support**
- Configure multiple providers (Anthropic, Gemini, LM Studio)
- Automatic fallback on failure
- Cost comparison

```bash
# Set API keys first
export ANTHROPIC_API_KEY="your-key"
export GEMINI_API_KEY="your-key"

python examples/basic/multi_provider.py
```

## Advanced Examples

### knowledge_brain_demo.py
**Autonomous Knowledge Brain**
- Document ingestion
- Agentic comprehension
- Knowledge graph construction
- Semantic retrieval with meta-learning

```bash
python examples/advanced/knowledge_brain_demo.py
```

## Coming Soon

- `web_search_integration.py` - DuckDuckGo search with agents
- `custom_agent.py` - Creating your own specialized agents
- `streaming_output.py` - Real-time token streaming
- `cost_optimization.py` - Token budgeting strategies

## Prerequisites

All examples require:
- Python 3.9+
- Felix dependencies installed (`pip install -r requirements.txt`)
- At least one LLM provider configured:
  - LM Studio running locally (http://localhost:1234)
  - OR Anthropic API key set
  - OR Gemini API key set

## Configuration

Examples use `config/llm.yaml` for LLM provider settings.

**Default (LM Studio):**
```yaml
primary:
  type: "lm_studio"
  base_url: "http://localhost:1234/v1"
```

**Cloud Provider:**
```yaml
primary:
  type: "anthropic"  # or "gemini"
  api_key: "${ANTHROPIC_API_KEY}"
  model: "claude-3-5-sonnet-20241022"
```

## Troubleshooting

**"Failed to connect to LLM provider"**
- Ensure LM Studio is running with a loaded model
- OR set cloud provider API keys
- Check `config/llm.yaml` is valid YAML

**Import errors**
```bash
# Make sure you're in the project root
cd /path/to/felix

# Activate virtual environment
source .venv/bin/activate

# Verify imports work
python -c "from src.core.helix_geometry import HelixGeometry; print('OK')"
```

**"No module named 'src'"**
```bash
# Add project to Python path
export PYTHONPATH="/path/to/felix:$PYTHONPATH"

# Or run from project root
cd /path/to/felix
python examples/basic/hello_felix.py
```

## Contributing Examples

Want to add an example? Great!

1. **Basic examples** - Single-file, < 100 lines, beginner-friendly
2. **Advanced examples** - Multi-file OK, demonstrates complex features
3. **Integration examples** - Shows Felix + external tools/APIs

Submit a PR with your example!
