# Felix Examples

## Purpose
Practical examples demonstrating Felix's capabilities, from basic workflows to advanced multi-agent patterns, API usage, custom agent development, and external integrations.

## Directory Structure

### [basic/](basic/)
Simple examples for getting started with Felix.
- **hello_felix.py**: Minimal workflow (< 50 lines)
- **multi_provider.py**: Multi-LLM provider configuration with fallback
- **knowledge_brain_demo.py**: Basic knowledge brain usage

### [advanced/](advanced/)
Complex patterns and advanced features.
- **knowledge_brain_demo.py**: Full knowledge brain demonstration with document ingestion, comprehension, and retrieval

### [api_examples/](api_examples/)
REST API and WebSocket usage examples.
- **knowledge_brain_client.py**: Python client for knowledge brain API endpoints
- **memory_history_client.py**: Query workflow history via REST API
- **websocket_client_example.py**: Real-time workflow streaming with WebSocket
- **websocket_client.html** / **knowledge_brain_demo.html**: Browser-based clients

### [custom_agents/](custom_agents/)
Custom agent plugin development examples.
- **frontend_agent.py**: Frontend-specialized agent plugin
- **backend_agent.py**: Backend development agent plugin
- **qa_agent.py**: Quality assurance agent plugin
- **code_review_agent.py**: Code review specialist agent
- **demo_engineering_agents.py**: Demonstration of custom agent system
- **README.md**: Plugin development guide

### [integrations/](integrations/)
External system integration examples.
- Coming soon: CI/CD pipelines, Slack bots, webhook handlers

## Quick Start

### Prerequisites
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Option 1: Start LM Studio (local)
# - Download and install LM Studio
# - Load a model (7B-13B recommended)
# - Start server on port 1234

# Option 2: Use cloud provider
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Run Your First Example
```bash
# Simplest possible example
python examples/basic/hello_felix.py

# Multi-provider with fallback
python examples/basic/multi_provider.py

# Knowledge brain demonstration
python examples/advanced/knowledge_brain_demo.py

# REST API client
python examples/api_examples/memory_history_client.py

# Custom agent plugins
python examples/custom_agents/demo_engineering_agents.py
```

## Example Categories

### 1. Basic Workflows
Learn core concepts:
- Minimal Felix setup
- Creating and running workflows
- Agent spawning and synthesis
- Multi-provider configuration

### 2. Advanced Features
Explore powerful capabilities:
- Knowledge Brain with document ingestion
- Semantic retrieval with meta-learning
- Knowledge graph construction
- Web search integration
- Context building and compression

### 3. API Integration
Use Felix programmatically:
- REST API clients (Python, JavaScript)
- WebSocket streaming for real-time updates
- Workflow history queries
- Knowledge brain operations
- Agent management

### 4. Custom Development
Extend Felix:
- Create custom agent plugins
- Implement specialized agent types
- Register plugins with AgentFactory
- Build domain-specific agents

### 5. External Integration
Connect Felix to other systems:
- CI/CD pipeline integration
- Slack bot integration
- Webhook handlers
- Custom API endpoints

## Configuration

Examples use `config/llm.yaml` for provider configuration:

### Local Development (LM Studio)
```yaml
providers:
  - type: lm_studio
    base_url: http://localhost:1234/v1
    priority: 1
```

### Cloud with Fallback
```yaml
providers:
  - type: gemini
    model: gemini-1.5-pro
    priority: 1
    api_key_env: GOOGLE_API_KEY

  - type: anthropic
    model: claude-3-sonnet-20240229
    priority: 2
    api_key_env: ANTHROPIC_API_KEY

  - type: lm_studio
    base_url: http://localhost:1234/v1
    priority: 3
```

## Troubleshooting

### "Failed to connect to LLM provider"
- **LM Studio**: Ensure server is running with loaded model on port 1234
- **Cloud**: Set API keys via environment variables
- **Config**: Verify `config/llm.yaml` syntax

### "No module named 'src'"
```bash
# Must run from project root
cd /path/to/felix
python examples/basic/hello_felix.py
```

### Import Errors
```bash
# Activate virtual environment
source .venv/bin/activate

# Verify installation
python -c "from src.core.helix_geometry import HelixGeometry; print('OK')"
```

### API Connection Errors
```bash
# Check API server is running
curl http://localhost:8000/health

# Start API server
python -m uvicorn src.api.main:app --reload --port 8000
```

## Learning Path

### Beginner
1. [basic/hello_felix.py](basic/hello_felix.py) - Understand basic architecture
2. [basic/multi_provider.py](basic/multi_provider.py) - Configure providers
3. [api_examples/memory_history_client.py](api_examples/memory_history_client.py) - Query API

### Intermediate
4. [advanced/knowledge_brain_demo.py](advanced/knowledge_brain_demo.py) - Knowledge system
5. [api_examples/websocket_client_example.py](api_examples/websocket_client_example.py) - Real-time streaming
6. [custom_agents/demo_engineering_agents.py](custom_agents/demo_engineering_agents.py) - Agent plugins

### Advanced
7. [custom_agents/README.md](custom_agents/README.md) - Build custom plugins
8. [integrations/](integrations/) - External system integration

## Additional Resources
- **Documentation**: [docs/](../docs/)
- **Project Overview**: [README.md](../README.md)
- **Development Guide**: [CLAUDE.md](../CLAUDE.md)
- **Plugin API**: [docs/PLUGIN_API.md](../docs/PLUGIN_API.md)
- **API Documentation**: http://localhost:8000/docs (when API server running)
