# Felix REST API - Quick Start Guide

## Overview

The Felix REST API provides programmatic access to the Felix multi-agent AI framework. Execute workflows, manage agents, and access knowledge brain functionality via HTTP/WebSocket.

## Features

- **Workflow Execution**: Create and monitor multi-agent workflows
- **Real-time Streaming**: WebSocket support for live workflow updates ✅
- **Agent Management**: Spawn, configure, and monitor agents ✅
- **Knowledge Brain**: Document ingestion and semantic search (coming soon)
- **Authentication**: API key-based authentication
- **Auto-generated Docs**: Interactive API explorer at `/docs`

## Installation

### 1. Install Dependencies

```bash
# Install core Felix dependencies (if not already installed)
pip install -r requirements.txt

# Install API-specific dependencies
pip install -r requirements-api.txt
```

### 2. Set Environment Variables

```bash
# Required: API authentication (optional for development)
export FELIX_API_KEY="your-secret-api-key-here"

# Optional: Felix configuration
export FELIX_LM_HOST="127.0.0.1"      # LM Studio host
export FELIX_LM_PORT="1234"            # LM Studio port
export FELIX_MAX_AGENTS="10"           # Maximum concurrent agents
export FELIX_ENABLE_KNOWLEDGE_BRAIN="false"  # Enable knowledge brain
```

### 3. Start LM Studio (Optional)

If using local LLM via LM Studio:

```bash
# Start LM Studio server on port 1234
# Load a model (e.g., Mistral 7B, Llama 2)
```

### 4. Start API Server

```bash
# Development mode with auto-reload
python3 -m uvicorn src.api.main:app --reload --port 8000

# Production mode
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Server will start at: `http://localhost:8000`

## Quick Start

### 1. Check API Health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "api_version": "1.0.0",
  "timestamp": 1698672000.0,
  "felix_initialized": false
}
```

### 2. Start Felix System

```bash
curl -X POST http://localhost:8000/api/v1/system/start \
  -H "Authorization: Bearer your-secret-api-key-here"
```

Response:
```json
{
  "status": "running",
  "felix_version": "0.9.0",
  "api_version": "1.0.0",
  "uptime_seconds": 0.0,
  "active_workflows": 0,
  "active_agents": 0,
  "llm_provider": "lm_studio",
  "knowledge_brain_enabled": false
}
```

### 3. Create a Workflow

```bash
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Explain quantum computing in simple terms",
    "max_steps": 10
  }'
```

Response (202 Accepted):
```json
{
  "workflow_id": "wf_abc123def456",
  "status": "pending",
  "task": "Explain quantum computing in simple terms",
  "created_at": "2025-10-30T10:00:00Z",
  "completed_at": null,
  "agents_spawned": [],
  "synthesis": null,
  "performance_metrics": null,
  "error": null
}
```

### 4. Check Workflow Status

```bash
curl http://localhost:8000/api/v1/workflows/wf_abc123def456 \
  -H "Authorization: Bearer your-secret-api-key-here"
```

Response (while running):
```json
{
  "workflow_id": "wf_abc123def456",
  "status": "running",
  "task": "Explain quantum computing in simple terms",
  "created_at": "2025-10-30T10:00:00Z",
  "completed_at": null,
  "agents_spawned": [
    {
      "agent_id": "research_001",
      "agent_type": "research",
      "spawn_time": 0.1,
      "confidence": null
    }
  ],
  "synthesis": null
}
```

Response (when completed):
```json
{
  "workflow_id": "wf_abc123def456",
  "status": "completed",
  "task": "Explain quantum computing in simple terms",
  "created_at": "2025-10-30T10:00:00Z",
  "completed_at": "2025-10-30T10:02:30Z",
  "agents_spawned": [
    {
      "agent_id": "research_001",
      "agent_type": "research",
      "spawn_time": 0.1,
      "confidence": 0.85
    },
    {
      "agent_id": "analysis_001",
      "agent_type": "analysis",
      "spawn_time": 0.4,
      "confidence": 0.82
    }
  ],
  "synthesis": {
    "content": "Quantum computing is a revolutionary computing paradigm that leverages quantum mechanical phenomena...",
    "confidence": 0.87,
    "agents_synthesized": 3,
    "token_count": 452
  },
  "performance_metrics": {
    "total_duration": 150.5,
    "llm_calls": 8,
    "total_tokens": 3420
  }
}
```

### 5. List Workflows

```bash
# List all workflows
curl http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer your-secret-api-key-here"

# Filter by status
curl "http://localhost:8000/api/v1/workflows?status_filter=completed&limit=10" \
  -H "Authorization: Bearer your-secret-api-key-here"
```

Response:
```json
{
  "workflows": [
    {
      "workflow_id": "wf_abc123def456",
      "status": "completed",
      ...
    }
  ],
  "total": 42,
  "page": 0,
  "page_size": 10
}
```

## Interactive API Documentation

Felix REST API provides auto-generated interactive documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

The interactive docs allow you to:
- Browse all available endpoints
- See request/response schemas
- Try out API calls directly in your browser
- View example requests and responses

## Authentication

### API Key Authentication

The API uses Bearer token authentication with API keys.

#### Setting API Key

```bash
# Set via environment variable (recommended)
export FELIX_API_KEY="your-secret-key-123"

# Or create a .env file
echo "FELIX_API_KEY=your-secret-key-123" > .env
```

#### Using API Key

Include the API key in the `Authorization` header:

```bash
curl http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer your-secret-key-123"
```

#### Development Mode (No Auth)

If `FELIX_API_KEY` is not set, the API runs in development mode without authentication:

```bash
# No API key required
curl http://localhost:8000/api/v1/system/status
```

**Warning**: Never run production deployments without authentication!

## Python Client Example

```python
import requests
import time

# Configuration
API_URL = "http://localhost:8000"
API_KEY = "your-secret-key-123"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 1. Start Felix system
response = requests.post(f"{API_URL}/api/v1/system/start", headers=headers)
print("System started:", response.json())

# 2. Create workflow
workflow_data = {
    "task": "Analyze the pros and cons of renewable energy",
    "max_steps": 15
}

response = requests.post(
    f"{API_URL}/api/v1/workflows",
    headers=headers,
    json=workflow_data
)

workflow = response.json()
workflow_id = workflow["workflow_id"]
print(f"Workflow created: {workflow_id}")

# 3. Poll for completion
while True:
    response = requests.get(
        f"{API_URL}/api/v1/workflows/{workflow_id}",
        headers=headers
    )

    workflow = response.json()
    status = workflow["status"]

    print(f"Status: {status}")

    if status in ["completed", "failed", "cancelled"]:
        break

    time.sleep(5)  # Poll every 5 seconds

# 4. Get results
if workflow["status"] == "completed":
    synthesis = workflow["synthesis"]
    print("\nSynthesis:")
    print(synthesis["content"])
    print(f"\nConfidence: {synthesis['confidence']:.2%}")
else:
    print(f"\nWorkflow failed: {workflow.get('error')}")
```

## Common Use Cases

### 1. Research Task

```bash
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer ${FELIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Research the latest developments in artificial general intelligence",
    "max_steps": 20
  }'
```

### 2. Analysis Task

```bash
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer ${FELIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Analyze the economic impact of climate change on developing nations",
    "max_steps": 15
  }'
```

### 3. Simple Question

```bash
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer ${FELIX_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "What is the current date and time?",
    "max_steps": 5
  }'
```

## WebSocket Streaming

Felix API supports WebSocket connections for real-time workflow updates. Instead of polling, you can receive push notifications as events occur.

### WebSocket Endpoints

| Endpoint | Description |
|----------|-------------|
| `WS /api/v1/ws/workflows/{workflow_id}` | Stream events for specific workflow |
| `WS /api/v1/ws/system/events` | Stream system-wide events |

### Event Types

WebSocket connections receive JSON events with the following structure:

```json
{
  "type": "event_type",
  "workflow_id": "wf_abc123",
  "timestamp": "2025-10-30T10:00:00Z",
  ... additional fields
}
```

**Available Event Types:**

1. **connected** - Initial connection confirmation
   ```json
   {
     "type": "connected",
     "workflow_id": "wf_abc123",
     "message": "Connected to workflow stream"
   }
   ```

2. **workflow_status** - Workflow status change
   ```json
   {
     "type": "workflow_status",
     "workflow_id": "wf_abc123",
     "status": "running"
   }
   ```

3. **agent_spawned** - New agent created
   ```json
   {
     "type": "agent_spawned",
     "workflow_id": "wf_abc123",
     "agent_id": "research_001",
     "agent_type": "research",
     "spawn_time": 0.1
   }
   ```

4. **synthesis_started** - Synthesis phase beginning
   ```json
   {
     "type": "synthesis_started",
     "workflow_id": "wf_abc123",
     "agent_count": 3
   }
   ```

5. **workflow_complete** - Workflow finished successfully
   ```json
   {
     "type": "workflow_complete",
     "workflow_id": "wf_abc123",
     "status": "completed",
     "synthesis": {
       "content": "Final synthesis...",
       "confidence": 0.87,
       "agents_synthesized": 3
     }
   }
   ```

6. **workflow_error** - Workflow failed
   ```json
   {
     "type": "workflow_error",
     "workflow_id": "wf_abc123",
     "error": "Error message"
   }
   ```

7. **ping** - Keepalive ping (respond with "pong")

### Python WebSocket Client

```python
import asyncio
import websockets
import json

async def stream_workflow(workflow_id, api_key=None):
    # Build WebSocket URL with optional API key
    ws_uri = f"ws://localhost:8000/api/v1/ws/workflows/{workflow_id}"
    if api_key:
        ws_uri += f"?api_key={api_key}"

    async with websockets.connect(ws_uri) as websocket:
        print(f"Connected to workflow {workflow_id}")

        async for message in websocket:
            # Handle keepalive pings
            if message == "pong":
                continue

            # Parse event
            event = json.loads(message)
            event_type = event.get('type')

            if event_type == 'workflow_status':
                print(f"Status: {event['status']}")

            elif event_type == 'agent_spawned':
                print(f"Agent spawned: {event['agent_id']}")

            elif event_type == 'synthesis_started':
                print(f"Synthesis starting with {event['agent_count']} agents")

            elif event_type == 'workflow_complete':
                synthesis = event['synthesis']
                print(f"\nCompleted! Confidence: {synthesis['confidence']:.2%}")
                print(f"Result: {synthesis['content']}")
                break

            elif event_type == 'workflow_error':
                print(f"Error: {event['error']}")
                break

            elif event_type == 'ping':
                # Respond to server keepalive
                await websocket.send('pong')

# Usage
asyncio.run(stream_workflow("wf_abc123", api_key="your-key"))
```

### JavaScript/Browser WebSocket Client

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/workflows/wf_abc123?api_key=your-key');

ws.onopen = () => {
  console.log('Connected to workflow stream');
};

ws.onmessage = (event) => {
  if (event.data === 'pong') return;

  const data = JSON.parse(event.data);

  switch (data.type) {
    case 'workflow_status':
      console.log('Status:', data.status);
      break;

    case 'agent_spawned':
      console.log('Agent spawned:', data.agent_id);
      break;

    case 'synthesis_started':
      console.log('Synthesis starting with', data.agent_count, 'agents');
      break;

    case 'workflow_complete':
      console.log('Completed!', data.synthesis.content);
      ws.close();
      break;

    case 'workflow_error':
      console.error('Error:', data.error);
      ws.close();
      break;

    case 'ping':
      ws.send('pong');
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected');
};
```

### Example Clients

Two complete example clients are provided:

1. **Python Client**: `examples/api_examples/websocket_client_example.py`
   - Full-featured async client with event handlers
   - Usage: `python examples/api_examples/websocket_client_example.py`

2. **Browser Client**: `examples/api_examples/websocket_client.html`
   - Interactive HTML/JavaScript client
   - Usage: Open in browser after starting API server

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FELIX_API_KEY` | None | API key for authentication (optional) |
| `FELIX_LM_HOST` | 127.0.0.1 | LM Studio host |
| `FELIX_LM_PORT` | 1234 | LM Studio port |
| `FELIX_MAX_AGENTS` | 10 | Maximum concurrent agents |
| `FELIX_BASE_TOKEN_BUDGET` | 2500 | Base token budget per agent |
| `FELIX_ENABLE_KNOWLEDGE_BRAIN` | false | Enable knowledge brain system |
| `FELIX_VERBOSE_LOGGING` | false | Enable verbose LLM logging |

### Configuration File (Coming Soon)

```yaml
# config/api.yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

felix:
  lm_host: 127.0.0.1
  lm_port: 1234
  max_agents: 10

auth:
  api_key: ${FELIX_API_KEY}
  require_auth: true
```

## API Endpoints Reference

### System Management

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | API root information | No |
| GET | `/health` | Health check | No |
| POST | `/api/v1/system/start` | Start Felix system | Yes |
| POST | `/api/v1/system/stop` | Stop Felix system | Yes |
| GET | `/api/v1/system/status` | Get system status | No |

### Workflows

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/workflows` | Create workflow | Yes |
| GET | `/api/v1/workflows/{id}` | Get workflow | Yes |
| GET | `/api/v1/workflows` | List workflows | Yes |
| DELETE | `/api/v1/workflows/{id}` | Cancel workflow | Yes |

### Agents

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/api/v1/agents` | List active agents | Yes |
| POST | `/api/v1/agents` | Spawn new agent | Yes |
| GET | `/api/v1/agents/{id}` | Get agent details | Yes |
| DELETE | `/api/v1/agents/{id}` | Terminate agent | Yes |
| GET | `/api/v1/agents/plugins` | List agent plugins | Yes |
| GET | `/api/v1/agents/plugins/{type}` | Get plugin metadata | Yes |
| POST | `/api/v1/agents/plugins/reload` | Hot-reload plugins | Yes |
| GET | `/api/v1/agents/plugins/suitable` | Get suitable agents for task | Yes |

### WebSocket

| Endpoint | Description | Auth Required |
|----------|-------------|---------------|
| WS `/api/v1/ws/workflows/{id}` | Stream workflow events | Optional (query param) |
| WS `/api/v1/ws/system/events` | Stream system events | Optional (query param) |

### Knowledge Brain

**Note:** Requires `FELIX_ENABLE_KNOWLEDGE_BRAIN=true` environment variable.

**Documents (5 endpoints)**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/knowledge/documents/ingest` | Ingest single document | Yes |
| POST | `/api/v1/knowledge/documents/batch` | Batch process directory | Yes |
| GET | `/api/v1/knowledge/documents` | List documents | Yes |
| GET | `/api/v1/knowledge/documents/{id}` | Get document details | Yes |
| DELETE | `/api/v1/knowledge/documents/{id}` | Delete document | Yes |

**Search (2 endpoints)**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/knowledge/search` | Semantic search | Yes |
| POST | `/api/v1/knowledge/search/augment` | Get augmented context | Yes |

**Knowledge Graph (3 endpoints)**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/knowledge/graph/build` | Build knowledge graph | Yes |
| POST | `/api/v1/knowledge/graph/relationships` | Get concept relationships | Yes |
| GET | `/api/v1/knowledge/graph/statistics` | Graph statistics | Yes |

**Daemon Control (4 endpoints)**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/knowledge/daemon/start` | Start daemon | Yes |
| POST | `/api/v1/knowledge/daemon/stop` | Stop daemon | Yes |
| GET | `/api/v1/knowledge/daemon/status` | Get daemon status | Yes |
| PUT | `/api/v1/knowledge/daemon/watch-dirs` | Update watched directories | Yes |

**Concepts (3 endpoints)**
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/api/v1/knowledge/concepts` | List concepts | Yes |
| GET | `/api/v1/knowledge/concepts/{id}` | Get concept details | Yes |
| GET | `/api/v1/knowledge/concepts/{id}/related` | Get related concepts | Yes |

**Quick Example:**
```bash
# Enable Knowledge Brain
export FELIX_ENABLE_KNOWLEDGE_BRAIN=true

# Ingest a document
curl -X POST http://localhost:8000/api/v1/knowledge/documents/ingest \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/document.pdf"}'

# Search knowledge
curl -X POST http://localhost:8000/api/v1/knowledge/search \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 10}'
```

For complete Knowledge Brain API documentation, see [KNOWLEDGE_BRAIN_API.md](KNOWLEDGE_BRAIN_API.md).

### Coming Soon

- **Memory**: Workflow history, task patterns
- **Commands**: System command execution with approvals

## Troubleshooting

### API Won't Start

**Problem**: `RuntimeError: Failed to start Felix system`

**Solutions**:
1. Check LM Studio is running on configured port
2. Verify environment variables are set correctly
3. Check logs: `uvicorn src.api.main:app --log-level debug`

### Authentication Errors

**Problem**: `401 Unauthorized - Invalid API key`

**Solutions**:
1. Verify `FELIX_API_KEY` environment variable is set
2. Check Authorization header format: `Bearer <key>`
3. For development, unset `FELIX_API_KEY` to disable auth

### 500 Error on System Start

**Problem**: `500 Internal Server Error` when calling `POST /api/v1/system/start`

**Cause**: This typically occurs when:
1. LM Studio is not running or not accessible
2. No model is loaded in LM Studio
3. LM Studio is running on a different port than configured
4. System status validation fails due to missing fields

**Solutions**:

1. **Verify LM Studio is running and accessible**:

   ```bash
   # Check if LM Studio is listening on port 1234
   curl http://localhost:1234/v1/models

   # Should return a list of loaded models
   ```

2. **Load a model in LM Studio**:
   - Open LM Studio
   - Navigate to the "Local Server" tab
   - Select and load a model (e.g., Mistral 7B, Llama 2)
   - Ensure "Start Server" is clicked
   - Verify the port matches your configuration (default: 1234)

3. **Check environment variables**:

   ```bash
   # Verify LM Studio host/port configuration
   echo $FELIX_LM_HOST  # Should be 127.0.0.1 or localhost
   echo $FELIX_LM_PORT  # Should match LM Studio port (default: 1234)
   ```

4. **Configure alternative LLM provider**:

   If not using LM Studio, create `config/llm.yaml` with your provider:

   ```yaml
   providers:
     - name: openai
       type: openai
       api_key: ${OPENAI_API_KEY}
       models:
         - gpt-4
         - gpt-3.5-turbo

   routing:
     default_provider: openai
   ```

5. **Check detailed error logs**:

   ```bash
   # Run API with debug logging
   python3 -m uvicorn src.api.main:app --reload --port 8000 --log-level debug
   ```

**Expected behavior**: When LM Studio is running with a loaded model, the `/start` endpoint should return:

```json
{
  "status": "running",
  "felix_version": "0.9.0",
  "api_version": "1.0.0",
  "uptime_seconds": 0.0,
  "active_workflows": 0,
  "active_agents": 0,
  "llm_provider": "lm_studio",
  "knowledge_brain_enabled": false
}
```

### Workflow Stuck in "Running"

**Problem**: Workflow status stuck at "running"

**Solutions**:
1. Check Felix system logs for errors
2. Verify LLM is responding (check LM Studio)
3. Check max_steps isn't too low for complex tasks
4. Cancel and retry: `DELETE /api/v1/workflows/{id}`

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'fastapi'`

**Cause**: This error can occur even when dependencies are installed if using the bare `uvicorn` command on Linux systems where only `python3` exists (no `python` symlink).

**Solutions**:

1. **Use python3 -m uvicorn (Recommended)**:

   ```bash
   python3 -m uvicorn src.api.main:app --reload --port 8000
   ```

   This explicitly tells Python 3 to run uvicorn, ensuring it uses the correct virtual environment.

2. **Verify dependencies are installed**:

   ```bash
   # Ensure your virtual environment is activated
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows

   # Install/reinstall API dependencies
   pip install -r requirements-api.txt

   # Verify fastapi is installed
   python3 -c "import fastapi; print(fastapi.__version__)"
   ```

3. **Create python symlink in venv (Alternative)**:

   ```bash
   cd .venv/bin
   ln -s python3 python
   cd ../..

   # Now bare uvicorn will work
   uvicorn src.api.main:app --reload --port 8000
   ```

**Why this happens**: On modern Linux distributions, the `python` command often doesn't exist by default - only `python3`. When uvicorn spawns subprocesses, it may try to use `python` which fails. Using `python3 -m uvicorn` ensures the correct interpreter is used.

## Performance Tips

### 1. Use Async Clients

For Python, use `httpx` for async requests:

```python
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/workflows",
            headers={"Authorization": "Bearer key"},
            json={"task": "Example task"}
        )
        return response.json()

asyncio.run(main())
```

### 2. Poll Efficiently

Don't poll too frequently for workflow status:

```python
# Good: Poll every 5-10 seconds
time.sleep(5)

# Bad: Poll every 0.1 seconds
time.sleep(0.1)  # Wastes resources
```

### 3. Use WebSocket for Real-time Updates

WebSocket provides push-based updates instead of polling:

```python
import asyncio
import websockets
import json

async def stream_workflow(workflow_id, api_key=None):
    # Build WebSocket URL
    ws_uri = f"ws://localhost:8000/api/v1/ws/workflows/{workflow_id}"
    if api_key:
        ws_uri += f"?api_key={api_key}"

    async with websockets.connect(ws_uri) as websocket:
        async for message in websocket:
            event = json.loads(message)

            if event['type'] == 'workflow_complete':
                print("Synthesis:", event['synthesis']['content'])
                break
            elif event['type'] == 'agent_spawned':
                print(f"Agent spawned: {event['agent_id']}")
            elif event['type'] == 'workflow_error':
                print(f"Error: {event['error']}")
                break

asyncio.run(stream_workflow("wf_abc123"))
```

## Next Steps

Completed features you can use now:
- ✅ **Agent Management**: Spawn custom agents, manage agent plugins
- ✅ **WebSocket Streaming**: Real-time workflow progress updates

Coming soon:
- **Knowledge Brain**: Upload documents, semantic search, knowledge graphs
- **System Commands**: Execute system commands with approval workflow
- **Memory API**: Query workflow history and task patterns

## Support

- **Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **GitHub Issues**: [https://github.com/yourusername/felix/issues](https://github.com/yourusername/felix/issues)
- **Examples**: `examples/api_examples/` directory

## License

MIT License - See LICENSE file for details
