# API Examples

## Purpose
REST API and WebSocket client examples demonstrating programmatic Felix usage from Python and JavaScript/HTML.

## Key Files

### [knowledge_brain_client.py](knowledge_brain_client.py)
Python client for Knowledge Brain API endpoints.
- Document ingestion via REST API
- Semantic search requests
- Knowledge graph queries
- Daemon control (start/stop/status)
- Concept browsing
- **Prerequisites**: API server running (`uvicorn src.api.main:app --port 8000`)
- **Run**: `python examples/api_examples/knowledge_brain_client.py`

### [memory_history_client.py](memory_history_client.py)
Query workflow history via REST API.
- List historical workflows with filters
- Retrieve detailed workflow records
- Aggregate statistics (success rate, avg confidence)
- Export workflow data
- **Prerequisites**: API server running, workflow history database exists
- **Run**: `python examples/api_examples/memory_history_client.py`

### [websocket_client_example.py](websocket_client_example.py)
Python WebSocket client for real-time workflow streaming.
- Connect to workflow WebSocket endpoint
- Receive real-time events (agent spawned, synthesis, completion)
- Handle disconnections gracefully
- Example event processing
- **Prerequisites**: API server running
- **Run**: `python examples/api_examples/websocket_client_example.py`

### [websocket_client.html](websocket_client.html)
Browser-based WebSocket client with live UI.
- JavaScript WebSocket implementation
- Real-time workflow progress display
- Interactive controls (start workflow, stop, reconnect)
- Visual agent spawning timeline
- **Prerequisites**: API server running with CORS enabled
- **Open**: `file:///.../examples/api_examples/websocket_client.html` in browser

### [knowledge_brain_demo.html](knowledge_brain_demo.html)
Browser-based Knowledge Brain interface.
- Document upload form
- Semantic search interface
- Knowledge graph visualization
- Concept explorer
- **Prerequisites**: API server running with CORS enabled
- **Open**: `file:///.../examples/api_examples/knowledge_brain_demo.html` in browser

## Quick Start

### 1. Start API Server
```bash
# Set API key (optional for development)
export FELIX_API_KEY="your-secret-key"

# Start server
python -m uvicorn src.api.main:app --reload --port 8000

# Verify
curl http://localhost:8000/health
```

### 2. Run Python Client
```bash
# Set API key in client
export FELIX_API_KEY="your-secret-key"

# Run example
python examples/api_examples/memory_history_client.py
```

### 3. Open Browser Client
```bash
# Open in browser
open examples/api_examples/websocket_client.html

# Or use HTTP server to avoid CORS issues
cd examples/api_examples
python -m http.server 8080

# Then open: http://localhost:8080/websocket_client.html
```

## API Endpoints Reference

### Workflows
- `POST /api/v1/workflows` - Create new workflow
- `GET /api/v1/workflows/{id}` - Get workflow status
- `GET /api/v1/workflows` - List workflows
- `DELETE /api/v1/workflows/{id}` - Cancel workflow

### Knowledge Brain
- `POST /api/v1/knowledge/ingest` - Ingest document
- `POST /api/v1/knowledge/search` - Semantic search
- `GET /api/v1/knowledge/concepts` - List concepts
- `POST /api/v1/knowledge/daemon/start` - Start daemon

### WebSocket
- `WS /api/v1/ws/workflows/{id}` - Stream workflow events

Full API documentation: http://localhost:8000/docs

## Related Modules
- [src/api/](../../src/api/) - API implementation
- [src/api/routers/](../../src/api/routers/) - Endpoint routers
- [src/api/websockets/](../../src/api/websockets/) - WebSocket implementation
