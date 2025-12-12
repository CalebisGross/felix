# REST API Module

## Purpose
FastAPI REST API server providing programmatic access to Felix's multi-agent workflow system, with async execution, WebSocket streaming, and optional authentication.

## Key Files

### [main.py](main.py)
Main FastAPI application entry point.
- **`app`**: FastAPI application instance with lifespan management
- **Endpoints**: System management (`/api/v1/system/*`), health checks (`/health`, `/ready`)
- **Middleware**: CORS configuration for web clients
- **Documentation**: Auto-generated Swagger UI at `/docs`

### [dependencies.py](dependencies.py)
Shared dependencies for authentication and Felix instance management.
- **`verify_api_key()`**: API key authentication via Bearer token (optional, skips if `FELIX_API_KEY` not set)
- **`get_felix()`**: Singleton Felix instance provider for dependency injection
- **`initialize_felix()`**: Startup function to initialize Felix system with environment config
- **`shutdown_felix()`**: Cleanup function for graceful shutdown

### [models.py](models.py)
Pydantic models for request/response schemas and validation.
- **System Models**: `SystemStatus`, `SystemConfig` for system management
- **Workflow Models**: `WorkflowRequest`, `WorkflowResponse`, `WorkflowStatus` enum, `AgentInfo`, `SynthesisResult`
- **Agent Models**: Agent interaction schemas
- **Memory Models**: Knowledge and task memory schemas
- **Compression Models**: Context compression schemas
- All models include JSON schema examples and validation rules

### [routers/](routers/)
API endpoint routers organized by domain (workflows, agents, knowledge, memory).

### [websockets/](websockets/)
WebSocket implementation for real-time workflow streaming.

### [middleware/](middleware/)
Custom middleware for authentication, rate limiting, and request processing (currently empty - using FastAPI built-ins).

## Key Concepts

### Async Architecture
FastAPI's async capabilities enable non-blocking workflow execution through background tasks and thread pool executors, allowing multiple workflows to run concurrently.

### Lifespan Management
Application uses FastAPI's `@asynccontextmanager` for proper startup/shutdown:
- **Startup**: Initializes Felix system, connects to LLM provider, loads configuration
- **Shutdown**: Gracefully stops workflows, closes connections, cleans up resources

### Environment Configuration
All configuration via environment variables:
- `FELIX_API_KEY`: Optional API key for authentication (development mode if unset)
- `FELIX_LM_HOST` / `FELIX_LM_PORT`: LM Studio connection (default: 127.0.0.1:1234)
- `FELIX_MAX_AGENTS`: Agent limit per workflow (default: 10)
- `FELIX_ENABLE_KNOWLEDGE_BRAIN`: Enable autonomous knowledge system (default: false)
- `FELIX_CORS_ORIGINS`: Allowed CORS origins (default: localhost:3000,8080)

### Optional Authentication
API key authentication is optional for development. If `FELIX_API_KEY` is not set, authentication is skipped. Production deployments should always set an API key.

### Router Organization
Endpoints organized by domain for clean separation:
- `/api/v1/system/*` - System lifecycle and status
- `/api/v1/workflows/*` - Workflow creation and management
- `/api/v1/agents/*` - Agent interaction
- `/api/v1/knowledge/*` - Knowledge brain operations
- `/api/v1/memory/*` - Memory access
- `/ws/workflow/{workflow_id}` - WebSocket streaming

### OpenAPI Documentation
Auto-generated API documentation available at:
- **Swagger UI**: `/docs` - Interactive API testing
- **ReDoc**: `/redoc` - Alternative documentation view
- **OpenAPI JSON**: `/openapi.json` - Machine-readable spec

## Related Modules
- [workflows/](../workflows/) - Workflow execution engine used by API
- [gui/felix_system.py](../gui/felix_system.py) - FelixSystem class managing core functionality
- [communication/](../communication/) - CentralPost and agent coordination
- [llm/](../llm/) - LLM provider integration and routing
