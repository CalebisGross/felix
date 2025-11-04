# API Routers Module

## Purpose
FastAPI router modules providing domain-specific REST endpoints for workflows, agents, knowledge operations, memory access, and context compression.

## Key Files

### [workflows.py](workflows.py)
Workflow creation, execution, and monitoring endpoints.
- **`POST /api/v1/workflows`**: Create and execute new workflow with async background processing
- **`GET /api/v1/workflows/{workflow_id}`**: Retrieve workflow status and results
- **`GET /api/v1/workflows`**: List all workflows with pagination and filtering
- **`DELETE /api/v1/workflows/{workflow_id}`**: Cancel running workflow
- **In-memory storage**: Fast access to active workflows (completed workflows persist to `felix_workflow_history.db`)
- **Thread pool executor**: Bridges sync Felix code with async FastAPI
- **WebSocket integration**: Broadcasts workflow events for real-time monitoring

### [knowledge.py](knowledge.py)
Knowledge Brain operations for document ingestion and semantic search.
- **Document ingestion**: `/ingest` (single), `/ingest/batch` (multiple), `/documents` (list)
- **Semantic search**: `/search` for concept retrieval with relevance scoring
- **Knowledge augmentation**: `/augment` to enrich task context with relevant knowledge
- **Graph operations**: `/graph/build`, `/graph/relationships`, `/graph/statistics`
- **Daemon control**: `/daemon/status`, `/daemon/start`, `/daemon/stop`, `/daemon/watch-dirs`
- **Concept browsing**: `/concepts`, `/concepts/{concept}`, `/concepts/{concept}/related`
- Requires Knowledge Brain enabled via `FELIX_ENABLE_KNOWLEDGE_BRAIN=true`

### [knowledge_memory.py](knowledge_memory.py)
Combined knowledge and memory operations (unified interface).
- **Unified search**: `/search` across both knowledge entries and task memory
- **Hybrid retrieval**: Combines semantic search with task pattern matching
- **Cross-domain queries**: Retrieves relevant information regardless of storage location

### [task_memory.py](task_memory.py)
Task memory and pattern storage endpoints.
- **`GET /api/v1/memory/tasks`**: Retrieve task patterns with filtering
- **`POST /api/v1/memory/tasks`**: Store new task memory
- **`GET /api/v1/memory/tasks/{task_id}`**: Get specific task details
- **`DELETE /api/v1/memory/tasks/{task_id}`**: Remove task from memory
- **Pattern search**: Query by task similarity and completion status

### [workflow_history.py](workflow_history.py)
Historical workflow data and analytics.
- **`GET /api/v1/history`**: Query workflow history with date range and status filters
- **`GET /api/v1/history/{workflow_id}`**: Retrieve complete workflow record
- **`GET /api/v1/history/stats`**: Aggregate statistics (success rate, avg confidence, token usage)
- **Export**: Download workflow data in JSON format for analysis
- Accesses `felix_workflow_history.db` for persistent storage

### [agents.py](agents.py)
Agent interaction and management endpoints.
- **`GET /api/v1/agents`**: List active agents with status
- **`GET /api/v1/agents/{agent_id}`**: Agent details and current state
- **`POST /api/v1/agents/{agent_id}/task`**: Direct task submission to specific agent
- **`GET /api/v1/agents/registry`**: Query AgentRegistry for team composition
- **Agent awareness**: Discover agents by role, phase, or capability

### [compression.py](compression.py)
Context compression operations for managing token budgets.
- **`POST /api/v1/compression/compress`**: Compress text using abstractive compression (0.3 ratio)
- **`POST /api/v1/compression/estimate`**: Estimate token count without compression
- **Compression modes**: Abstractive (default), extractive, hybrid
- **Token management**: Helps stay within LLM context limits

## Key Concepts

### Router Organization
Each router is a self-contained module with:
- **APIRouter instance**: Defines path prefix and tags
- **Dependency injection**: Uses FastAPI's `Depends()` for auth and resource access
- **Pydantic validation**: All requests/responses use models from `models.py`
- **Error handling**: Consistent HTTPException patterns with proper status codes

### Async/Sync Bridge
Felix's core is synchronous, but FastAPI is async. Routers use:
- **ThreadPoolExecutor**: Run sync Felix code without blocking async event loop
- **Background tasks**: Long-running workflows execute in background
- **Async wrappers**: Convert sync operations to async for FastAPI compatibility

### Authentication
All endpoints require authentication via `verify_api_key` dependency (unless API key not configured - development mode).

### In-Memory vs Persistent Storage
- **Active workflows**: In-memory dict for low-latency access
- **Completed workflows**: Auto-saved to `felix_workflow_history.db`
- **Knowledge**: Persistent SQLite databases
- **Design note**: Suitable for single-instance deployments; use Redis/PostgreSQL for multi-instance

### WebSocket Event Broadcasting
Workflow endpoints broadcast events to connected WebSocket clients:
- `workflow.created`: New workflow started
- `workflow.agent_spawned`: Agent joined team
- `workflow.synthesis`: Final result ready
- `workflow.completed` / `workflow.failed`: Terminal states

### Error Responses
Routers return structured error responses:
- **400**: Bad request (validation error)
- **401**: Unauthorized (missing/invalid API key)
- **404**: Resource not found
- **503**: Service unavailable (e.g., Knowledge Brain disabled)
- **500**: Internal server error

## Related Modules
- [models.py](../models.py) - Pydantic schemas for all request/response data
- [dependencies.py](../dependencies.py) - Shared dependencies and authentication
- [websockets/](../websockets/) - WebSocket implementation for real-time updates
- [workflows/](../../workflows/) - Workflow execution engine
- [knowledge/](../../knowledge/) - Knowledge Brain implementation
- [memory/](../../memory/) - Memory storage backends
