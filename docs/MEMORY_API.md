# Felix Memory & History API

Comprehensive API documentation for Felix's Memory & History systems, providing access to task patterns, workflow execution history, knowledge entries, and context compression.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
  - [Task Memory API](#task-memory-api)
  - [Workflow History API](#workflow-history-api)
  - [Knowledge Memory API](#knowledge-memory-api)
  - [Context Compression API](#context-compression-api)
- [Usage Examples](#usage-examples)
- [Integration Workflows](#integration-workflows)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Memory & History API exposes Felix's four memory systems through 22 REST endpoints:

1. **Task Memory** (7 endpoints) - Pattern recognition and strategy recommendations from execution history
2. **Workflow History** (7 endpoints) - Complete workflow execution records with conversation threading
3. **Knowledge Memory** (6 endpoints) - Agent insights and domain expertise with meta-learning
4. **Context Compression** (2 endpoints) - Intelligent context size management

### Key Features

- **Pattern Learning**: Automatic pattern extraction from task execution history
- **Strategy Recommendations**: Historical data-driven task strategy suggestions
- **Conversation Threading**: Multi-turn workflow continuity tracking
- **Meta-Learning Boost**: Knowledge ranking based on historical usefulness
- **Knowledge Graph**: Relationship discovery between knowledge entries
- **Multiple Compression Strategies**: 6 algorithms for context optimization
- **Full-Text Search**: Across workflows and knowledge entries
- **Performance Analytics**: Comprehensive workflow metrics and statistics

---

## Architecture

### Memory Systems Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Felix Memory & History API                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Task Memory  │  │  Workflow    │  │  Knowledge   │      │
│  │              │  │   History    │  │    Memory    │      │
│  │ - Patterns   │  │              │  │              │      │
│  │ - Executions │  │ - Records    │  │ - Entries    │      │
│  │ - Strategies │  │ - Threads    │  │ - Relations  │      │
│  │              │  │ - Analytics  │  │ - Usage      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                            │                                 │
│                   ┌────────▼────────┐                        │
│                   │   Compression   │                        │
│                   │                 │                        │
│                   │  6 Strategies   │                        │
│                   │  4 Levels       │                        │
│                   └─────────────────┘                        │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│               Database Layer (SQLite)                        │
│                                                               │
│  felix_task_memory.db  felix_workflow_history.db             │
│  felix_knowledge.db                                          │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Write Path**: API → Memory System → SQLite Database
2. **Read Path**: API → Memory System → Database Query → Response Transformation
3. **Meta-Learning**: Usage tracking → Statistical analysis → Ranking boost

---

## Authentication

All endpoints require authentication via Bearer token:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:8000/api/v1/memory/...
```

Set API key via environment variable:

```bash
export FELIX_API_KEY=your_secret_key_here
```

If `FELIX_API_KEY` is not set, the API runs in development mode without authentication.

---

## API Endpoints

### Task Memory API

Task Memory learns patterns from execution history and provides strategy recommendations.

**Base Path**: `/api/v1/memory/tasks/`

#### 1. List Task Patterns

Get task patterns with optional filtering.

**Endpoint**: `GET /api/v1/memory/tasks/patterns`

**Query Parameters**:
- `task_types` (optional): Filter by task types (e.g., `research`, `analysis`)
- `complexity_levels` (optional): Filter by complexity (`simple`, `moderate`, `complex`, `very_complex`)
- `min_success_rate` (optional): Minimum success rate (0.0-1.0)
- `max_duration` (optional): Maximum typical duration in seconds
- `keywords` (optional): Keywords to match
- `limit` (default: 50): Maximum results

**Response**:
```json
{
  "patterns": [
    {
      "pattern_id": "research_complex_abc123",
      "task_type": "research",
      "complexity": "complex",
      "keywords": ["quantum", "computing", "algorithms"],
      "typical_duration": 45.5,
      "success_rate": 0.87,
      "failure_modes": ["timeout", "insufficient_data"],
      "optimal_strategies": ["web-search", "multi-agent"],
      "required_agents": ["research", "analysis", "critic"],
      "context_requirements": {"min_tokens": 2000},
      "usage_count": 15,
      "created_at": "2025-10-30T10:00:00Z",
      "updated_at": "2025-10-31T14:30:00Z"
    }
  ],
  "total": 1
}
```

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  "http://localhost:8000/api/v1/memory/tasks/patterns?task_types=research&min_success_rate=0.8&limit=10"
```

---

#### 2. Get Task Pattern

Get specific pattern by ID.

**Endpoint**: `GET /api/v1/memory/tasks/patterns/{pattern_id}`

**Response**: Same as pattern object in list response.

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  http://localhost:8000/api/v1/memory/tasks/patterns/research_complex_abc123
```

---

#### 3. List Task Executions

Get task execution records with filtering.

**Endpoint**: `GET /api/v1/memory/tasks/executions`

**Query Parameters**:
- `task_types` (optional): Filter by task types
- `complexity_levels` (optional): Filter by complexity
- `outcomes` (optional): Filter by outcomes (`success`, `partial_success`, `failure`, `timeout`, `error`)
- `from_date` (optional): Start date (ISO format)
- `to_date` (optional): End date (ISO format)
- `limit` (default: 50): Maximum results

**Response**:
```json
{
  "executions": [
    {
      "execution_id": "exec_xyz789",
      "task_description": "Research quantum computing fundamentals",
      "task_type": "research",
      "complexity": "complex",
      "outcome": "success",
      "duration": 42.5,
      "agents_used": ["research_001", "analysis_002", "critic_003"],
      "strategies_used": ["web-search", "multi-agent"],
      "context_size": 3500,
      "error_messages": [],
      "success_metrics": {"quality": 0.92, "completeness": 0.88},
      "patterns_matched": ["research_complex_abc123"],
      "created_at": "2025-10-31T10:00:00Z"
    }
  ],
  "total": 1
}
```

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  "http://localhost:8000/api/v1/memory/tasks/executions?task_types=research&outcomes=success&limit=20"
```

---

#### 4. Get Task Execution

Get specific execution by ID.

**Endpoint**: `GET /api/v1/memory/tasks/executions/{execution_id}`

**Response**: Same as execution object in list response.

---

#### 5. Record Task Execution

Record a new task execution.

**Endpoint**: `POST /api/v1/memory/tasks/executions`

**Request Body**:
```json
{
  "task_description": "Research quantum computing fundamentals for technical report",
  "task_type": "research",
  "complexity": "complex",
  "outcome": "success",
  "duration": 42.5,
  "agents_used": ["research_001", "analysis_002"],
  "strategies_used": ["web-search", "multi-agent"],
  "context_size": 3500,
  "error_messages": [],
  "success_metrics": {"quality": 0.92}
}
```

**Response**:
```json
{
  "execution_id": "exec_xyz789",
  "patterns_matched": ["research_complex_abc123"],
  "patterns_updated": ["research_complex_abc123"],
  "message": "Execution recorded successfully. Matched 1 existing patterns."
}
```

**Example**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Analyze market trends",
    "task_type": "analysis",
    "complexity": "moderate",
    "outcome": "success",
    "duration": 25.3,
    "agents_used": ["analysis_001"],
    "strategies_used": ["statistical-analysis"],
    "context_size": 2000
  }' \
  http://localhost:8000/api/v1/memory/tasks/executions
```

---

#### 6. Recommend Strategy

Get strategy recommendation for a task.

**Endpoint**: `POST /api/v1/memory/tasks/recommend-strategy`

**Request Body**:
```json
{
  "task_description": "Analyze market trends for renewable energy",
  "task_type": "analysis",
  "complexity": "complex"
}
```

**Response**:
```json
{
  "recommended_strategies": ["web-search", "multi-agent", "knowledge-augment"],
  "recommended_agents": ["research", "analysis", "critic"],
  "estimated_duration": 45.0,
  "success_probability": 0.87,
  "similar_patterns": [
    {
      "pattern_id": "analysis_complex_def456",
      "task_type": "analysis",
      "complexity": "complex",
      "success_rate": 0.89,
      "usage_count": 12
    }
  ],
  "confidence": 0.92
}
```

**Example**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "Research latest AI developments",
    "task_type": "research",
    "complexity": "complex"
  }' \
  http://localhost:8000/api/v1/memory/tasks/recommend-strategy
```

---

#### 7. Get Task Memory Summary

Get task memory statistics.

**Endpoint**: `GET /api/v1/memory/tasks/summary`

**Response**:
```json
{
  "total_patterns": 45,
  "total_executions": 238,
  "average_success_rate": 0.85,
  "most_common_task_types": {
    "research": 120,
    "analysis": 85,
    "synthesis": 33
  },
  "complexity_distribution": {
    "simple": 45,
    "moderate": 98,
    "complex": 75,
    "very_complex": 20
  },
  "outcome_distribution": {
    "success": 202,
    "partial_success": 24,
    "failure": 12
  }
}
```

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  http://localhost:8000/api/v1/memory/tasks/summary
```

---

### Workflow History API

Workflow History tracks complete workflow execution records with conversation threading.

**Base Path**: `/api/v1/memory/workflows/`

#### 8. List Workflows

Get workflow execution records with filtering and pagination.

**Endpoint**: `GET /api/v1/memory/workflows/`

**Query Parameters**:
- `status_filter` (optional): Filter by status (`completed`, `failed`, etc.)
- `from_date` (optional): Start date (ISO format)
- `to_date` (optional): End date (ISO format)
- `search_query` (optional): Search in task_input and synthesis
- `parent_workflow_id` (optional): Filter by parent workflow
- `conversation_thread_id` (optional): Filter by thread
- `limit` (default: 50): Maximum results
- `offset` (default: 0): Pagination offset

**Response**:
```json
{
  "workflows": [
    {
      "workflow_id": 123,
      "task_input": "Explain quantum computing in simple terms",
      "status": "completed",
      "created_at": "2025-10-30T10:00:00Z",
      "completed_at": "2025-10-30T10:02:30Z",
      "final_synthesis": "Quantum computing uses quantum mechanical phenomena...",
      "confidence": 0.87,
      "agents_count": 3,
      "tokens_used": 2450,
      "max_tokens": 3000,
      "processing_time": 150.5,
      "temperature": 0.3,
      "metadata": {},
      "parent_workflow_id": null,
      "conversation_thread_id": "thread_abc123"
    }
  ],
  "total": 1,
  "offset": 0,
  "limit": 50
}
```

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  "http://localhost:8000/api/v1/memory/workflows/?status_filter=completed&limit=10"
```

---

#### 9. Get Workflow

Get specific workflow by ID.

**Endpoint**: `GET /api/v1/memory/workflows/{workflow_id}`

**Response**: Same as workflow object in list response.

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  http://localhost:8000/api/v1/memory/workflows/123
```

---

#### 10. Get Conversation Thread

Get complete conversation thread for a workflow.

**Endpoint**: `GET /api/v1/memory/workflows/{workflow_id}/thread`

**Response**:
```json
{
  "thread_id": "thread_abc123",
  "root_workflow": {
    "workflow_id": 120,
    "task_input": "What is quantum computing?",
    "status": "completed",
    "confidence": 0.85
  },
  "child_workflows": [
    {
      "workflow_id": 121,
      "task_input": "Can you explain quantum entanglement?",
      "parent_workflow_id": 120,
      "confidence": 0.88
    },
    {
      "workflow_id": 122,
      "task_input": "What are the applications?",
      "parent_workflow_id": 121,
      "confidence": 0.90
    }
  ],
  "total_workflows": 3,
  "thread_depth": 3
}
```

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  http://localhost:8000/api/v1/memory/workflows/122/thread
```

---

#### 11. Save Workflow

Save a new workflow to history.

**Endpoint**: `POST /api/v1/memory/workflows/`

**Request Body**:
```json
{
  "task_input": "Explain quantum computing",
  "status": "completed",
  "final_synthesis": "Quantum computing...",
  "confidence": 0.87,
  "agents_count": 3,
  "tokens_used": 2450,
  "max_tokens": 3000,
  "processing_time": 150.5,
  "temperature": 0.3,
  "metadata": {},
  "parent_workflow_id": null,
  "conversation_thread_id": "thread_abc123"
}
```

**Response**: Saved workflow with assigned `workflow_id`.

---

#### 12. Delete Workflow

Delete a workflow from history.

**Endpoint**: `DELETE /api/v1/memory/workflows/{workflow_id}`

**Response**:
```json
{
  "message": "Workflow 123 deleted successfully"
}
```

---

#### 13. Search Workflows

Full-text search across workflows.

**Endpoint**: `GET /api/v1/memory/workflows/search/`

**Query Parameters**:
- `query` (required): Search query
- `limit` (default: 50): Maximum results
- `offset` (default: 0): Pagination offset

**Response**: Same as list workflows response.

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  "http://localhost:8000/api/v1/memory/workflows/search/?query=quantum%20computing&limit=10"
```

---

#### 14. Get Workflow Analytics

Get workflow performance metrics.

**Endpoint**: `GET /api/v1/memory/workflows/analytics/summary`

**Query Parameters**:
- `from_date` (optional): Start date (ISO format)
- `to_date` (optional): End date (ISO format)

**Response**:
```json
{
  "total_workflows": 150,
  "completed_workflows": 142,
  "failed_workflows": 8,
  "average_confidence": 0.85,
  "average_agents_count": 4.2,
  "average_processing_time": 38.5,
  "average_tokens_used": 2850.0,
  "status_distribution": {
    "completed": 142,
    "failed": 8
  },
  "workflows_by_date": {
    "2025-10-30": 45,
    "2025-10-31": 105
  }
}
```

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  "http://localhost:8000/api/v1/memory/workflows/analytics/summary?from_date=2025-10-01T00:00:00Z"
```

---

### Knowledge Memory API

Knowledge Memory stores agent insights, task results, and domain expertise with meta-learning boost.

**Base Path**: `/api/v1/memory/knowledge/`

#### 15. Retrieve Knowledge

Get knowledge entries with filtering and meta-learning boost.

**Endpoint**: `GET /api/v1/memory/knowledge/`

**Query Parameters**:
- `knowledge_types` (optional): Filter by types (`task_result`, `agent_insight`, `pattern_recognition`, `failure_analysis`, `optimization_data`, `domain_expertise`)
- `domains` (optional): Filter by domains
- `tags` (optional): Filter by tags
- `min_confidence` (optional): Minimum confidence (`low`, `medium`, `high`, `verified`)
- `min_success_rate` (optional): Minimum success rate (0.0-1.0)
- `content_keywords` (optional): Keywords in content
- `from_date` (optional): Start date
- `to_date` (optional): End date
- `task_type` (optional): For meta-learning boost
- `task_complexity` (optional): For meta-learning boost
- `limit` (default: 50): Maximum results
- `offset` (default: 0): Pagination offset

**Response**:
```json
{
  "entries": [
    {
      "knowledge_id": "kb_abc123",
      "knowledge_type": "domain_expertise",
      "content": {
        "concept": "quantum entanglement",
        "definition": "A phenomenon where quantum states become correlated",
        "applications": ["quantum computing", "quantum cryptography"]
      },
      "confidence_level": "high",
      "source_agent": "research_001",
      "domain": "physics",
      "tags": ["quantum", "physics", "advanced"],
      "created_at": "2025-10-30T10:00:00Z",
      "updated_at": "2025-10-31T15:00:00Z",
      "access_count": 15,
      "success_rate": 0.92,
      "validation_score": 0.88,
      "validation_status": "trusted"
    }
  ],
  "total": 1,
  "offset": 0,
  "limit": 50
}
```

**Meta-Learning**: When `task_type` and `task_complexity` are provided, results are re-ranked based on historical usefulness for similar tasks.

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  "http://localhost:8000/api/v1/memory/knowledge/?domains=physics&min_confidence=high&task_type=research&limit=10"
```

---

#### 16. Get Knowledge Entry

Get specific knowledge entry by ID.

**Endpoint**: `GET /api/v1/memory/knowledge/{knowledge_id}`

**Response**: Same as knowledge entry object.

**Note**: Increments `access_count` for usage tracking.

---

#### 17. Store Knowledge

Store a new knowledge entry.

**Endpoint**: `POST /api/v1/memory/knowledge/`

**Request Body**:
```json
{
  "knowledge_type": "domain_expertise",
  "content": {
    "concept": "quantum entanglement",
    "definition": "...",
    "applications": ["..."]
  },
  "confidence_level": "high",
  "source_agent": "research_001",
  "domain": "physics",
  "tags": ["quantum", "physics"]
}
```

**Response**:
```json
{
  "knowledge_id": "kb_abc123",
  "stored": true,
  "updated": false,
  "message": "Knowledge entry stored successfully"
}
```

**Deduplication**: Entries with same domain, content, and source are updated instead of duplicated.

**Example**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_type": "agent_insight",
    "content": {"insight": "Multi-agent approach improves research quality"},
    "confidence_level": "high",
    "source_agent": "analysis_001",
    "domain": "methodology",
    "tags": ["multi-agent", "research"]
  }' \
  http://localhost:8000/api/v1/memory/knowledge/
```

---

#### 18. Record Knowledge Usage

Record knowledge usage for meta-learning.

**Endpoint**: `POST /api/v1/memory/knowledge/{knowledge_id}/usage`

**Request Body**:
```json
{
  "knowledge_id": "kb_abc123",
  "workflow_id": "wf_xyz789",
  "task_type": "research",
  "task_complexity": "complex",
  "useful_score": 0.9,
  "retrieval_method": "semantic"
}
```

**Response**:
```json
{
  "recorded": true,
  "message": "Usage recorded successfully for meta-learning"
}
```

**Purpose**: Tracks how useful each knowledge entry is for specific task types, enabling intelligent re-ranking in future searches.

---

#### 19. Update Knowledge Success Rate

Update the success rate of a knowledge entry.

**Endpoint**: `PATCH /api/v1/memory/knowledge/{knowledge_id}/success-rate`

**Request Body**:
```json
{
  "new_success_rate": 0.92
}
```

**Response**:
```json
{
  "knowledge_id": "kb_abc123",
  "stored": false,
  "updated": true,
  "message": "Success rate updated to 0.92"
}
```

---

#### 20. Get Related Knowledge

Get knowledge entries related to a specific entry.

**Endpoint**: `GET /api/v1/memory/knowledge/{knowledge_id}/related`

**Query Parameters**:
- `max_results` (default: 10): Maximum related entries

**Response**:
```json
{
  "knowledge_id": "kb_abc123",
  "relationships": [
    {
      "target_knowledge_id": "kb_def456",
      "target_content_preview": "Quantum computing applications in cryptography...",
      "relationship_strength": 0.85,
      "relationship_type": "related_domain"
    }
  ],
  "total": 1
}
```

**Relationship Types**:
- `explicit_mention`: Explicitly linked in knowledge_relationships table
- `related_domain`: Same domain
- `related_tags`: Shared tags
- `embedding_similarity`: Similar content (if embeddings available)

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  "http://localhost:8000/api/v1/memory/knowledge/kb_abc123/related?max_results=5"
```

---

#### 21. Get Knowledge Summary

Get knowledge memory statistics.

**Endpoint**: `GET /api/v1/memory/knowledge/summary/stats`

**Response**:
```json
{
  "total_entries": 485,
  "entries_by_type": {
    "domain_expertise": 150,
    "agent_insight": 200,
    "task_result": 100,
    "pattern_recognition": 35
  },
  "entries_by_domain": {
    "physics": 120,
    "computer_science": 180,
    "methodology": 185
  },
  "entries_by_confidence": {
    "high": 300,
    "verified": 100,
    "medium": 85
  },
  "average_success_rate": 0.87,
  "total_access_count": 5420
}
```

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  http://localhost:8000/api/v1/memory/knowledge/summary/stats
```

---

### Context Compression API

Context Compression provides intelligent context size management with multiple strategies.

**Base Path**: `/api/v1/memory/compression/`

#### 22. Compress Context

Compress context using specified strategy and level.

**Endpoint**: `POST /api/v1/memory/compression/compress`

**Request Body**:
```json
{
  "context": {
    "agent_outputs": [
      "Research Agent: Quantum computing uses quantum-mechanical phenomena...",
      "Analysis Agent: Current state shows significant progress...",
      "Critic Agent: While promising, practical applications remain limited..."
    ],
    "synthesis": "Quantum computing shows promise but faces technical challenges"
  },
  "strategy": "progressive_refinement",
  "level": "moderate",
  "preserve_keywords": ["quantum", "error correction"],
  "preserve_structure": true,
  "topic_keywords": ["quantum computing"]
}
```

**Compression Strategies**:
- `extractive_summary`: Keep important sentences (fast, preserves exact text)
- `abstractive_summary`: Create brief summaries (concise, loses detail)
- `keyword_extraction`: Extract key concepts and keywords
- `hierarchical_summary`: 3-level structure (core/supporting/auxiliary)
- `relevance_filtering`: Keep only topic-relevant content
- `progressive_refinement`: Multiple passes (best quality, slower)

**Compression Levels**:
- `light`: 80% of original size
- `moderate`: 60% of original size
- `heavy`: 40% of original size
- `extreme`: 20% of original size

**Response**:
```json
{
  "context_id": "ctx_abc123def456",
  "original_size": 5000,
  "compressed_size": 3000,
  "compression_ratio": 0.6,
  "strategy_used": "progressive_refinement",
  "level_used": "moderate",
  "compressed_content": {
    "core_concepts": ["quantum computing", "superposition", "entanglement"],
    "summary": "Quantum computing leverages quantum phenomena for computation..."
  },
  "relevance_scores": {
    "agent_outputs[0]": 0.95,
    "agent_outputs[1]": 0.88,
    "agent_outputs[2]": 0.72
  },
  "processing_time_ms": 45.3
}
```

**Example**:
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "context": {"text": "Very long context..."},
    "strategy": "hierarchical_summary",
    "level": "moderate"
  }' \
  http://localhost:8000/api/v1/memory/compression/compress
```

---

#### 23. Get Compression Stats

Get compression system statistics and configuration.

**Endpoint**: `GET /api/v1/memory/compression/stats`

**Response**:
```json
{
  "max_context_size": 4000,
  "default_strategy": "progressive_refinement",
  "default_level": "moderate",
  "available_strategies": [
    "extractive_summary",
    "abstractive_summary",
    "keyword_extraction",
    "hierarchical_summary",
    "relevance_filtering",
    "progressive_refinement"
  ],
  "available_levels": [
    "light",
    "moderate",
    "heavy",
    "extreme"
  ]
}
```

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_KEY" \
  http://localhost:8000/api/v1/memory/compression/stats
```

---

## Usage Examples

### Example 1: Get Strategy Recommendation

```python
import httpx

API_URL = "http://localhost:8000"
API_KEY = "your_key"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Get recommendation
response = httpx.post(
    f"{API_URL}/api/v1/memory/tasks/recommend-strategy",
    headers=headers,
    json={
        "task_description": "Research quantum computing applications in cryptography",
        "task_type": "research",
        "complexity": "complex"
    }
)

recommendation = response.json()

print(f"Recommended strategies: {recommendation['recommended_strategies']}")
print(f"Recommended agents: {recommendation['recommended_agents']}")
print(f"Success probability: {recommendation['success_probability']:.1%}")
```

---

### Example 2: Record Task Execution and Update Patterns

```python
# Execute a task (your application logic)
start_time = time.time()
# ... perform task ...
duration = time.time() - start_time

# Record execution
response = httpx.post(
    f"{API_URL}/api/v1/memory/tasks/executions",
    headers=headers,
    json={
        "task_description": "Research quantum cryptography",
        "task_type": "research",
        "complexity": "complex",
        "outcome": "success",
        "duration": duration,
        "agents_used": ["research_001", "analysis_002"],
        "strategies_used": ["web-search", "multi-agent"],
        "context_size": 3200,
        "success_metrics": {"quality": 0.90}
    }
)

result = response.json()
print(f"Patterns matched: {len(result['patterns_matched'])}")
```

---

### Example 3: Retrieve Knowledge with Meta-Learning Boost

```python
# Retrieve knowledge for a research task
response = httpx.get(
    f"{API_URL}/api/v1/memory/knowledge/",
    headers=headers,
    params={
        "domains": ["physics", "computer_science"],
        "min_confidence": "high",
        "task_type": "research",  # Triggers meta-learning boost
        "task_complexity": "complex",
        "limit": 10
    }
)

knowledge = response.json()

for entry in knowledge['entries']:
    print(f"Knowledge: {entry['knowledge_id']}")
    print(f"Success rate: {entry['success_rate']:.1%}")
    print(f"Access count: {entry['access_count']}")
```

---

### Example 4: Track Conversation Thread

```python
# Start a conversation
workflow_1 = create_workflow("What is quantum computing?")

# Continue the conversation
workflow_2 = create_workflow(
    "Can you explain quantum entanglement?",
    parent_workflow_id=workflow_1['workflow_id']
)

# Get full thread
response = httpx.get(
    f"{API_URL}/api/v1/memory/workflows/{workflow_2['workflow_id']}/thread",
    headers=headers
)

thread = response.json()

print(f"Thread contains {thread['total_workflows']} workflows")
print(f"Thread depth: {thread['thread_depth']}")
```

---

### Example 5: Compress Large Context

```python
# Large context from multiple agents
large_context = {
    "agent_outputs": [
        "Research Agent: <very long text>...",
        "Analysis Agent: <very long text>...",
        "Critic Agent: <very long text>..."
    ],
    "synthesis": "<synthesis text>"
}

# Compress using progressive refinement
response = httpx.post(
    f"{API_URL}/api/v1/memory/compression/compress",
    headers=headers,
    json={
        "context": large_context,
        "strategy": "progressive_refinement",
        "level": "moderate",
        "preserve_keywords": ["quantum", "entanglement"]
    }
)

compressed = response.json()

print(f"Original: {compressed['original_size']} chars")
print(f"Compressed: {compressed['compressed_size']} chars")
print(f"Ratio: {compressed['compression_ratio']:.1%}")
```

---

## Integration Workflows

### Workflow 1: Augmented Task Execution

Combine strategy recommendation, execution, and knowledge retrieval:

```python
def execute_augmented_task(task_description, task_type, complexity):
    # 1. Get strategy recommendation
    recommendation = get_strategy_recommendation(task_description, task_type, complexity)

    # 2. Retrieve relevant knowledge with meta-learning boost
    knowledge = retrieve_knowledge(
        task_type=task_type,
        task_complexity=complexity,
        domains=infer_domains(task_description)
    )

    # 3. Execute task using recommended strategies and knowledge
    result = execute_task(
        description=task_description,
        strategies=recommendation['recommended_strategies'],
        agents=recommendation['recommended_agents'],
        knowledge_context=knowledge
    )

    # 4. Record execution for pattern learning
    record_execution(
        task_description=task_description,
        task_type=task_type,
        complexity=complexity,
        outcome="success" if result.success else "failure",
        duration=result.duration,
        agents_used=result.agents,
        strategies_used=recommendation['recommended_strategies']
    )

    # 5. Record knowledge usage for meta-learning
    for entry in knowledge['entries']:
        record_knowledge_usage(
            knowledge_id=entry['knowledge_id'],
            workflow_id=result.workflow_id,
            task_type=task_type,
            task_complexity=complexity,
            useful_score=calculate_usefulness(entry, result)
        )

    return result
```

---

### Workflow 2: Conversation Continuity

Maintain context across multi-turn conversations:

```python
def handle_conversation_turn(user_message, parent_workflow_id=None):
    # 1. Get conversation history if continuing
    context = ""
    if parent_workflow_id:
        thread = get_conversation_thread(parent_workflow_id)
        # Extract relevant context from thread
        context = summarize_thread(thread)

    # 2. Execute workflow with context
    workflow = execute_workflow(
        task=user_message,
        parent_workflow_id=parent_workflow_id,
        additional_context=context
    )

    # 3. Save to history with thread info
    save_workflow(workflow)

    return workflow
```

---

### Workflow 3: Performance Monitoring

Track and analyze workflow performance:

```python
def monitor_performance(from_date, to_date):
    # 1. Get workflow analytics
    analytics = get_workflow_analytics(from_date, to_date)

    # 2. Get task memory summary
    task_summary = get_task_memory_summary()

    # 3. Generate insights
    insights = {
        "completion_rate": analytics['completed_workflows'] / analytics['total_workflows'],
        "average_confidence": analytics['average_confidence'],
        "pattern_learning_rate": task_summary['total_patterns'] / task_summary['total_executions'],
        "optimization_opportunities": identify_low_performing_patterns(task_summary)
    }

    return insights
```

---

## Best Practices

### 1. Strategy Recommendations

- **Always request recommendations** before executing complex tasks
- **Update execution records** immediately after task completion
- **Track success metrics** to improve pattern quality
- **Minimum 5 executions** needed for reliable patterns

### 2. Knowledge Management

- **Use specific domains** for better organization
- **Tag liberally** for improved discoverability
- **Record usage** to enable meta-learning boost
- **Update success rates** based on actual outcomes
- **Validate entries** before marking as "verified"

### 3. Workflow History

- **Use conversation threading** for multi-turn interactions
- **Include metadata** for rich context
- **Search regularly** to avoid duplicate workflows
- **Clean up old workflows** to manage database size
- **Analyze trends** using analytics endpoints

### 4. Context Compression

- **Choose strategy** based on use case:
  - Use `extractive_summary` for speed and accuracy
  - Use `progressive_refinement` for best quality
  - Use `relevance_filtering` when topic is well-defined
- **Start with moderate level** and adjust based on results
- **Preserve keywords** for important domain terms
- **Monitor compression ratios** to ensure quality

### 5. Performance Optimization

- **Use pagination** for large result sets
- **Filter aggressively** to reduce data transfer
- **Cache frequently accessed data** on client side
- **Batch operations** when possible
- **Monitor API response times** and adjust queries

---

## Troubleshooting

### Issue: Strategy recommendation returns low confidence

**Cause**: Insufficient historical data for the task type/complexity combination.

**Solution**:
- Record more executions for this task type
- Patterns require minimum 2 similar executions
- Check if `keywords` in request match existing patterns

---

### Issue: Meta-learning boost not working

**Cause**: No usage records for the specified task type.

**Solution**:
- Record knowledge usage via `POST /knowledge/{id}/usage`
- Requires minimum 2 usage records for boost calculation
- Verify `task_type` and `task_complexity` match usage records

---

### Issue: Conversation thread not linking workflows

**Cause**: `parent_workflow_id` or `conversation_thread_id` not set correctly.

**Solution**:
- Set `parent_workflow_id` when creating follow-up workflow
- Use same `conversation_thread_id` for all related workflows
- Verify parent workflow exists before referencing

---

### Issue: Compression not reducing size enough

**Cause**: Strategy or level not aggressive enough.

**Solution**:
- Try `progressive_refinement` strategy for best results
- Increase compression level (`moderate` → `heavy` → `extreme`)
- For very long contexts, use `keyword_extraction`
- Check `preserve_keywords` - too many preserved keywords reduce compression

---

### Issue: Knowledge retrieval returns irrelevant entries

**Cause**: Query filters too broad or no meta-learning context.

**Solution**:
- Add `task_type` and `task_complexity` for meta-learning boost
- Use `domains` and `tags` filters
- Increase `min_confidence` level
- Add `content_keywords` for specific concepts

---

### Issue: Task patterns not being created

**Cause**: Not enough similar executions recorded.

**Solution**:
- Patterns require 2+ executions with 50% keyword overlap
- Ensure `task_type` and `complexity` are consistent
- Check keyword extraction (min 3 characters, excluding stopwords)
- Verify executions are actually being recorded (check execution count)

---

### Issue: Database connection errors

**Cause**: Database file locked or corrupted.

**Solution**:
- Check if another process is accessing the database
- Verify database file permissions
- Check disk space
- Restart Felix API server
- If persistent, backup and recreate database

---

### Issue: High API latency

**Cause**: Large result sets, complex queries, or database size.

**Solution**:
- Use smaller `limit` values
- Add more specific filters
- Enable pagination with `offset`
- Consider database cleanup for old entries
- Check server resources (CPU, memory, disk I/O)

---

## Support

For issues, questions, or feature requests:

- **Documentation**: [docs/API_QUICKSTART.md](API_QUICKSTART.md)
- **Issues**: https://github.com/anthropics/felix/issues
- **API Reference**: http://localhost:8000/docs (when server running)

---

## Appendix: Complete Endpoint Reference

| # | Method | Endpoint | Description |
|---|--------|----------|-------------|
| 1 | GET | `/api/v1/memory/tasks/patterns` | List task patterns |
| 2 | GET | `/api/v1/memory/tasks/patterns/{id}` | Get task pattern |
| 3 | GET | `/api/v1/memory/tasks/executions` | List executions |
| 4 | GET | `/api/v1/memory/tasks/executions/{id}` | Get execution |
| 5 | POST | `/api/v1/memory/tasks/executions` | Record execution |
| 6 | POST | `/api/v1/memory/tasks/recommend-strategy` | Get recommendation |
| 7 | GET | `/api/v1/memory/tasks/summary` | Get task summary |
| 8 | GET | `/api/v1/memory/workflows/` | List workflows |
| 9 | GET | `/api/v1/memory/workflows/{id}` | Get workflow |
| 10 | GET | `/api/v1/memory/workflows/{id}/thread` | Get thread |
| 11 | POST | `/api/v1/memory/workflows/` | Save workflow |
| 12 | DELETE | `/api/v1/memory/workflows/{id}` | Delete workflow |
| 13 | GET | `/api/v1/memory/workflows/search/` | Search workflows |
| 14 | GET | `/api/v1/memory/workflows/analytics/summary` | Get analytics |
| 15 | GET | `/api/v1/memory/knowledge/` | Retrieve knowledge |
| 16 | GET | `/api/v1/memory/knowledge/{id}` | Get entry |
| 17 | POST | `/api/v1/memory/knowledge/` | Store knowledge |
| 18 | POST | `/api/v1/memory/knowledge/{id}/usage` | Record usage |
| 19 | PATCH | `/api/v1/memory/knowledge/{id}/success-rate` | Update rate |
| 20 | GET | `/api/v1/memory/knowledge/{id}/related` | Get related |
| 21 | GET | `/api/v1/memory/knowledge/summary/stats` | Get summary |
| 22 | POST | `/api/v1/memory/compression/compress` | Compress context |
| 23 | GET | `/api/v1/memory/compression/stats` | Get stats |

---

**Last Updated**: 2025-10-31
**API Version**: 0.2.0
