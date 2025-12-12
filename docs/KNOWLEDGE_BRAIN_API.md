# Felix Knowledge Brain API - Complete Guide

## Overview

The Felix Knowledge Brain API provides programmatic access to Felix's autonomous document comprehension and knowledge management system. It enables document ingestion, semantic search, knowledge graph exploration, and autonomous processing through a daemon.

### Key Features

- **Document Ingestion**: Process PDFs, text, markdown, and code files
- **Agentic Comprehension**: Multi-stage agent pipeline for document understanding
- **Semantic Search**: Embedding-based search with meta-learning boost
- **Knowledge Graph**: Automatic relationship discovery between concepts
- **Autonomous Daemon**: Background processing with file watching
- **Zero External Dependencies**: Intelligent fallback (LM Studio → TF-IDF → FTS5)

### Architecture

```
Documents → DocumentReader → Chunks
                              ↓
Chunks → KnowledgeComprehensionEngine → Concepts + Entities
                                        ↓
Concepts → EmbeddingProvider → Vector Embeddings
                               ↓
Embeddings + Concepts → KnowledgeStore (SQLite)
                        ↓
KnowledgeGraphBuilder → Relationships
                        ↓
KnowledgeRetriever ← Search Queries
```

## Getting Started

### 1. Enable Knowledge Brain

```bash
export FELIX_ENABLE_KNOWLEDGE_BRAIN=true
export FELIX_KNOWLEDGE_WATCH_DIRS="./knowledge_sources"
```

### 2. Start Felix API

```bash
# Install dependencies
pip install -r requirements-api.txt

# Start server
python -m uvicorn src.api.main:app --reload --port 8000
```

### 3. Verify It's Working

```bash
curl http://localhost:8000/health
```

## API Endpoints

### Document Management (5 endpoints)

#### POST /api/v1/knowledge/documents/ingest

Ingest a single document for processing.

**Request:**
```json
{
  "file_path": "/path/to/document.pdf",
  "process_immediately": true
}
```

**Response (202 Accepted):**
```json
{
  "document_id": "doc_abc123def456",
  "file_name": "document.pdf",
  "status": "processing",
  "chunks_count": 42,
  "metadata": {
    "file_path": "/path/to/document.pdf",
    "file_name": "document.pdf",
    "file_type": "pdf",
    "file_size": 1048576,
    "file_hash": "sha256_hash",
    "page_count": 10,
    "created_at": "2025-10-30T10:00:00Z",
    "updated_at": "2025-10-30T10:00:00Z"
  },
  "message": "Document ingested successfully with 42 chunks"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/v1/knowledge/documents/ingest \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/docs/quantum_computing.pdf", "process_immediately": true}'
```

**Python:**
```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/knowledge/documents/ingest",
    headers={"Authorization": "Bearer your-key"},
    json={"file_path": "/docs/quantum_computing.pdf"}
)
result = response.json()
print(f"Document {result['document_id']} ingested with {result['chunks_count']} chunks")
```

---

#### POST /api/v1/knowledge/documents/batch

Batch process documents from a directory.

**Request:**
```json
{
  "directory_path": "/path/to/documents",
  "recursive": true,
  "file_patterns": ["*.pdf", "*.txt", "*.md"]
}
```

**Response (202 Accepted):**
```json
{
  "total_files": 25,
  "processed": 23,
  "failed": 2,
  "documents": [
    {
      "document_id": "doc_123",
      "file_name": "file1.pdf",
      "status": "complete",
      "chunks_count": 15,
      "metadata": {...},
      "message": "Success"
    }
  ],
  "processing_time_seconds": 45.3
}
```

---

#### GET /api/v1/knowledge/documents

List all ingested documents with optional status filter.

**Query Parameters:**
- `status_filter`: Optional (processing/complete/failed)

**Response:**
```json
{
  "documents": [
    {
      "document_id": "doc_123",
      "file_name": "quantum.pdf",
      "file_type": "pdf",
      "file_size": 1048576,
      "status": "complete",
      "chunks_count": 42,
      "created_at": "2025-10-30T10:00:00Z"
    }
  ],
  "total": 150,
  "filtered_by_status": "complete"
}
```

---

#### GET /api/v1/knowledge/documents/{document_id}

Get detailed information about a specific document.

**Response:**
```json
{
  "document_id": "doc_123",
  "metadata": {
    "file_path": "/docs/quantum.pdf",
    "file_name": "quantum.pdf",
    "file_type": "pdf",
    "file_size": 1048576,
    "file_hash": "sha256_hash",
    "page_count": 10,
    "created_at": "2025-10-30T10:00:00Z",
    "updated_at": "2025-10-30T10:05:00Z"
  },
  "status": "complete",
  "chunks_count": 42,
  "concepts_extracted": 42,
  "created_at": "2025-10-30T10:00:00Z",
  "updated_at": "2025-10-30T10:05:00Z"
}
```

---

#### DELETE /api/v1/knowledge/documents/{document_id}

Delete a document and all associated knowledge entries.

**Response:**
```json
{
  "status": "success",
  "message": "Document doc_123 deleted successfully"
}
```

---

### Search & Retrieval (2 endpoints)

#### POST /api/v1/knowledge/search

Semantic search across the knowledge base with optional meta-learning boost.

**Request:**
```json
{
  "query": "quantum computing fundamentals",
  "task_type": "research",
  "task_complexity": "medium",
  "top_k": 10,
  "min_confidence": 0.7,
  "domains": ["physics", "computer_science"]
}
```

**Response:**
```json
{
  "query": "quantum computing fundamentals",
  "results": [
    {
      "knowledge_id": "know_abc123",
      "content": "Quantum computing utilizes quantum mechanical phenomena...",
      "relevance_score": 0.92,
      "confidence": 0.85,
      "domain": "physics",
      "source_document_id": "doc_123",
      "tags": ["quantum", "computing", "superposition"]
    }
  ],
  "total_results": 10,
  "retrieval_method": "embedding",
  "processing_time_ms": 45.2
}
```

**Example:**
```python
response = httpx.post(
    "http://localhost:8000/api/v1/knowledge/search",
    headers={"Authorization": "Bearer your-key"},
    json={
        "query": "machine learning algorithms",
        "top_k": 10,
        "domains": ["computer_science"]
    }
)

for i, result in enumerate(response.json()["results"], 1):
    print(f"{i}. Score: {result['relevance_score']:.3f}")
    print(f"   {result['content'][:100]}...")
```

---

#### POST /api/v1/knowledge/search/augment

Get augmented context formatted for agent consumption.

**Request:**
```json
{
  "task_description": "Write a report on renewable energy",
  "task_type": "analysis",
  "max_concepts": 10
}
```

**Response:**
```json
{
  "task_description": "Write a report on renewable energy",
  "augmented_context": "### Relevant Knowledge\n\n#### Concept 1...",
  "concepts_used": 8,
  "retrieval_method": "embedding"
}
```

**Use Case:**
```python
# Get augmented context before workflow
augment_response = httpx.post(
    "http://localhost:8000/api/v1/knowledge/search/augment",
    headers={"Authorization": "Bearer your-key"},
    json={"task_description": "Explain quantum entanglement", "max_concepts": 5}
)

context = augment_response.json()["augmented_context"]

# Include in workflow
workflow_response = httpx.post(
    "http://localhost:8000/api/v1/workflows",
    headers={"Authorization": "Bearer your-key"},
    json={"task": f"Task: Explain quantum entanglement\n\n{context}"}
)
```

---

### Knowledge Graph (3 endpoints)

#### POST /api/v1/knowledge/graph/build

Build knowledge graph relationships.

**Request:**
```json
{
  "document_id": null,
  "max_documents": 100,
  "similarity_threshold": 0.75
}
```

- `document_id`: Build graph for specific document, or `null` for global
- `max_documents`: Max documents for global graph (null = all)
- `similarity_threshold`: Minimum similarity for relationships (0.0-1.0)

**Response:**
```json
{
  "relationships_created": 1250,
  "concepts_processed": 500,
  "documents_processed": 50,
  "entities_linked": 120,
  "concepts_merged": 15,
  "processing_time_seconds": 12.5
}
```

**Example:**
```bash
# Build global graph
curl -X POST http://localhost:8000/api/v1/knowledge/graph/build \
  -H "Authorization: Bearer your-key" \
  -H "Content-Type: application/json" \
  -d '{"document_id": null, "similarity_threshold": 0.75}'
```

---

#### POST /api/v1/knowledge/graph/relationships

Get relationships for a specific concept.

**Request:**
```json
{
  "concept_id": "know_abc123",
  "max_depth": 1,
  "min_strength": 0.5
}
```

**Response:**
```json
{
  "concept_id": "know_abc123",
  "concept_content": "Quantum superposition is the principle...",
  "relationships": [
    {
      "source_id": "know_abc123",
      "source_content": "Quantum superposition...",
      "target_id": "know_def456",
      "target_content": "Wave-particle duality...",
      "relationship_type": "related_to",
      "strength": 0.87,
      "basis": "embedding_similarity"
    }
  ],
  "total_relationships": 15
}
```

---

#### GET /api/v1/knowledge/graph/statistics

Get overall knowledge graph statistics.

**Response:**
```json
{
  "total_nodes": 500,
  "total_relationships": 1250,
  "nodes_with_relationships": 450,
  "average_degree": 2.5,
  "documents_covered": 50,
  "relationship_types": {
    "related_to": 800,
    "similar_to": 250,
    "cooccurs_with": 150,
    "prerequisite_of": 50
  }
}
```

---

### Daemon Control (4 endpoints)

#### POST /api/v1/knowledge/daemon/start

Start the autonomous knowledge daemon.

The daemon runs three concurrent modes:
1. **Batch Processor**: Initial document ingestion
2. **Continuous Refiner**: Periodic re-analysis
3. **File Watcher**: Monitor directories for new files

**Response:**
```json
{
  "status": "started",
  "message": "Knowledge daemon started successfully",
  "timestamp": "2025-10-30T10:00:00Z"
}
```

---

#### POST /api/v1/knowledge/daemon/stop

Stop the knowledge daemon gracefully.

**Response:**
```json
{
  "status": "stopped",
  "message": "Knowledge daemon stopped successfully",
  "final_stats": {
    "documents_processed": 150,
    "uptime_seconds": 3600
  },
  "timestamp": "2025-10-30T11:00:00Z"
}
```

---

#### GET /api/v1/knowledge/daemon/status

Get current daemon status and activity metrics.

**Response:**
```json
{
  "running": true,
  "batch_processor_active": true,
  "refiner_active": true,
  "file_watcher_active": true,
  "documents_processed": 150,
  "documents_pending": 5,
  "documents_failed": 2,
  "current_activity": "Processing document quantum_physics.pdf",
  "uptime_seconds": 3600,
  "watch_directories": ["/docs/source1", "/docs/source2"]
}
```

---

#### PUT /api/v1/knowledge/daemon/watch-dirs

Update watched directories for file monitoring.

**Request:**
```json
{
  "directories": ["/docs/source1", "/docs/source2", "/docs/source3"]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Watch directories updated to 3 paths",
  "directories": ["/docs/source1", "/docs/source2", "/docs/source3"]
}
```

---

### Concept Browsing (3 endpoints)

#### POST /api/v1/knowledge/concepts

List concepts with filtering and pagination.

**Request:**
```json
{
  "domain": "physics",
  "search_query": "quantum",
  "min_confidence": 0.7,
  "limit": 50,
  "offset": 0
}
```

**Response:**
```json
{
  "concepts": [
    {
      "knowledge_id": "know_abc123",
      "concept_name": "Quantum Superposition",
      "definition": "The principle that quantum systems can exist in...",
      "confidence": 0.85,
      "domain": "physics",
      "tags": ["quantum", "superposition", "physics"],
      "source_document_id": "doc_123"
    }
  ],
  "total": 200,
  "offset": 0,
  "limit": 50
}
```

---

#### GET /api/v1/knowledge/concepts/{knowledge_id}

Get detailed concept information.

**Response:**
```json
{
  "knowledge_id": "know_abc123",
  "concept_name": "Quantum Superposition",
  "definition": "The principle that quantum systems can exist in multiple states simultaneously...",
  "confidence": 0.85,
  "domain": "physics",
  "tags": ["quantum", "superposition", "physics"],
  "examples": [
    "Schrödinger's cat thought experiment",
    "Double-slit experiment"
  ],
  "source_document_id": "doc_123",
  "related_concept_ids": ["know_def456", "know_ghi789"],
  "access_count": 42,
  "created_at": "2025-10-30T10:00:00Z"
}
```

---

#### GET /api/v1/knowledge/concepts/{knowledge_id}/related

Get related concepts for a given concept.

**Response:**
```json
{
  "concept_id": "know_abc123",
  "concept_name": "Quantum Superposition",
  "related_concepts": [
    {
      "knowledge_id": "know_def456",
      "concept_name": "Wave-Particle Duality",
      "definition": "The concept that particles exhibit wave-like properties...",
      "relationship_type": "related_to",
      "strength": 0.87
    }
  ],
  "total": 15
}
```

---

## Workflows

### Complete Document Processing Workflow

```python
import httpx
import time

API_URL = "http://localhost:8000"
headers = {"Authorization": "Bearer your-key"}

# 1. Start daemon for autonomous processing
httpx.post(f"{API_URL}/api/v1/knowledge/daemon/start", headers=headers)

# 2. Batch ingest documents
batch_response = httpx.post(
    f"{API_URL}/api/v1/knowledge/documents/batch",
    headers=headers,
    json={
        "directory_path": "/docs/research_papers",
        "recursive": True,
        "file_patterns": ["*.pdf"]
    }
)
print(f"Processed: {batch_response.json()['processed']} documents")

# 3. Wait for processing (or check daemon status periodically)
time.sleep(10)

# 4. Build knowledge graph
graph_response = httpx.post(
    f"{API_URL}/api/v1/knowledge/graph/build",
    headers=headers,
    json={"document_id": null, "similarity_threshold": 0.75}
)
print(f"Graph: {graph_response.json()['relationships_created']} relationships")

# 5. Search knowledge
search_response = httpx.post(
    f"{API_URL}/api/v1/knowledge/search",
    headers=headers,
    json={"query": "quantum algorithms", "top_k": 10}
)

for result in search_response.json()["results"]:
    print(f"- {result['content'][:100]}... (score: {result['relevance_score']:.3f})")
```

### Augmented Workflow Execution

```python
# Get relevant knowledge before running workflow
augment_response = httpx.post(
    f"{API_URL}/api/v1/knowledge/search/augment",
    headers=headers,
    json={
        "task_description": "Explain quantum entanglement for beginners",
        "max_concepts": 5
    }
)

context = augment_response.json()["augmented_context"]

# Run workflow with augmented context
workflow_response = httpx.post(
    f"{API_URL}/api/v1/workflows",
    headers=headers,
    json={
        "task": f"""Create an explanation of quantum entanglement for beginners.

{context}

Please synthesize this knowledge into a clear explanation.""",
        "max_steps": 10
    }
)

workflow_id = workflow_response.json()["workflow_id"]
```

---

## Configuration

### Environment Variables

```bash
# Enable Knowledge Brain
FELIX_ENABLE_KNOWLEDGE_BRAIN=true

# Watch directories for daemon (comma-separated)
FELIX_KNOWLEDGE_WATCH_DIRS="./knowledge_sources,./docs"

# Embedding mode (auto/lm_studio/tfidf/fts5)
FELIX_KNOWLEDGE_EMBEDDING_MODE=auto

# Daemon settings
FELIX_KNOWLEDGE_REFINEMENT_INTERVAL=3600  # seconds
FELIX_KNOWLEDGE_PROCESSING_THREADS=2
FELIX_KNOWLEDGE_MAX_MEMORY_MB=512
FELIX_KNOWLEDGE_FILE_WATCHER=true

# Chunking settings
FELIX_KNOWLEDGE_CHUNK_SIZE=1000  # characters
FELIX_KNOWLEDGE_CHUNK_OVERLAP=200  # characters
```

---

## Best Practices

### 1. Document Ingestion

**DO:**
- Use batch ingestion for multiple documents
- Process documents during off-peak hours
- Monitor daemon status during large ingests
- Use appropriate file patterns to filter files

**DON'T:**
- Ingest very large files (>100MB) without testing
- Process duplicate documents repeatedly
- Ignore failed document status

### 2. Search Optimization

**DO:**
- Use specific queries for better results
- Filter by domain when you know the topic
- Set appropriate `min_confidence` thresholds
- Use `task_type` for meta-learning boost

**DON'T:**
- Set `top_k` too high unnecessarily
- Use very generic queries without filters
- Ignore relevance scores in results

### 3. Knowledge Graph

**DO:**
- Build graph after ingesting multiple documents
- Use reasonable similarity thresholds (0.7-0.8)
- Rebuild periodically as knowledge grows
- Explore relationships to discover connections

**DON'T:**
- Build graph after every document
- Use very low similarity thresholds (<0.5)
- Skip graph building for related documents

### 4. Daemon Management

**DO:**
- Start daemon for continuous processing
- Monitor status regularly
- Set reasonable watch directories
- Check for failed documents

**DON'T:**
- Leave daemon running with no documents to process
- Watch system directories or too many paths
- Ignore daemon errors

---

## Troubleshooting

### Knowledge Brain Not Available

**Problem**: `503 Service Unavailable - Knowledge Brain not enabled`

**Solution**:
```bash
export FELIX_ENABLE_KNOWLEDGE_BRAIN=true
# Restart API server
python -m uvicorn src.api.main:app --reload --port 8000
```

### Document Ingestion Fails

**Problem**: `500 Internal Server Error` during document ingestion

**Common Causes**:
1. File doesn't exist or wrong path
2. Unsupported file format
3. File is corrupted or locked

**Solution**:
- Verify file exists: `ls -lh /path/to/document.pdf`
- Check file type is supported: PDF, TXT, MD, PY, JS, JAVA, CPP, C
- Check API logs for detailed error

### Search Returns No Results

**Problem**: Search returns 0 results despite having documents

**Possible Causes**:
1. Documents not yet processed
2. Embeddings not generated
3. Query doesn't match content

**Solution**:
- Check document status: `GET /api/v1/knowledge/documents`
- Wait for processing to complete
- Try broader search query
- Check embedding tier: Should show "embedding" or "tfidf" in retrieval_method

### Daemon Won't Start

**Problem**: `409 Conflict - daemon already running` or fails to start

**Solution**:
```python
# Check status first
status = httpx.get(f"{API_URL}/api/v1/knowledge/daemon/status").json()

if status["running"]:
    # Stop existing daemon
    httpx.post(f"{API_URL}/api/v1/knowledge/daemon/stop")
    time.sleep(2)

# Start fresh
httpx.post(f"{API_URL}/api/v1/knowledge/daemon/start")
```

---

## Example Clients

### Python Client

Complete example client: `examples/api_examples/knowledge_brain_client.py`

```bash
python examples/api_examples/knowledge_brain_client.py
```

### Browser Client

Interactive HTML client: `examples/api_examples/knowledge_brain_demo.html`

```bash
# Start API server
python -m uvicorn src.api.main:app --port 8000

# Open in browser
open examples/api_examples/knowledge_brain_demo.html
```

---

## Support

- **API Documentation**: http://localhost:8000/docs
- **Interactive Testing**: http://localhost:8000/redoc
- **Example Clients**: `examples/api_examples/`
- **Main Documentation**: `CLAUDE.md`

For more information on the Knowledge Brain system architecture, see the Knowledge Brain section in `CLAUDE.md`.
