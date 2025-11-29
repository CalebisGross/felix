# Memory Module

## Purpose
Multi-layer persistence systems providing knowledge storage, task memory, workflow history tracking, and context compression with SQLite-based databases.

## Key Files

### [knowledge_store.py](knowledge_store.py)
Persistent knowledge storage with semantic search capabilities.
- **`KnowledgeStore`**: Main interface for storing and retrieving agent insights with embedding support
- **`KnowledgeEntry`**: Structured knowledge records with domain, confidence, timestamps, and source tracking
- **`KnowledgeType`**: Enum defining knowledge categories (FACT, INSIGHT, PATTERN, HYPOTHESIS, etc.)
- **`ConfidenceLevel`**: Enum for confidence classification (LOW, MEDIUM, HIGH, VERY_HIGH)

**Database**: `felix_knowledge.db`
- `knowledge_entries`: Agent insights with embeddings and source references
- `document_sources`: Tracked document ingestion status
- `knowledge_relationships`: Bidirectional concept relationships (explicit, similarity, co-occurrence)
- `knowledge_fts`: FTS5 full-text search index with BM25 ranking
- `knowledge_usage`: Meta-learning tracking for retrieval optimization

### [task_memory.py](task_memory.py)
Pattern storage and recognition for workflow optimization.
- **`TaskMemory`**: Stores task execution patterns and outcomes
- **`TaskPattern`**: Structure for recurring task characteristics
- **`TaskOutcome`**: Success/failure tracking with performance metrics

**Database**: `felix_memory.db`, `felix_task_memory.db`
- `tasks`: Task patterns with timestamps and performance data

### [workflow_history.py](workflow_history.py)
Complete workflow execution tracking and analysis.
- **`WorkflowHistory`**: Records full workflow runs with metrics
- **Methods**: `add_workflow()`, `get_recent_workflows()`, `search_workflows()`, `get_statistics()`

**Database**: `felix_workflow_history.db`
- `workflow_history`: Task description, synthesis output, confidence, agent count, token usage, processing time
- Indexed on `created_at` and `status` for fast queries

### [context_compression.py](context_compression.py)
Memory optimization through abstractive compression.
- **`ContextCompressor`**: Reduces context size while preserving meaning (0.3 compression ratio)
- **`CompressionStrategy`**: Enum for compression approaches (ABSTRACTIVE, EXTRACTIVE, HYBRID)

### [agent_performance_tracker.py](agent_performance_tracker.py)
Agent-level performance metrics and optimization.
- **`AgentPerformanceTracker`**: Tracks individual agent effectiveness, latency, and quality scores

### [audit_log.py](audit_log.py)
Comprehensive audit trail for all CRUD operations on knowledge entries.
- **`AuditLogger`**: Records all knowledge base operations with full context and state tracking
- **`audit_logged`**: Decorator for automatic audit logging of CRUD operations
- **`get_audit_logger()`**: Singleton accessor for audit logger instance
- **Methods**: `log_operation()`, `get_audit_history()`, `get_entry_history()`, `get_recent_changes()`, `get_statistics()`, `cleanup_old_logs()`, `export_to_csv()`
- **Features**: Transaction-level grouping, before/after state capture, query and export capabilities, automatic cleanup of old logs

**Integration**: Works with `knowledge_audit_log` table (requires migration: `add_audit_log_table.py`)

## Key Concepts

### Knowledge Entry Structure
Each entry contains:
- Domain classification
- Confidence score (0.0-1.0)
- Source document reference
- Vector embedding (768-dim when available)
- Chunk index for source tracking
- Creation and update timestamps

### Relationship Types
Knowledge graph connections via three mechanisms:
1. **Explicit mentions**: Direct references between concepts in text
2. **Embedding similarity**: Cosine similarity threshold (0.75)
3. **Co-occurrence**: Concepts appearing together (5-chunk window)

### Auto-Compression
`KnowledgeStore` automatically compresses entries when context exceeds limits using `ContextCompressor` for optimal memory usage.

### Meta-Learning
`knowledge_usage` table tracks which knowledge entries help which workflows, enabling retrieval optimization based on historical usefulness.

### Workflow Metrics
Tracked per execution:
- Task description and synthesis output
- Agent count and token consumption
- Processing time and confidence score
- Status (SUCCESS, PARTIAL, FAILED)
- Full timestamp trail

## Related Modules
- [knowledge/](../knowledge/) - Knowledge Brain system using KnowledgeStore backend
- [communication/](../communication/) - MemoryFacade provides agent access
- [learning/](../learning/) - Pattern learning and recommendations
- [migration/](../migration/) - Schema evolution for memory databases
- [workflows/](../workflows/) - Workflow execution stored in WorkflowHistory
