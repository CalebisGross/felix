# Knowledge Module

## Purpose
Autonomous Knowledge Brain system enabling continuous learning from documents through agentic comprehension, knowledge graph construction, semantic retrieval with meta-learning, and zero-dependency operation via intelligent fallback.

## Key Files

### [document_ingest.py](document_ingest.py)
Multi-format document reading and semantic chunking.
- **`DocumentReader`**: Reads PDF, TXT, MD, Python, JS, Java, C++ files with intelligent chunking
- **`DocumentType`**: Enum for supported file formats
- **`ChunkingStrategy`**: Enum for chunking approaches (SEMANTIC, FIXED_SIZE, SENTENCE)
- **`DocumentMetadata`**: File metadata and processing information
- **`IngestionResult`**: Chunking results with statistics
- **`BatchDocumentProcessor`**: Concurrent document processing

### [comprehension.py](comprehension.py)
Agentic document understanding using Research, Analysis, and Critic agents.
- **`KnowledgeComprehensionEngine`**: Orchestrates multi-agent document analysis
- **`ConceptExtraction`**: Identifies key concepts and their relationships
- **`EntityExtraction`**: Extracts named entities and categorizes them
- **`ComprehensionResult`**: Structured understanding output with concepts, entities, summaries

### [embeddings.py](embeddings.py)
Three-tier embedding system with automatic fallback.
- **`EmbeddingProvider`**: Main interface with mode selection (auto/lm_studio/tfidf/fts5)
- **`LMStudioEmbedder`**: 768-dimensional embeddings via LM Studio (Tier 1)
- **`TFIDFEmbedder`**: TF-IDF vectorization fallback (Tier 2)
- **`FTS5Searcher`**: SQLite FTS5 BM25 ranking fallback (Tier 3)

**Zero External Dependencies**: Operates without cloud APIs through intelligent fallback chain.

### [graph_builder.py](graph_builder.py)
Knowledge graph construction from comprehension results.
- **`KnowledgeGraphBuilder`**: Builds bidirectional concept relationships
- **Relationship Discovery**:
  - Explicit mentions: Direct textual references
  - Embedding similarity: Cosine threshold 0.75
  - Co-occurrence: 5-chunk window tracking

### [retrieval.py](retrieval.py)
Semantic search with meta-learning optimization.
- **`KnowledgeRetriever`**: Context-aware search across knowledge graph
- **`SearchResult`**: Ranked results with relevance scores and source tracking
- **`RetrievalContext`**: Query context for relevance tuning
- **Meta-learning boost**: Tracks historical usefulness to optimize future retrievals

### [knowledge_daemon.py](knowledge_daemon.py)
Autonomous background processor with three concurrent modes.
- **`KnowledgeDaemon`**: Background service for continuous learning
- **Processing Modes**:
  1. **Batch processing**: Initial document ingestion on startup
  2. **Hourly refinement**: Periodic relationship strengthening (configurable interval)
  3. **File watching**: Real-time monitoring of knowledge source directories

### [workflow_integration.py](workflow_integration.py)
Bridge connecting Knowledge Brain to Felix workflows.
- **`WorkflowIntegration`**: Auto-injects relevant knowledge into workflows when enabled
- **Context augmentation**: Enriches agent context with domain-relevant knowledge

## Key Concepts

### Agentic Comprehension
Documents are understood through multi-agent collaboration:
1. **Research Agent**: Extracts concepts and entities
2. **Analysis Agent**: Identifies relationships and themes
3. **Critic Agent**: Validates extracted knowledge

### Three-Tier Embedding Fallback
1. **LM Studio (Primary)**: 768-dim dense vectors for semantic similarity
2. **TF-IDF (Secondary)**: Sparse vectors when LM Studio unavailable
3. **FTS5 (Tertiary)**: Full-text BM25 search when embeddings unavailable

### Semantic Chunking
Documents split into meaningful chunks:
- Default: 1000 characters per chunk
- Overlap: 200 characters between chunks
- Preserves sentence boundaries
- Maintains code block integrity

### Knowledge Graph Relationships
Three relationship types with bidirectional links:
- **explicit_mention**: Confidence 1.0 (direct references)
- **similarity**: Confidence based on embedding cosine similarity (≥0.75)
- **co_occurrence**: Confidence based on frequency (5-chunk window)

### Meta-Learning Boost
Tracks which knowledge helps which workflows:
- Records successful retrievals
- Boosts frequently useful knowledge
- Optimizes search ranking over time

### Autonomous Daemon Modes
1. **Batch**: Process all pending documents on startup
2. **Refinement**: Strengthen relationships hourly (default interval)
3. **Watch**: Monitor directories for new/modified files (requires `watchdog`)

## Configuration

```yaml
knowledge_brain:
  enable_knowledge_brain: false        # Enable system
  knowledge_watch_dirs: ["./knowledge_sources"]
  knowledge_embedding_mode: "auto"     # auto/lm_studio/tfidf/fts5
  knowledge_auto_augment: true         # Auto-inject into workflows
  knowledge_daemon_enabled: true       # Background daemon
  knowledge_refinement_interval: 1     # Hours between refinement
  knowledge_processing_threads: 2      # Concurrent threads
  knowledge_max_memory_mb: 512         # Memory limit
  knowledge_chunk_size: 1000           # Chars per chunk
  knowledge_chunk_overlap: 200         # Overlap between chunks
```

## Optional Dependencies
- **PyPDF2**: PDF reading support (graceful degradation without it)
- **watchdog**: File system monitoring for daemon watch mode

## Related Modules
- [memory/](../memory/) - KnowledgeStore backend for persistence
- [agents/](../agents/) - Research, Analysis, Critic agents for comprehension
- [communication/](../communication/) - CentralPost coordination
- [workflows/](../workflows/) - WorkflowIntegration for context augmentation
- [gui/](../gui/) - Knowledge Brain tab for monitoring and control
