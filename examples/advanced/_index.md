# Advanced Examples

## Purpose
Complex patterns demonstrating Felix's advanced capabilities: autonomous knowledge brain, multi-agent coordination, web search integration, and sophisticated workflows.

## Key Files

### [knowledge_brain_demo.py](knowledge_brain_demo.py)
**Complete Knowledge Brain demonstration with all features**
- **What it demonstrates**:
  - Document ingestion (PDF, TXT, MD, code files)
  - Agentic comprehension using Research/Analysis/Critic agents
  - Three-tier embedding system (LM Studio → TF-IDF → FTS5 BM25)
  - Knowledge graph construction (explicit mentions, similarity, co-occurrence)
  - Semantic retrieval with meta-learning boost
  - Daemon modes (batch, refinement, file watching)
  - Workflow integration with auto-augmentation
- **Prerequisites**:
  - Knowledge Brain enabled: `enable_knowledge_brain: true` in config
  - Optional: `pip install PyPDF2 watchdog` for full features
  - Documents in `knowledge_sources/` directory
- **Run**: `python examples/advanced/knowledge_brain_demo.py`
- **Features demonstrated**:
  - Document processing with chunking (1000 chars, 200 overlap)
  - Multi-agent comprehension for concept extraction
  - Relationship discovery (3 types: mention, similarity 0.75+, co-occurrence 5+ chunks)
  - Semantic search with embedding vectors (768-dim)
  - Meta-learning: tracks which knowledge helps which tasks
  - Autonomous daemon: batch processing, hourly refinement, file watching

## Key Concepts

### Autonomous Knowledge Brain
Felix's knowledge system operates independently:
- **Document reading**: Multi-format support with semantic chunking
- **Agentic comprehension**: Spawns Research/Analysis/Critic agents to understand documents
- **Knowledge graph**: Bidirectional relationships between concepts
- **Semantic retrieval**: Vector similarity + BM25 full-text search
- **Meta-learning**: Learns which knowledge is useful for which task types
- **Zero dependencies**: Operates without external APIs (TF-IDF/FTS5 fallback)

### Three-Tier Embedding System
Intelligent fallback for air-gapped operation:
1. **LM Studio** (768-dim vectors): Best quality, requires local LLM
2. **TF-IDF** (sparse vectors): Good quality, no dependencies
3. **FTS5 BM25**: Full-text search, always available

### Knowledge Graph Construction
Three relationship types:
- **Explicit mentions**: Concept A mentions concept B in same chunk
- **Embedding similarity**: Cosine similarity ≥ 0.75 between concept embeddings
- **Co-occurrence**: Concepts appear together in ≥5 chunks

### Daemon Modes
Three concurrent processing modes:
- **Batch mode**: Process all pending documents immediately
- **Refinement mode**: Re-process old documents every N hours for better comprehension
- **Watch mode**: Monitor directories for new files, process automatically

### Meta-Learning Boost
System tracks knowledge usefulness:
- Records which knowledge entries help complete workflows
- Stores usefulness scores (0.0-1.0) per knowledge-task pair
- Boosts retrieval ranking for historically useful knowledge
- Requires ≥3 samples for reliable boost (0.5-1.0 multiplier)
- Task-type specific: research tasks vs analysis tasks learn separately

## Usage Example

### Full Knowledge Brain Demo
```python
from src.knowledge.knowledge_daemon import KnowledgeDaemon
from src.knowledge.knowledge_retriever import KnowledgeRetriever
from src.workflows.felix_workflow import run_felix_workflow

# 1. Start daemon for autonomous processing
daemon = KnowledgeDaemon(
    knowledge_store,
    document_reader,
    comprehension_engine,
    graph_builder,
    watch_dirs=["./knowledge_sources"],
    refinement_interval_hours=1
)
daemon.start()

# 2. Ingest documents
daemon.enqueue_document("./docs/quantum_computing.pdf")
daemon.process_batch()  # Process immediately

# 3. Query knowledge
retriever = KnowledgeRetriever(knowledge_store, embedding_provider)
results = retriever.retrieve_relevant_knowledge(
    "Explain quantum entanglement",
    task_type="research",
    top_k=5
)

# 4. Run workflow with auto-augmentation
result = run_felix_workflow(
    felix_system,
    "Explain quantum entanglement",
    enable_knowledge_augmentation=True  # Automatically injects relevant knowledge
)

# 5. Knowledge is automatically recorded as helpful
# Next similar query will rank this knowledge higher
```

## Expected Output

```
=== Knowledge Brain Demo ===

Step 1: Document Ingestion
- Reading: quantum_computing.pdf (42 pages)
- Chunked into 87 semantic chunks
- Status: pending_comprehension

Step 2: Agentic Comprehension
- Spawning comprehension agents...
- Research agent analyzing chunk 1/87...
- Analysis agent synthesizing concepts...
- Critic agent validating quality...
- Extracted: 124 concepts, 89 entities, 312 relationships

Step 3: Knowledge Graph Construction
- Building bidirectional relationships...
- Explicit mentions: 156
- Embedding similarity (≥0.75): 98
- Co-occurrence (≥5 chunks): 58
- Total relationships: 312

Step 4: Semantic Retrieval
Query: "quantum entanglement"
Results (with meta-learning boost):
1. Quantum Entanglement Fundamentals (relevance: 0.94, boost: 1.0)
2. EPR Paradox and Bell's Theorem (relevance: 0.87, boost: 0.8)
3. Quantum Superposition Principles (relevance: 0.82, boost: 0.7)
...

Step 5: Workflow Integration
- Task: "Explain quantum entanglement applications"
- Auto-augmented with 5 relevant knowledge entries
- Agents have enhanced context from knowledge base
- Synthesis confidence: 0.91 (vs 0.76 without knowledge)

=== Daemon Status ===
- Mode: batch + refinement + watch
- Documents processed: 15
- Concepts in graph: 1,247
- Relationships: 3,821
- Pending queue: 0
- Last refinement: 2 hours ago
```

## Advanced Patterns

### Pattern 1: Continuous Learning
```python
# Daemon continuously learns from new documents
daemon.start()

# As documents are added, knowledge grows
add_document("new_research_paper.pdf")
# Daemon processes automatically

# Workflows benefit from growing knowledge base
result = run_workflow("Complex task")
# Automatically uses relevant knowledge
```

### Pattern 2: Domain-Specific Knowledge
```python
# Build specialized knowledge bases
daemon.watch_directories = [
    "./legal_docs",     # Legal domain
    "./medical_papers", # Medical domain
    "./code_repos"      # Code documentation
]

# Retrieve domain-specific knowledge
legal_knowledge = retriever.retrieve_relevant_knowledge(
    "Contract analysis",
    domain="legal",
    top_k=10
)
```

### Pattern 3: Meta-Learning Optimization
```python
# System learns from usage patterns
for task in tasks:
    # Execute workflow
    result = run_workflow(task)

    # Record knowledge usefulness
    record_knowledge_usage(
        task_type=task.category,
        knowledge_ids=result.knowledge_used,
        usefulness=result.confidence
    )

# Future workflows benefit from learned patterns
# Knowledge that helped legal research ranks higher for legal tasks
# Knowledge that helped code review ranks higher for code tasks
```

## Next Steps

After completing advanced examples:
1. Explore [api_examples/](../api_examples/) for REST API integration
2. Study [custom_agents/](../custom_agents/) to build specialized agents
3. Review [integrations/](../integrations/) for external system connections

## Related Documentation
- [Knowledge Brain Architecture](../../docs/KNOWLEDGE_BRAIN.md)
- [Meta-Learning Guide](../../docs/META_LEARNING.md)
- [Document Ingestion](../../docs/DOCUMENT_INGESTION.md)
