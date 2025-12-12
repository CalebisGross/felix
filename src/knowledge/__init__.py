"""
Knowledge Brain System for Felix Framework

Autonomous document ingestion, comprehension, and knowledge retrieval system
that builds a queryable "brain" from PDFs, documents, and code.

Key Components:
- document_ingest: Multi-format document readers and chunking
- embeddings: Tiered embedding system (LM Studio → TF-IDF → FTS5)
- comprehension: Agent-based document understanding
- graph_builder: Entity and relationship extraction
- knowledge_daemon: Autonomous background processing
- retrieval: Semantic search and meta-learning
"""

from .document_ingest import (
    DocumentReader,
    DocumentChunk,
    IngestionResult,
    ChunkingStrategy,
    BatchDocumentProcessor
)

from .embeddings import (
    EmbeddingProvider,
    EmbeddingTier,
    serialize_embedding,
    deserialize_embedding
)

from .comprehension import (
    KnowledgeComprehensionEngine,
    ComprehensionResult,
    ConceptExtraction,
    EntityExtraction
)

from .document_outline import (
    DocumentOutlineBuilder,
    DocumentOutline,
    ChunkAnalysis,
    ChunkComplexity,
    ContentType
)

from .basic_extraction import (
    BasicExtractionEngine,
    extract_without_llm
)

from .strategic_comprehension import (
    StrategicComprehensionEngine,
    ProcessingStats
)

from .graph_builder import (
    KnowledgeGraphBuilder,
    ConceptNode,
    RelationshipEdge
)

from .knowledge_daemon import (
    KnowledgeDaemon,
    DaemonConfig,
    DaemonStatus
)

from .retrieval import (
    KnowledgeRetriever,
    SearchResult,
    RetrievalContext
)

from .workflow_integration import (
    KnowledgeBrainIntegration,
    integrate_with_felix_workflow,
    add_knowledge_context_to_workflow_input
)

__all__ = [
    # Document Ingestion
    'DocumentReader',
    'DocumentChunk',
    'IngestionResult',
    'ChunkingStrategy',
    'BatchDocumentProcessor',

    # Embeddings
    'EmbeddingProvider',
    'EmbeddingTier',
    'serialize_embedding',
    'deserialize_embedding',

    # Comprehension
    'KnowledgeComprehensionEngine',
    'ComprehensionResult',
    'ConceptExtraction',
    'EntityExtraction',

    # Document Outline (strategic analysis)
    'DocumentOutlineBuilder',
    'DocumentOutline',
    'ChunkAnalysis',
    'ChunkComplexity',
    'ContentType',

    # Basic Extraction (no LLM)
    'BasicExtractionEngine',
    'extract_without_llm',

    # Strategic Comprehension (optimized processing)
    'StrategicComprehensionEngine',
    'ProcessingStats',

    # Graph Builder
    'KnowledgeGraphBuilder',
    'ConceptNode',
    'RelationshipEdge',

    # Daemon
    'KnowledgeDaemon',
    'DaemonConfig',
    'DaemonStatus',

    # Retrieval
    'KnowledgeRetriever',
    'SearchResult',
    'RetrievalContext',

    # Workflow Integration
    'KnowledgeBrainIntegration',
    'integrate_with_felix_workflow',
    'add_knowledge_context_to_workflow_input',
]
