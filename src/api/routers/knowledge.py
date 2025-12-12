"""
Knowledge Brain API Router

Endpoints for document ingestion, semantic search, knowledge graph operations,
daemon control, and concept browsing.
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, status, Depends, Query

from src.api.models import (
    # Document models
    DocumentIngestRequest, DocumentIngestResponse,
    DocumentBatchRequest, DocumentBatchResponse,
    DocumentListResponse, DocumentDetailResponse,
    DocumentStatus, DocumentListItem, DocumentMetadataModel,
    # Search models
    SearchRequest, SearchResponse, SearchResultItem,
    AugmentRequest, AugmentResponse,
    # Graph models
    GraphBuildRequest, GraphBuildResponse,
    GraphRelationshipsRequest, GraphRelationshipsResponse,
    GraphStatisticsResponse, RelationshipItem,
    # Daemon models
    DaemonStatusResponse, WatchDirectoriesRequest,
    # Concept models
    ConceptListRequest, ConceptListResponse, ConceptDetailResponse,
    RelatedConceptsResponse, ConceptItem, RelatedConceptItem,
)
from src.api.dependencies import (
    verify_api_key,
    get_knowledge_store,
    get_document_reader,
    get_knowledge_retriever,
    get_knowledge_daemon,
    get_graph_builder,
)

logger = logging.getLogger(__name__)

# Router
router = APIRouter(
    prefix="/api/v1/knowledge",
    tags=["Knowledge Brain"],
    responses={404: {"description": "Not found"}, 503: {"description": "Knowledge Brain not enabled"}},
)


# ============================================================================
# Helper Functions
# ============================================================================

def execute_query(knowledge_store, query: str, params: list = None):
    """
    Execute SQL query on knowledge store database.

    Args:
        knowledge_store: KnowledgeStore instance
        query: SQL query string
        params: Query parameters

    Returns:
        List of rows
    """
    import sqlite3
    with sqlite3.connect(knowledge_store.storage_path) as conn:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.fetchall()


# ============================================================================
# Document Endpoints
# ============================================================================

@router.post("/documents/ingest", response_model=DocumentIngestResponse, status_code=202)
async def ingest_document(
    request: DocumentIngestRequest,
    api_key: str = Depends(verify_api_key),
    document_reader=Depends(get_document_reader),
    knowledge_store=Depends(get_knowledge_store)
) -> DocumentIngestResponse:
    """
    Ingest a single document for processing.

    The document will be read, chunked, comprehended, and indexed.
    Returns 202 Accepted as processing may take time.

    Args:
        request: Document ingest request with file_path
        api_key: Authentication token
        document_reader: DocumentReader instance
        knowledge_store: KnowledgeStore instance

    Returns:
        DocumentIngestResponse with status and metadata

    Raises:
        HTTPException: If file not found or processing fails
    """
    try:
        # Check file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {request.file_path}"
            )

        logger.info(f"Ingesting document: {request.file_path}")

        # Ingest document
        from src.knowledge.document_ingest import IngestionResult
        result: IngestionResult = document_reader.ingest_document(request.file_path)

        if not result.success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Document ingestion failed: {result.error_message}"
            )

        # Create response
        response = DocumentIngestResponse(
            document_id=result.document_id,
            file_name=result.metadata.file_name,
            status="processing" if request.process_immediately else "complete",
            chunks_count=len(result.chunks),
            metadata=DocumentMetadataModel(
                file_path=result.metadata.file_path,
                file_name=result.metadata.file_name,
                file_type=result.metadata.file_type,
                file_size=result.metadata.file_size,
                file_hash=result.metadata.file_hash,
                page_count=result.metadata.page_count,
                created_at=datetime.fromtimestamp(result.metadata.created_at),
                updated_at=datetime.fromtimestamp(result.metadata.updated_at)
            ),
            message=f"Document ingested successfully with {len(result.chunks)} chunks"
        )

        logger.info(f"Document {result.document_id} ingested: {len(result.chunks)} chunks")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error ingesting document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting document: {str(e)}"
        )


@router.post("/documents/batch", response_model=DocumentBatchResponse, status_code=202)
async def batch_ingest_documents(
    request: DocumentBatchRequest,
    api_key: str = Depends(verify_api_key),
    document_reader=Depends(get_document_reader)
) -> DocumentBatchResponse:
    """
    Batch ingest documents from a directory.

    Processes multiple documents in a directory (optionally recursive).
    Returns 202 Accepted as processing may take significant time.

    Args:
        request: Batch processing request with directory_path
        api_key: Authentication token
        document_reader: DocumentReader instance

    Returns:
        DocumentBatchResponse with processing summary

    Raises:
        HTTPException: If directory not found or processing fails
    """
    try:
        # Check directory exists
        if not os.path.isdir(request.directory_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Directory not found: {request.directory_path}"
            )

        logger.info(f"Batch processing directory: {request.directory_path}")
        start_time = time.time()

        # Process directory
        from src.knowledge.document_ingest import BatchDocumentProcessor
        processor = BatchDocumentProcessor(document_reader)

        results = processor.process_directory(
            request.directory_path,
            recursive=request.recursive,
            file_patterns=request.file_patterns
        )

        # Count successes/failures
        processed = sum(1 for r in results if r.success)
        failed = len(results) - processed

        # Convert results to responses
        documents = []
        for result in results:
            if result.success:
                documents.append(DocumentIngestResponse(
                    document_id=result.document_id,
                    file_name=result.metadata.file_name,
                    status="complete",
                    chunks_count=len(result.chunks),
                    metadata=DocumentMetadataModel(
                        file_path=result.metadata.file_path,
                        file_name=result.metadata.file_name,
                        file_type=result.metadata.file_type,
                        file_size=result.metadata.file_size,
                        file_hash=result.metadata.file_hash,
                        page_count=result.metadata.page_count,
                        created_at=datetime.fromtimestamp(result.metadata.created_at),
                        updated_at=datetime.fromtimestamp(result.metadata.updated_at)
                    ),
                    message="Success"
                ))

        processing_time = time.time() - start_time

        logger.info(f"Batch processing complete: {processed} successful, {failed} failed")

        return DocumentBatchResponse(
            total_files=len(results),
            processed=processed,
            failed=failed,
            documents=documents,
            processing_time_seconds=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in batch processing")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in batch processing: {str(e)}"
        )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    status_filter: Optional[DocumentStatus] = Query(None, description="Filter by status"),
    api_key: str = Depends(verify_api_key),
    knowledge_store=Depends(get_knowledge_store)
) -> DocumentListResponse:
    """
    List all ingested documents.

    Optionally filter by processing status.

    Args:
        status_filter: Optional status filter (processing/complete/failed)
        api_key: Authentication token
        knowledge_store: KnowledgeStore instance

    Returns:
        DocumentListResponse with list of documents

    Raises:
        HTTPException: If query fails
    """
    try:
        # Query document sources
        query = """
            SELECT doc_id, file_name, file_type, file_size, ingestion_status, added_at
            FROM document_sources
        """
        params = []

        if status_filter:
            # Map status filter values (complete/processing/failed) to ingestion_status
            status_map = {
                "complete": "completed",
                "processing": "processing",
                "failed": "failed"
            }
            query += " WHERE ingestion_status = ?"
            params.append(status_map.get(status_filter.value, status_filter.value))

        query += " ORDER BY added_at DESC"

        rows = execute_query(knowledge_store, query, params)

        # Convert to response models
        documents = []
        for row in rows:
            # Count chunks for this document
            chunks_query = "SELECT COUNT(*) FROM knowledge_entries WHERE source_doc_id = ?"
            chunks_result = execute_query(knowledge_store, chunks_query, [row[0]])
            chunks_count = chunks_result[0][0] if chunks_result else 0

            # Map ingestion_status to DocumentStatus
            status_map = {
                "pending": DocumentStatus.PROCESSING,
                "processing": DocumentStatus.PROCESSING,
                "completed": DocumentStatus.COMPLETE,
                "failed": DocumentStatus.FAILED
            }
            doc_status = status_map.get(row[4], DocumentStatus.PROCESSING)

            documents.append(DocumentListItem(
                document_id=row[0],
                file_name=row[1],
                file_type=row[2],
                file_size=row[3],
                status=doc_status,
                chunks_count=chunks_count if chunks_count > 0 else None,
                created_at=datetime.fromtimestamp(row[5])
            ))

        logger.info(f"Listed {len(documents)} documents")

        return DocumentListResponse(
            documents=documents,
            total=len(documents),
            filtered_by_status=status_filter
        )

    except Exception as e:
        logger.exception("Error listing documents")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/documents/{document_id}", response_model=DocumentDetailResponse)
async def get_document_details(
    document_id: str,
    api_key: str = Depends(verify_api_key),
    knowledge_store=Depends(get_knowledge_store)
) -> DocumentDetailResponse:
    """
    Get detailed information about a document.

    Args:
        document_id: Document ID
        api_key: Authentication token
        knowledge_store: KnowledgeStore instance

    Returns:
        DocumentDetailResponse with full document information

    Raises:
        HTTPException: If document not found
    """
    try:
        # Query document
        query = """
            SELECT doc_id, file_path, file_name, file_type, file_size,
                   file_hash, page_count, ingestion_status, added_at, modified_date
            FROM document_sources
            WHERE doc_id = ?
        """
        rows = execute_query(knowledge_store, query, [document_id])

        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        row = rows[0]

        # Count chunks
        chunks_query = "SELECT COUNT(*) FROM knowledge_entries WHERE source_doc_id = ?"
        chunks_result = execute_query(knowledge_store, chunks_query, [document_id])
        chunks_count = chunks_result[0][0] if chunks_result else 0

        # Count concepts (assume chunks are concepts)
        concepts_extracted = chunks_count

        # Map ingestion_status to DocumentStatus
        status_map = {
            "pending": DocumentStatus.PROCESSING,
            "processing": DocumentStatus.PROCESSING,
            "completed": DocumentStatus.COMPLETE,
            "failed": DocumentStatus.FAILED
        }
        doc_status = status_map.get(row[7], DocumentStatus.PROCESSING)

        return DocumentDetailResponse(
            document_id=row[0],
            metadata=DocumentMetadataModel(
                file_path=row[1],
                file_name=row[2],
                file_type=row[3],
                file_size=row[4],
                file_hash=row[5],
                page_count=row[6],
                created_at=datetime.fromtimestamp(row[8]),
                updated_at=datetime.fromtimestamp(row[9] if row[9] else row[8])
            ),
            status=doc_status,
            chunks_count=chunks_count,
            concepts_extracted=concepts_extracted,
            created_at=datetime.fromtimestamp(row[8]),
            updated_at=datetime.fromtimestamp(row[9] if row[9] else row[8])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting document details")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document details: {str(e)}"
        )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    api_key: str = Depends(verify_api_key),
    knowledge_store=Depends(get_knowledge_store)
) -> dict:
    """
    Delete a document and its associated knowledge entries.

    Args:
        document_id: Document ID
        api_key: Authentication token
        knowledge_store: KnowledgeStore instance

    Returns:
        Success message

    Raises:
        HTTPException: If document not found or deletion fails
    """
    try:
        # Check document exists
        query = "SELECT doc_id FROM document_sources WHERE doc_id = ?"
        rows = execute_query(knowledge_store, query, [document_id])

        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        # Delete knowledge entries and document source
        import sqlite3
        with sqlite3.connect(knowledge_store.storage_path) as conn:
            conn.execute("DELETE FROM knowledge_entries WHERE source_doc_id = ?", [document_id])
            conn.execute("DELETE FROM document_sources WHERE doc_id = ?", [document_id])
            conn.commit()

        logger.info(f"Deleted document: {document_id}")

        return {
            "status": "success",
            "message": f"Document {document_id} deleted successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting document")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


# ============================================================================
# Search Endpoints
# ============================================================================

@router.post("/search", response_model=SearchResponse)
async def search_knowledge(
    request: SearchRequest,
    api_key: str = Depends(verify_api_key),
    retriever=Depends(get_knowledge_retriever)
) -> SearchResponse:
    """
    Semantic search across knowledge base.

    Uses embeddings for semantic matching with optional meta-learning boost.

    Args:
        request: Search request with query and filters
        api_key: Authentication token
        retriever: KnowledgeRetriever instance

    Returns:
        SearchResponse with ranked results

    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info(f"Searching knowledge: {request.query[:100]}")
        start_time = time.time()

        # Perform search
        from src.knowledge.retrieval import RetrievalContext
        context: RetrievalContext = retriever.search(
            query=request.query,
            task_type=request.task_type,
            task_complexity=request.task_complexity,
            top_k=request.top_k,
            min_confidence=request.min_confidence,
            domains=request.domains
        )

        # Convert results
        results = []
        for result in context.results:
            results.append(SearchResultItem(
                knowledge_id=result.knowledge_id,
                content=result.content,
                relevance_score=result.relevance_score,
                confidence=result.confidence,
                domain=result.domain,
                source_document_id=result.source_document_id,
                tags=result.tags
            ))

        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.info(f"Search complete: {len(results)} results in {processing_time:.2f}ms")

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            retrieval_method=context.retrieval_method,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.exception("Error searching knowledge")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching knowledge: {str(e)}"
        )


@router.post("/search/augment", response_model=AugmentResponse)
async def augment_context(
    request: AugmentRequest,
    api_key: str = Depends(verify_api_key),
    retriever=Depends(get_knowledge_retriever)
) -> AugmentResponse:
    """
    Augment task description with relevant knowledge.

    Searches knowledge base and formats results as context for agents.

    Args:
        request: Augmentation request with task description
        api_key: Authentication token
        retriever: KnowledgeRetriever instance

    Returns:
        AugmentResponse with formatted context

    Raises:
        HTTPException: If augmentation fails
    """
    try:
        logger.info(f"Augmenting context for task: {request.task_description[:100]}")

        # Build augmented context
        augmented_context = retriever.build_augmented_context(
            task_description=request.task_description,
            task_type=request.task_type,
            max_concepts=request.max_concepts
        )

        # Count concepts used (rough estimate based on context length)
        concepts_used = min(request.max_concepts, augmented_context.count("###") if augmented_context else 0)

        return AugmentResponse(
            task_description=request.task_description,
            augmented_context=augmented_context,
            concepts_used=concepts_used,
            retrieval_method="embedding"  # Default method
        )

    except Exception as e:
        logger.exception("Error augmenting context")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error augmenting context: {str(e)}"
        )


# ============================================================================
# Knowledge Graph Endpoints
# ============================================================================

@router.post("/graph/build", response_model=GraphBuildResponse)
async def build_knowledge_graph(
    request: GraphBuildRequest,
    api_key: str = Depends(verify_api_key),
    graph_builder=Depends(get_graph_builder)
) -> GraphBuildResponse:
    """
    Build knowledge graph relationships.

    Can build for a specific document or globally across all documents.

    Args:
        request: Graph build request
        api_key: Authentication token
        graph_builder: KnowledgeGraphBuilder instance

    Returns:
        GraphBuildResponse with build statistics

    Raises:
        HTTPException: If build fails
    """
    try:
        logger.info(f"Building knowledge graph: document_id={request.document_id}")
        start_time = time.time()

        relationships_created = 0
        concepts_processed = 0
        documents_processed = None
        entities_linked = None
        concepts_merged = None

        if request.document_id:
            # Build for specific document
            result = graph_builder.build_graph_for_document(
                request.document_id,
                similarity_threshold=request.similarity_threshold
            )
            relationships_created = result.get("relationships_created", 0)
            concepts_processed = result.get("concepts_processed", 0)

        else:
            # Build global graph
            result = graph_builder.build_global_graph(
                max_documents=request.max_documents,
                similarity_threshold=request.similarity_threshold
            )
            relationships_created = result.get("total_relationships", 0)
            concepts_processed = result.get("total_concepts", 0)
            documents_processed = result.get("documents_processed", 0)

            # Link entities and merge duplicates
            entities_linked = graph_builder.link_entities_across_documents()
            concepts_merged = graph_builder.merge_duplicate_concepts(
                similarity_threshold=0.95
            )

        processing_time = time.time() - start_time

        logger.info(f"Graph built: {relationships_created} relationships, {concepts_processed} concepts")

        return GraphBuildResponse(
            relationships_created=relationships_created,
            concepts_processed=concepts_processed,
            documents_processed=documents_processed,
            entities_linked=entities_linked,
            concepts_merged=concepts_merged,
            processing_time_seconds=processing_time
        )

    except Exception as e:
        logger.exception("Error building knowledge graph")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error building knowledge graph: {str(e)}"
        )


@router.post("/graph/relationships", response_model=GraphRelationshipsResponse)
async def get_concept_relationships(
    request: GraphRelationshipsRequest,
    api_key: str = Depends(verify_api_key),
    knowledge_store=Depends(get_knowledge_store)
) -> GraphRelationshipsResponse:
    """
    Get relationships for a concept.

    Retrieves all relationships where the concept is source or target.

    Args:
        request: Relationships request with concept_id
        api_key: Authentication token
        knowledge_store: KnowledgeStore instance

    Returns:
        GraphRelationshipsResponse with relationships

    Raises:
        HTTPException: If concept not found
    """
    try:
        # Get concept
        concept_query = """
            SELECT knowledge_id, content_json
            FROM knowledge_entries
            WHERE knowledge_id = ?
        """
        concept_rows = execute_query(knowledge_store, concept_query, [request.concept_id])

        if not concept_rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept not found: {request.concept_id}"
            )

        import json
        concept_content = json.loads(concept_rows[0][1])
        concept_text = concept_content.get("content", "")[:200]

        # Get relationships
        relationships_query = """
            SELECT source_id, target_id, relationship_type, strength, basis
            FROM knowledge_relationships
            WHERE (source_id = ? OR target_id = ?)
              AND strength >= ?
            ORDER BY strength DESC
        """
        rel_rows = execute_query(
            knowledge_store,
            relationships_query,
            [request.concept_id, request.concept_id, request.min_strength]
        )

        # Build relationship list
        relationships = []
        for row in rel_rows:
            source_id, target_id, rel_type, strength, basis = row

            # Get the other concept's content
            other_id = target_id if source_id == request.concept_id else source_id
            other_query = "SELECT content_json FROM knowledge_entries WHERE knowledge_id = ?"
            other_rows = execute_query(knowledge_store, other_query, [other_id])

            if other_rows:
                other_content = json.loads(other_rows[0][0])
                other_text = other_content.get("content", "")[:200]

                relationships.append(RelationshipItem(
                    source_id=source_id,
                    source_content=concept_text if source_id == request.concept_id else other_text,
                    target_id=target_id,
                    target_content=other_text if target_id != request.concept_id else concept_text,
                    relationship_type=rel_type,
                    strength=strength,
                    basis=basis
                ))

        logger.info(f"Found {len(relationships)} relationships for concept {request.concept_id}")

        return GraphRelationshipsResponse(
            concept_id=request.concept_id,
            concept_content=concept_text,
            relationships=relationships,
            total_relationships=len(relationships)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting relationships")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting relationships: {str(e)}"
        )


@router.get("/graph/statistics", response_model=GraphStatisticsResponse)
async def get_graph_statistics(
    api_key: str = Depends(verify_api_key),
    graph_builder=Depends(get_graph_builder)
) -> GraphStatisticsResponse:
    """
    Get knowledge graph statistics.

    Returns overview of graph size, connectivity, and coverage.

    Args:
        api_key: Authentication token
        graph_builder: KnowledgeGraphBuilder instance

    Returns:
        GraphStatisticsResponse with statistics

    Raises:
        HTTPException: If query fails
    """
    try:
        stats = graph_builder.get_graph_statistics()

        return GraphStatisticsResponse(
            total_nodes=stats.get("total_nodes", 0),
            total_relationships=stats.get("total_relationships", 0),
            nodes_with_relationships=stats.get("nodes_with_relationships", 0),
            average_degree=stats.get("average_degree", 0.0),
            documents_covered=stats.get("documents_covered", 0),
            relationship_types=stats.get("relationship_types", {})
        )

    except Exception as e:
        logger.exception("Error getting graph statistics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting graph statistics: {str(e)}"
        )


# ============================================================================
# Daemon Control Endpoints
# ============================================================================

@router.post("/daemon/start")
async def start_daemon(
    api_key: str = Depends(verify_api_key),
    daemon=Depends(get_knowledge_daemon)
) -> dict:
    """
    Start the knowledge daemon for autonomous processing.

    The daemon will monitor watch directories and process documents automatically.

    Args:
        api_key: Authentication token
        daemon: KnowledgeDaemon instance

    Returns:
        Success message with timestamp

    Raises:
        HTTPException: If daemon already running or start fails
    """
    try:
        if daemon.is_running():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Knowledge daemon is already running"
            )

        logger.info("Starting knowledge daemon...")
        daemon.start()

        return {
            "status": "started",
            "message": "Knowledge daemon started successfully",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error starting daemon")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting daemon: {str(e)}"
        )


@router.post("/daemon/stop")
async def stop_daemon(
    api_key: str = Depends(verify_api_key),
    daemon=Depends(get_knowledge_daemon)
) -> dict:
    """
    Stop the knowledge daemon.

    Gracefully shuts down all daemon threads.

    Args:
        api_key: Authentication token
        daemon: KnowledgeDaemon instance

    Returns:
        Success message with final statistics

    Raises:
        HTTPException: If daemon not running or stop fails
    """
    try:
        if not daemon.is_running():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Knowledge daemon is not running"
            )

        logger.info("Stopping knowledge daemon...")
        status_before = daemon.get_status()
        daemon.stop()

        return {
            "status": "stopped",
            "message": "Knowledge daemon stopped successfully",
            "final_stats": {
                "documents_processed": status_before.documents_processed,
                "uptime_seconds": status_before.uptime_seconds
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error stopping daemon")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error stopping daemon: {str(e)}"
        )


@router.get("/daemon/status", response_model=DaemonStatusResponse)
async def get_daemon_status(
    api_key: str = Depends(verify_api_key),
    daemon=Depends(get_knowledge_daemon)
) -> DaemonStatusResponse:
    """
    Get current daemon status.

    Returns detailed status including activity metrics.

    Args:
        api_key: Authentication token
        daemon: KnowledgeDaemon instance

    Returns:
        DaemonStatusResponse with current status

    Raises:
        HTTPException: If status query fails
    """
    try:
        status = daemon.get_status()

        return DaemonStatusResponse(
            running=status.running,
            batch_processor_active=status.batch_processor_active,
            refiner_active=status.refiner_active,
            file_watcher_active=status.file_watcher_active,
            documents_processed=status.documents_processed,
            documents_pending=status.documents_pending,
            documents_failed=status.documents_failed,
            current_activity=status.current_activity,
            uptime_seconds=status.uptime_seconds,
            watch_directories=status.watch_directories
        )

    except Exception as e:
        logger.exception("Error getting daemon status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting daemon status: {str(e)}"
        )


@router.put("/daemon/watch-dirs")
async def update_watch_directories(
    request: WatchDirectoriesRequest,
    api_key: str = Depends(verify_api_key),
    daemon=Depends(get_knowledge_daemon)
) -> dict:
    """
    Update watched directories for file monitoring.

    Changes take effect immediately if daemon is running.

    Args:
        request: Watch directories request
        api_key: Authentication token
        daemon: KnowledgeDaemon instance

    Returns:
        Success message

    Raises:
        HTTPException: If update fails
    """
    try:
        # Validate directories exist
        for directory in request.directories:
            if not os.path.isdir(directory):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Directory not found: {directory}"
                )

        daemon.set_watch_directories(request.directories)

        logger.info(f"Updated watch directories: {request.directories}")

        return {
            "status": "success",
            "message": f"Watch directories updated to {len(request.directories)} paths",
            "directories": request.directories
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating watch directories")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating watch directories: {str(e)}"
        )


# ============================================================================
# Concept Browsing Endpoints
# ============================================================================

@router.post("/concepts", response_model=ConceptListResponse)
async def list_concepts(
    request: ConceptListRequest,
    api_key: str = Depends(verify_api_key),
    knowledge_store=Depends(get_knowledge_store)
) -> ConceptListResponse:
    """
    List knowledge concepts with optional filtering.

    Supports filtering by domain, search query, and confidence.

    Args:
        request: Concept list request with filters
        api_key: Authentication token
        knowledge_store: KnowledgeStore instance

    Returns:
        ConceptListResponse with paginated results

    Raises:
        HTTPException: If query fails
    """
    try:
        # Build query
        query = """
            SELECT knowledge_id, content_json, confidence_level, domain, tags_json, source_doc_id
            FROM knowledge_entries
            WHERE 1=1
        """
        params = []

        if request.domain:
            query += " AND domain = ?"
            params.append(request.domain)

        if request.min_confidence:
            confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.8, "verified": 0.9}
            query += " AND confidence_level IN ("
            valid_levels = [k for k, v in confidence_map.items() if v >= request.min_confidence]
            query += ",".join("?" * len(valid_levels)) + ")"
            params.extend(valid_levels)

        if request.search_query:
            query += " AND (content_json LIKE ? OR tags_json LIKE ?)"
            search_pattern = f"%{request.search_query}%"
            params.extend([search_pattern, search_pattern])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([request.limit, request.offset])

        # Execute query
        rows = execute_query(knowledge_store, query, params)

        # Parse results
        import json
        concepts = []
        for row in rows:
            content = json.loads(row[1])
            tags = json.loads(row[4]) if row[4] else []

            concepts.append(ConceptItem(
                knowledge_id=row[0],
                concept_name=content.get("concept_name", "Unknown"),
                definition=content.get("definition", content.get("content", ""))[:500],
                confidence={"low": 0.3, "medium": 0.6, "high": 0.8, "verified": 0.9}.get(row[2], 0.5),
                domain=row[3],
                tags=tags,
                source_document_id=row[5]
            ))

        # Get total count
        count_query = "SELECT COUNT(*) FROM knowledge_entries WHERE 1=1"
        if request.domain:
            count_query += f" AND domain = '{request.domain}'"
        count_result = execute_query(knowledge_store, count_query)
        total = count_result[0][0] if count_result else 0

        logger.info(f"Listed {len(concepts)} concepts (total: {total})")

        return ConceptListResponse(
            concepts=concepts,
            total=total,
            offset=request.offset,
            limit=request.limit
        )

    except Exception as e:
        logger.exception("Error listing concepts")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing concepts: {str(e)}"
        )


@router.get("/concepts/{knowledge_id}", response_model=ConceptDetailResponse)
async def get_concept_details(
    knowledge_id: str,
    api_key: str = Depends(verify_api_key),
    knowledge_store=Depends(get_knowledge_store)
) -> ConceptDetailResponse:
    """
    Get detailed concept information.

    Args:
        knowledge_id: Knowledge entry ID
        api_key: Authentication token
        knowledge_store: KnowledgeStore instance

    Returns:
        ConceptDetailResponse with full concept data

    Raises:
        HTTPException: If concept not found
    """
    try:
        query = """
            SELECT knowledge_id, content_json, confidence_level, domain,
                   tags_json, source_doc_id, related_entries_json,
                   access_count, created_at
            FROM knowledge_entries
            WHERE knowledge_id = ?
        """
        rows = execute_query(knowledge_store, query, [knowledge_id])

        if not rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept not found: {knowledge_id}"
            )

        import json
        row = rows[0]
        content = json.loads(row[1])
        tags = json.loads(row[4]) if row[4] else []
        related_ids = json.loads(row[6]) if row[6] else []

        return ConceptDetailResponse(
            knowledge_id=row[0],
            concept_name=content.get("concept_name", "Unknown"),
            definition=content.get("definition", content.get("content", "")),
            confidence={"low": 0.3, "medium": 0.6, "high": 0.8, "verified": 0.9}.get(row[2], 0.5),
            domain=row[3],
            tags=tags,
            examples=content.get("examples", []),
            source_document_id=row[5],
            related_concept_ids=related_ids,
            access_count=row[7],
            created_at=datetime.fromtimestamp(row[8])
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting concept details")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting concept details: {str(e)}"
        )


@router.get("/concepts/{knowledge_id}/related", response_model=RelatedConceptsResponse)
async def get_related_concepts(
    knowledge_id: str,
    api_key: str = Depends(verify_api_key),
    knowledge_store=Depends(get_knowledge_store)
) -> RelatedConceptsResponse:
    """
    Get related concepts for a given concept.

    Uses knowledge graph relationships.

    Args:
        knowledge_id: Knowledge entry ID
        api_key: Authentication token
        knowledge_store: KnowledgeStore instance

    Returns:
        RelatedConceptsResponse with related concepts

    Raises:
        HTTPException: If concept not found
    """
    try:
        # Get main concept
        main_query = """
            SELECT knowledge_id, content_json
            FROM knowledge_entries
            WHERE knowledge_id = ?
        """
        main_rows = execute_query(knowledge_store, main_query, [knowledge_id])

        if not main_rows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept not found: {knowledge_id}"
            )

        import json
        main_content = json.loads(main_rows[0][1])
        concept_name = main_content.get("concept_name", "Unknown")

        # Get relationships
        rel_query = """
            SELECT target_id, relationship_type, strength
            FROM knowledge_relationships
            WHERE source_id = ?
            ORDER BY strength DESC
            LIMIT 20
        """
        rel_rows = execute_query(knowledge_store, rel_query, [knowledge_id])

        # Get related concept details
        related_concepts = []
        for rel_row in rel_rows:
            target_id, rel_type, strength = rel_row

            target_query = "SELECT content_json FROM knowledge_entries WHERE knowledge_id = ?"
            target_rows = execute_query(knowledge_store, target_query, [target_id])

            if target_rows:
                target_content = json.loads(target_rows[0][0])
                related_concepts.append(RelatedConceptItem(
                    knowledge_id=target_id,
                    concept_name=target_content.get("concept_name", "Unknown"),
                    definition=target_content.get("definition", target_content.get("content", ""))[:200],
                    relationship_type=rel_type,
                    strength=strength
                ))

        logger.info(f"Found {len(related_concepts)} related concepts for {knowledge_id}")

        return RelatedConceptsResponse(
            concept_id=knowledge_id,
            concept_name=concept_name,
            related_concepts=related_concepts,
            total=len(related_concepts)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting related concepts")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting related concepts: {str(e)}"
        )
