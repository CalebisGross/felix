"""
Knowledge Memory API router.

Endpoints for accessing agent insights, task results, and knowledge entries
stored in the knowledge store memory system.
"""

import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query

from src.api.dependencies import verify_api_key, get_knowledge_store_memory
from src.api.models import (
    # Request models
    KnowledgeQueryRequest,
    KnowledgeStoreRequest,
    KnowledgeUsageRequest,
    KnowledgeSuccessRateRequest,
    # Response models
    KnowledgeEntryModel,
    KnowledgeListResponse,
    KnowledgeStoreResponse,
    KnowledgeUsageResponse,
    KnowledgeRelationshipItem,
    KnowledgeRelationshipsResponse,
    KnowledgeMemorySummaryResponse,
    # Enums
    KnowledgeType,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/memory/knowledge",
    tags=["Knowledge Memory"],
    responses={
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    }
)


# ============================================================================
# Helper Functions
# ============================================================================

def get_enum_value(value):
    """
    Safely get string value from enum or string.

    If value is an enum, returns its .value attribute.
    If value is already a string, returns it as-is.
    """
    if isinstance(value, str):
        return value
    return getattr(value, 'value', str(value))


def map_knowledge_entry_to_model(entry) -> KnowledgeEntryModel:
    """Convert KnowledgeEntry to Pydantic model."""
    entry_dict = entry.to_dict()
    return KnowledgeEntryModel(
        knowledge_id=entry_dict['knowledge_id'],
        knowledge_type=entry_dict['knowledge_type'],
        content=entry_dict['content'],
        confidence_level=entry_dict['confidence_level'],
        source_agent=entry_dict['source_agent'],
        domain=entry_dict['domain'],
        tags=entry_dict['tags'],
        created_at=datetime.fromtimestamp(entry_dict['created_at']),
        updated_at=datetime.fromtimestamp(entry_dict['updated_at']),
        access_count=entry_dict['access_count'],
        success_rate=entry_dict['success_rate'],
        validation_score=entry_dict['validation_score'],
        validation_status=entry_dict['validation_status']
    )


# ============================================================================
# Endpoint Implementations
# ============================================================================

@router.get("/", response_model=KnowledgeListResponse)
async def retrieve_knowledge(
    knowledge_types: Optional[List[KnowledgeType]] = Query(None, description="Filter by knowledge types"),
    domains: Optional[List[str]] = Query(None, description="Filter by domains"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    min_confidence: Optional[ConfidenceLevel] = Query(None, description="Minimum confidence level"),
    min_success_rate: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum success rate"),
    content_keywords: Optional[List[str]] = Query(None, description="Content keywords"),
    from_date: Optional[datetime] = Query(None, description="Start date filter"),
    to_date: Optional[datetime] = Query(None, description="End date filter"),
    task_type: Optional[str] = Query(None, description="Task type for meta-learning boost"),
    task_complexity: Optional[str] = Query(None, description="Task complexity for meta-learning"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    api_key: str = Depends(verify_api_key),
    knowledge_store = Depends(get_knowledge_store_memory)
):
    """
    Retrieve knowledge entries with complex filtering and meta-learning boost.

    Meta-learning boost: When task_type and task_complexity are provided,
    results are re-ranked based on historical usefulness for similar tasks.
    """
    try:
        from src.memory.knowledge_store import KnowledgeQuery
        from src.memory.knowledge_store import KnowledgeType as MemKnowledgeType
        from src.memory.knowledge_store import ConfidenceLevel as MemConfidenceLevel

        # Build query - convert API enums to memory enums
        mem_knowledge_types = None
        if knowledge_types:
            mem_knowledge_types = [MemKnowledgeType(get_enum_value(kt)) for kt in knowledge_types]

        mem_min_confidence = None
        if min_confidence:
            mem_min_confidence = MemConfidenceLevel(get_enum_value(min_confidence))

        query = KnowledgeQuery(
            knowledge_types=mem_knowledge_types,
            domains=domains,
            tags=tags,
            min_confidence=mem_min_confidence,
            min_success_rate=min_success_rate,
            content_keywords=content_keywords,
            time_range=(from_date.timestamp(), to_date.timestamp()) if from_date and to_date else None,
            limit=limit + offset,  # Get more to handle offset
            task_type=task_type,
            task_complexity=task_complexity
        )

        # Retrieve knowledge
        entries = knowledge_store.retrieve_knowledge(query)

        # Apply offset
        entries = entries[offset:offset + limit]

        # Convert to models
        entry_models = [map_knowledge_entry_to_model(e) for e in entries]

        return KnowledgeListResponse(
            entries=entry_models,
            total=len(entry_models),
            offset=offset,
            limit=limit
        )

    except Exception as e:
        logger.exception("Error retrieving knowledge")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving knowledge: {str(e)}"
        )


@router.get("/{knowledge_id}", response_model=KnowledgeEntryModel)
async def get_knowledge_entry(
    knowledge_id: str,
    api_key: str = Depends(verify_api_key),
    knowledge_store = Depends(get_knowledge_store_memory)
):
    """
    Get specific knowledge entry by ID.
    """
    try:
        from src.memory.knowledge_store import KnowledgeQuery

        # Query for specific entry
        query = KnowledgeQuery(limit=1000)  # Get enough to find it
        entries = knowledge_store.retrieve_knowledge(query)

        # Find matching entry
        for entry in entries:
            if entry.knowledge_id == knowledge_id:
                # Increment access count
                import sqlite3
                with sqlite3.connect(knowledge_store.storage_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE knowledge_entries SET access_count = access_count + 1 WHERE knowledge_id = ?",
                        (knowledge_id,)
                    )
                    conn.commit()

                return map_knowledge_entry_to_model(entry)

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge entry not found: {knowledge_id}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting knowledge entry")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting knowledge entry: {str(e)}"
        )


@router.post("/", response_model=KnowledgeStoreResponse)
async def store_knowledge(
    request: KnowledgeStoreRequest,
    api_key: str = Depends(verify_api_key),
    knowledge_store = Depends(get_knowledge_store_memory)
):
    """
    Store a new knowledge entry or update existing one.

    Knowledge entries are deduplicated based on domain, content, and source agent.
    If a duplicate exists, it will be updated with a new timestamp.
    """
    try:
        # Store knowledge
        knowledge_id = knowledge_store.store_knowledge(
            knowledge_type=get_enum_value(request.knowledge_type),
            content=request.content,
            confidence_level=get_enum_value(request.confidence_level),
            source_agent=request.source_agent,
            domain=request.domain,
            tags=request.tags
        )

        # Check if it was an update (try to find existing entry with same content)
        from src.memory.knowledge_store import KnowledgeQuery
        query = KnowledgeQuery(
            domains=[request.domain],
            limit=10
        )
        entries = knowledge_store.retrieve_knowledge(query)

        is_update = False
        for entry in entries:
            if entry.knowledge_id == knowledge_id and entry.access_count > 0:
                is_update = True
                break

        message = "Knowledge entry updated successfully" if is_update else "Knowledge entry stored successfully"

        return KnowledgeStoreResponse(
            knowledge_id=knowledge_id,
            stored=True,
            updated=is_update,
            message=message
        )

    except Exception as e:
        logger.exception("Error storing knowledge")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error storing knowledge: {str(e)}"
        )


@router.post("/{knowledge_id}/usage", response_model=KnowledgeUsageResponse)
async def record_usage(
    knowledge_id: str,
    request: KnowledgeUsageRequest,
    api_key: str = Depends(verify_api_key),
    knowledge_store = Depends(get_knowledge_store_memory)
):
    """
    Record knowledge usage for meta-learning.

    Tracks how useful each knowledge entry was for specific workflows and tasks.
    This data is used to boost relevant results in future searches.
    """
    try:
        # Record usage
        success = knowledge_store.record_knowledge_usage(
            workflow_id=request.workflow_id,
            knowledge_id=knowledge_id,
            task_type=request.task_type,
            task_complexity=request.task_complexity,
            useful_score=request.useful_score,
            retrieval_method=request.retrieval_method
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to record knowledge usage"
            )

        return KnowledgeUsageResponse(
            recorded=True,
            message="Usage recorded successfully for meta-learning"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error recording usage")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recording usage: {str(e)}"
        )


@router.patch("/{knowledge_id}/success-rate", response_model=KnowledgeStoreResponse)
async def update_success_rate(
    knowledge_id: str,
    request: KnowledgeSuccessRateRequest,
    api_key: str = Depends(verify_api_key),
    knowledge_store = Depends(get_knowledge_store_memory)
):
    """
    Update the success rate of a knowledge entry.

    Success rate reflects how often this knowledge led to successful task outcomes.
    """
    try:
        # Update success rate
        success = knowledge_store.update_success_rate(
            knowledge_id=knowledge_id,
            new_success_rate=request.new_success_rate
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge entry not found: {knowledge_id}"
            )

        return KnowledgeStoreResponse(
            knowledge_id=knowledge_id,
            stored=False,
            updated=True,
            message=f"Success rate updated to {request.new_success_rate}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error updating success rate")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating success rate: {str(e)}"
        )


@router.get("/{knowledge_id}/related", response_model=KnowledgeRelationshipsResponse)
async def get_related_knowledge(
    knowledge_id: str,
    max_results: int = Query(10, ge=1, le=50, description="Maximum related entries"),
    api_key: str = Depends(verify_api_key),
    knowledge_store = Depends(get_knowledge_store_memory)
):
    """
    Get knowledge entries related to the specified entry.

    Relationships are discovered through:
    - Explicit relationships stored in knowledge_relationships table
    - Content similarity
    - Shared tags and domains
    """
    try:
        import sqlite3
        import json

        # Get the source entry first
        from src.memory.knowledge_store import KnowledgeQuery
        query = KnowledgeQuery(limit=1000)
        entries = knowledge_store.retrieve_knowledge(query)

        source_entry = None
        for entry in entries:
            if entry.knowledge_id == knowledge_id:
                source_entry = entry
                break

        if not source_entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge entry not found: {knowledge_id}"
            )

        # Get explicit relationships from database
        relationships = []

        # Try to use knowledge_relationships table if it exists
        try:
            with sqlite3.connect(knowledge_store.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Check if relationships table exists
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_relationships'"
                )
                table_exists = cursor.fetchone() is not None

                if table_exists:
                    # Get relationships
                    cursor.execute("""
                        SELECT source_id, target_id, relationship_type, strength, basis
                        FROM knowledge_relationships
                        WHERE source_id = ? OR target_id = ?
                        LIMIT ?
                    """, (knowledge_id, knowledge_id, max_results))

                    rows = cursor.fetchall()

                    for row in rows:
                        # Get the target entry (the one that's not the source)
                        target_id = row["target_id"] if row["source_id"] == knowledge_id else row["source_id"]

                        # Find target entry
                        target_entry = None
                        for entry in entries:
                            if entry.knowledge_id == target_id:
                                target_entry = entry
                                break

                        if target_entry:
                            # Create preview of content (first 100 chars)
                            content_preview = str(target_entry.content)[:100]
                            if len(str(target_entry.content)) > 100:
                                content_preview += "..."

                            relationships.append(KnowledgeRelationshipItem(
                                target_knowledge_id=target_id,
                                target_content_preview=content_preview,
                                relationship_strength=row["strength"],
                                relationship_type=row["relationship_type"]
                            ))

        except Exception as e:
            logger.warning(f"Could not query relationships table: {e}")

        # If no explicit relationships, find related entries by domain and tags
        if not relationships:
            for entry in entries[:max_results]:
                if entry.knowledge_id == knowledge_id:
                    continue

                # Calculate similarity score
                score = 0.0

                # Same domain = +0.5
                if entry.domain == source_entry.domain:
                    score += 0.5

                # Shared tags
                shared_tags = set(entry.tags) & set(source_entry.tags)
                if shared_tags:
                    score += min(0.5, len(shared_tags) * 0.1)

                if score > 0.3:  # Threshold for relevance
                    # Create preview
                    content_preview = str(entry.content)[:100]
                    if len(str(entry.content)) > 100:
                        content_preview += "..."

                    relationships.append(KnowledgeRelationshipItem(
                        target_knowledge_id=entry.knowledge_id,
                        target_content_preview=content_preview,
                        relationship_strength=score,
                        relationship_type="related_domain" if entry.domain == source_entry.domain else "related_tags"
                    ))

            # Sort by strength
            relationships.sort(key=lambda x: x.relationship_strength, reverse=True)
            relationships = relationships[:max_results]

        return KnowledgeRelationshipsResponse(
            knowledge_id=knowledge_id,
            relationships=relationships,
            total=len(relationships)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting related knowledge")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting related knowledge: {str(e)}"
        )


@router.get("/summary/stats", response_model=KnowledgeMemorySummaryResponse)
async def get_summary(
    api_key: str = Depends(verify_api_key),
    knowledge_store = Depends(get_knowledge_store_memory)
):
    """
    Get knowledge memory statistics and summary.
    """
    try:
        summary = knowledge_store.get_knowledge_summary()

        # Convert to enum-keyed dicts
        entries_by_type = {}
        for key, value in summary.get("entries_by_type", {}).items():
            try:
                entries_by_type[KnowledgeType(key)] = value
            except ValueError:
                entries_by_type[key] = value

        entries_by_confidence = {}
        for key, value in summary.get("entries_by_confidence", {}).items():
            try:
                entries_by_confidence[ConfidenceLevel(key)] = value
            except ValueError:
                entries_by_confidence[key] = value

        return KnowledgeMemorySummaryResponse(
            total_entries=summary.get("total_entries", 0),
            entries_by_type=entries_by_type,
            entries_by_domain=summary.get("entries_by_domain", {}),
            entries_by_confidence=entries_by_confidence,
            average_success_rate=summary.get("average_success_rate", 0.0),
            total_access_count=summary.get("total_access_count", 0)
        )

    except Exception as e:
        logger.exception("Error getting summary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting summary: {str(e)}"
        )
