"""
Workflow History API router.

Endpoints for querying workflow execution history, conversation threads, and analytics.
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query

from src.api.dependencies import verify_api_key, get_workflow_history
from src.api.models import (
    # Request models
    WorkflowHistoryQueryRequest,
    # Response models
    WorkflowHistoryModel,
    WorkflowHistoryListResponse,
    WorkflowThreadResponse,
    WorkflowAnalyticsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/memory/workflows",
    tags=["Workflow History"],
    responses={
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    }
)


# ============================================================================
# Helper Functions
# ============================================================================

def map_workflow_output_to_model(output) -> WorkflowHistoryModel:
    """Convert WorkflowOutput to Pydantic model."""
    return WorkflowHistoryModel(
        workflow_id=output.workflow_id,
        task_input=output.task_input,
        status=output.status,
        created_at=datetime.fromisoformat(output.created_at.replace('Z', '+00:00')) if isinstance(output.created_at, str) else output.created_at,
        completed_at=datetime.fromisoformat(output.completed_at.replace('Z', '+00:00')) if output.completed_at and isinstance(output.completed_at, str) else output.completed_at,
        final_synthesis=output.final_synthesis,
        confidence=output.confidence,
        agents_count=output.agents_count,
        tokens_used=output.tokens_used,
        max_tokens=output.max_tokens,
        processing_time=output.processing_time,
        temperature=output.temperature,
        metadata=output.metadata,
        parent_workflow_id=output.parent_workflow_id,
        conversation_thread_id=output.conversation_thread_id
    )


# ============================================================================
# Endpoint Implementations
# ============================================================================

@router.get("/", response_model=WorkflowHistoryListResponse)
async def list_workflows(
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    from_date: Optional[datetime] = Query(None, description="Start date filter"),
    to_date: Optional[datetime] = Query(None, description="End date filter"),
    search_query: Optional[str] = Query(None, description="Search in task_input and synthesis"),
    parent_workflow_id: Optional[int] = Query(None, description="Filter by parent workflow ID"),
    conversation_thread_id: Optional[str] = Query(None, description="Filter by conversation thread"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    api_key: str = Depends(verify_api_key),
    workflow_history = Depends(get_workflow_history)
):
    """
    List workflow execution records with optional filtering and pagination.

    Supports filtering by:
    - Status (completed, failed, etc.)
    - Date range
    - Full-text search
    - Parent workflow ID
    - Conversation thread ID
    """
    try:
        # Get workflows
        workflows = workflow_history.get_workflow_outputs(
            status_filter=status_filter,
            limit=limit,
            offset=offset
        )

        # Apply additional filters
        filtered_workflows = []

        for workflow in workflows:
            # Date filter
            if from_date:
                workflow_date = datetime.fromisoformat(workflow.created_at.replace('Z', '+00:00')) if isinstance(workflow.created_at, str) else workflow.created_at
                if workflow_date < from_date:
                    continue

            if to_date:
                workflow_date = datetime.fromisoformat(workflow.created_at.replace('Z', '+00:00')) if isinstance(workflow.created_at, str) else workflow.created_at
                if workflow_date > to_date:
                    continue

            # Search filter
            if search_query:
                search_lower = search_query.lower()
                if search_lower not in workflow.task_input.lower():
                    if not workflow.final_synthesis or search_lower not in workflow.final_synthesis.lower():
                        continue

            # Parent workflow filter
            if parent_workflow_id is not None:
                if workflow.parent_workflow_id != parent_workflow_id:
                    continue

            # Thread filter
            if conversation_thread_id:
                if workflow.conversation_thread_id != conversation_thread_id:
                    continue

            filtered_workflows.append(workflow)

        # Convert to models
        workflow_models = [map_workflow_output_to_model(w) for w in filtered_workflows]

        return WorkflowHistoryListResponse(
            workflows=workflow_models,
            total=len(workflow_models),
            offset=offset,
            limit=limit
        )

    except Exception as e:
        logger.exception("Error listing workflows")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing workflows: {str(e)}"
        )


@router.get("/{workflow_id}", response_model=WorkflowHistoryModel)
async def get_workflow(
    workflow_id: int,
    api_key: str = Depends(verify_api_key),
    workflow_history = Depends(get_workflow_history)
):
    """
    Get specific workflow by ID.
    """
    try:
        workflow = workflow_history.get_workflow_by_id(workflow_id)

        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )

        return map_workflow_output_to_model(workflow)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting workflow")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting workflow: {str(e)}"
        )


@router.get("/{workflow_id}/thread", response_model=WorkflowThreadResponse)
async def get_conversation_thread(
    workflow_id: int,
    api_key: str = Depends(verify_api_key),
    workflow_history = Depends(get_workflow_history)
):
    """
    Get complete conversation thread for a workflow.

    Returns the root workflow and all related workflows in the conversation thread.
    Useful for understanding multi-turn conversations.
    """
    try:
        thread_workflows = workflow_history.get_conversation_thread(workflow_id)

        if not thread_workflows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No thread found for workflow: {workflow_id}"
            )

        # Convert to models
        workflow_models = [map_workflow_output_to_model(w) for w in thread_workflows]

        # Find root (workflow with no parent)
        root_workflow = None
        child_workflows = []

        for wf_model in workflow_models:
            if wf_model.parent_workflow_id is None:
                root_workflow = wf_model
            else:
                child_workflows.append(wf_model)

        if not root_workflow:
            # If no root found, use first workflow
            root_workflow = workflow_models[0]
            child_workflows = workflow_models[1:]

        # Calculate thread depth (max distance from root)
        thread_depth = 1
        if child_workflows:
            # Simple depth calculation: count parent chains
            max_depth = 1
            for child in child_workflows:
                depth = 1
                current = child
                # Walk up parent chain
                while current.parent_workflow_id:
                    depth += 1
                    # Find parent
                    found = False
                    for wf in workflow_models:
                        if wf.workflow_id == current.parent_workflow_id:
                            current = wf
                            found = True
                            break
                    if not found:
                        break
                max_depth = max(max_depth, depth)
            thread_depth = max_depth

        return WorkflowThreadResponse(
            thread_id=root_workflow.conversation_thread_id or str(root_workflow.workflow_id),
            root_workflow=root_workflow,
            child_workflows=child_workflows,
            total_workflows=len(workflow_models),
            thread_depth=thread_depth
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting conversation thread")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting conversation thread: {str(e)}"
        )


@router.post("/", response_model=WorkflowHistoryModel)
async def save_workflow(
    workflow: WorkflowHistoryModel,
    api_key: str = Depends(verify_api_key),
    workflow_history = Depends(get_workflow_history)
):
    """
    Save a new workflow to history.

    This endpoint is typically called by the Felix system after workflow completion.
    """
    try:
        # Create workflow result dict
        result = {
            "task": workflow.task_input,
            "synthesis": workflow.final_synthesis,
            "confidence": workflow.confidence,
            "agents_count": workflow.agents_count,
            "tokens_used": workflow.tokens_used,
            "max_tokens": workflow.max_tokens,
            "processing_time": workflow.processing_time,
            "temperature": workflow.temperature,
            "metadata": workflow.metadata
        }

        # Save workflow
        workflow_id = workflow_history.save_workflow_output(
            task=workflow.task_input,
            result=result,
            status=workflow.status,
            parent_workflow_id=workflow.parent_workflow_id,
            conversation_thread_id=workflow.conversation_thread_id
        )

        # Get saved workflow
        saved_workflow = workflow_history.get_workflow_by_id(workflow_id)

        return map_workflow_output_to_model(saved_workflow)

    except Exception as e:
        logger.exception("Error saving workflow")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving workflow: {str(e)}"
        )


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: int,
    api_key: str = Depends(verify_api_key),
    workflow_history = Depends(get_workflow_history)
):
    """
    Delete a workflow from history.
    """
    try:
        success = workflow_history.delete_workflow(workflow_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )

        return {"message": f"Workflow {workflow_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error deleting workflow")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting workflow: {str(e)}"
        )


@router.get("/search/", response_model=WorkflowHistoryListResponse)
async def search_workflows(
    query: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    api_key: str = Depends(verify_api_key),
    workflow_history = Depends(get_workflow_history)
):
    """
    Full-text search across workflow task inputs and synthesis results.
    """
    try:
        # Use workflow history's search method
        workflows = workflow_history.search_workflows(query, limit=limit, offset=offset)

        # Convert to models
        workflow_models = [map_workflow_output_to_model(w) for w in workflows]

        return WorkflowHistoryListResponse(
            workflows=workflow_models,
            total=len(workflow_models),
            offset=offset,
            limit=limit
        )

    except Exception as e:
        logger.exception("Error searching workflows")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error searching workflows: {str(e)}"
        )


@router.get("/analytics/summary", response_model=WorkflowAnalyticsResponse)
async def get_analytics(
    from_date: Optional[datetime] = Query(None, description="Start date for analytics"),
    to_date: Optional[datetime] = Query(None, description="End date for analytics"),
    api_key: str = Depends(verify_api_key),
    workflow_history = Depends(get_workflow_history)
):
    """
    Get workflow analytics and performance metrics.

    Provides aggregated statistics including:
    - Total workflows
    - Completion rates
    - Average confidence
    - Average processing time
    - Status distribution
    - Workflows by date
    """
    try:
        # Get all workflows
        workflows = workflow_history.get_workflow_outputs(limit=10000)

        # Apply date filters
        filtered_workflows = []
        for workflow in workflows:
            workflow_date = datetime.fromisoformat(workflow.created_at.replace('Z', '+00:00')) if isinstance(workflow.created_at, str) else workflow.created_at

            if from_date and workflow_date < from_date:
                continue
            if to_date and workflow_date > to_date:
                continue

            filtered_workflows.append(workflow)

        # Calculate analytics
        total_workflows = len(filtered_workflows)

        if total_workflows == 0:
            # Return empty analytics
            return WorkflowAnalyticsResponse(
                total_workflows=0,
                completed_workflows=0,
                failed_workflows=0,
                average_confidence=0.0,
                average_agents_count=0.0,
                average_processing_time=0.0,
                average_tokens_used=0.0,
                status_distribution={},
                workflows_by_date={}
            )

        # Count by status
        status_distribution = {}
        completed_count = 0
        failed_count = 0

        # Calculate averages
        total_confidence = 0.0
        confidence_count = 0
        total_agents = 0
        total_time = 0.0
        total_tokens = 0

        # Workflows by date
        workflows_by_date = {}

        for workflow in filtered_workflows:
            # Status distribution
            status = workflow.status
            status_distribution[status] = status_distribution.get(status, 0) + 1

            if status == "completed":
                completed_count += 1
            elif status == "failed":
                failed_count += 1

            # Confidence
            if workflow.confidence is not None:
                total_confidence += workflow.confidence
                confidence_count += 1

            # Agents count
            total_agents += workflow.agents_count

            # Processing time
            total_time += workflow.processing_time

            # Tokens used
            total_tokens += workflow.tokens_used

            # Date distribution
            workflow_date = datetime.fromisoformat(workflow.created_at.replace('Z', '+00:00')) if isinstance(workflow.created_at, str) else workflow.created_at
            date_key = workflow_date.strftime("%Y-%m-%d")
            workflows_by_date[date_key] = workflows_by_date.get(date_key, 0) + 1

        return WorkflowAnalyticsResponse(
            total_workflows=total_workflows,
            completed_workflows=completed_count,
            failed_workflows=failed_count,
            average_confidence=total_confidence / confidence_count if confidence_count > 0 else 0.0,
            average_agents_count=total_agents / total_workflows,
            average_processing_time=total_time / total_workflows,
            average_tokens_used=total_tokens / total_workflows,
            status_distribution=status_distribution,
            workflows_by_date=workflows_by_date
        )

    except Exception as e:
        logger.exception("Error getting analytics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting analytics: {str(e)}"
        )
