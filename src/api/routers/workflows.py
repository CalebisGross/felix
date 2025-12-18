"""
Workflow management endpoints.

Provides REST API for creating, executing, and monitoring Felix workflows.
"""

import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from src.api.dependencies import get_authenticated_felix, verify_api_key
from src.api.models import (
    WorkflowRequest,
    WorkflowResponse,
    WorkflowListResponse,
    WorkflowStatus,
    AgentInfo,
    SynthesisResult
)
from src.core.felix_system import FelixSystem

logger = logging.getLogger(__name__)

# Import WebSocket helpers for event streaming
try:
    from src.api.websockets.connection_manager import get_connection_manager
    from src.api.websockets.workflow_stream import send_workflow_event
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("WebSocket support not available")

    # Create dummy function for when WebSocket not available
    async def send_workflow_event(workflow_id, event_type, data):
        pass  # No-op when WebSocket not available

# Router
router = APIRouter(
    prefix="/api/v1/workflows",
    tags=["Workflows"],
    responses={404: {"description": "Workflow not found"}}
)

# Thread pool for running sync Felix code in async context
executor = ThreadPoolExecutor(max_workers=4)

# Workflow storage
# NOTE: Using in-memory storage for simplicity and low latency
# For production deployments with multiple API instances, consider:
# - felix_workflow_history.db (already tracks completed workflows)
# - Redis for distributed caching
# - PostgreSQL for persistent workflow state
#
# Current approach is acceptable for:
# - Single-instance deployments (Docker, air-gapped environments)
# - Development and testing
# - Workflows that complete within server uptime
#
# workflow_id -> workflow_data
workflows_db: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def generate_workflow_id() -> str:
    """Generate unique workflow ID."""
    return f"wf_{uuid.uuid4().hex[:12]}"


def _send_event_sync(workflow_id: str, event_type: str, data: Dict[str, Any]):
    """
    Helper to send WebSocket events from sync code.

    Creates a new event loop to send the async event.
    """
    if WEBSOCKET_AVAILABLE:
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send_workflow_event(workflow_id, event_type, data))
            loop.close()
        except Exception as e:
            logger.debug(f"Failed to send WebSocket event: {e}")


def run_workflow_sync(
    felix: FelixSystem,
    workflow_id: str,
    task: str,
    max_steps: Optional[int]
) -> Dict[str, Any]:
    """
    Run workflow synchronously in thread pool.

    This function is called from the thread pool to execute the
    synchronous Felix workflow in a non-blocking way.

    Args:
        felix: FelixSystem instance
        workflow_id: Workflow ID for tracking
        task: Task description
        max_steps: Maximum workflow steps

    Returns:
        Workflow result dictionary
    """
    try:
        logger.info(f"Starting workflow {workflow_id}: {task[:100]}")

        # Update status
        if workflow_id in workflows_db:
            workflows_db[workflow_id]["status"] = WorkflowStatus.RUNNING

            # Send status change event
            _send_event_sync(workflow_id, "workflow_status", {
                "status": "running",
                "timestamp": datetime.now().isoformat()
            })

        # Run Felix workflow
        result = felix.run_workflow(
            task_input=task,
            max_steps_override=max_steps,
            parent_workflow_id=None
        )

        # Process result
        agents_spawned = []
        agent_metadata = result.get("agent_metadata", {})  # Issue #56.8: Get agent metadata
        if "agents_spawned" in result:
            for agent_id in result["agents_spawned"]:
                # Get actual agent type and spawn time from metadata (Issue #56.8)
                meta = agent_metadata.get(agent_id, {})
                agent_type = meta.get("agent_type", "unknown")
                spawn_time = meta.get("spawn_time", 0.0)

                agent_info = AgentInfo(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    spawn_time=spawn_time
                )
                agents_spawned.append(agent_info)

                # Send agent spawned event
                _send_event_sync(workflow_id, "agent_spawned", {
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "spawn_time": spawn_time,
                    "timestamp": datetime.now().isoformat()
                })

        # Extract synthesis
        synthesis = None
        if "centralpost_synthesis" in result and result["centralpost_synthesis"]:
            synth_data = result["centralpost_synthesis"]
            synthesis = SynthesisResult(
                content=synth_data.get("synthesis_content", ""),
                confidence=synth_data.get("confidence", 0.0),
                agents_synthesized=synth_data.get("agents_synthesized", 0),
                token_count=synth_data.get("token_count")
            )

            # Send synthesis started event
            _send_event_sync(workflow_id, "synthesis_started", {
                "agent_count": synth_data.get("agents_synthesized", 0),
                "timestamp": datetime.now().isoformat()
            })

        # Determine status
        workflow_status = WorkflowStatus.COMPLETED
        error_msg = None

        if result.get("status") == "failed":
            workflow_status = WorkflowStatus.FAILED
            error_msg = result.get("error", "Unknown error")

        # Update workflow data
        if workflow_id in workflows_db:
            workflows_db[workflow_id].update({
                "status": workflow_status,
                "completed_at": datetime.now(),
                "agents_spawned": agents_spawned,
                "synthesis": synthesis,
                "performance_metrics": result.get("performance_metrics"),
                "error": error_msg,
                "raw_result": result
            })

        logger.info(f"Workflow {workflow_id} completed with status: {workflow_status}")

        # Send completion event
        if workflow_status == WorkflowStatus.COMPLETED and synthesis:
            _send_event_sync(workflow_id, "workflow_complete", {
                "status": "completed",
                "synthesis": {
                    "content": synthesis.content,
                    "confidence": synthesis.confidence,
                    "agents_synthesized": synthesis.agents_synthesized
                },
                "timestamp": datetime.now().isoformat()
            })
        elif workflow_status == WorkflowStatus.FAILED:
            _send_event_sync(workflow_id, "workflow_error", {
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })

        return workflows_db.get(workflow_id, {})

    except Exception as e:
        logger.exception(f"Error running workflow {workflow_id}")

        # Update workflow with error
        if workflow_id in workflows_db:
            workflows_db[workflow_id].update({
                "status": WorkflowStatus.FAILED,
                "completed_at": datetime.now(),
                "error": str(e)
            })

        # Send error event
        _send_event_sync(workflow_id, "workflow_error", {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

        return workflows_db.get(workflow_id, {})


# ============================================================================
# Workflow Endpoints
# ============================================================================

@router.post("", response_model=WorkflowResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_workflow(
    request: WorkflowRequest,
    background_tasks: BackgroundTasks,
    felix: FelixSystem = Depends(get_authenticated_felix)
) -> WorkflowResponse:
    """
    Create and execute a new workflow.

    Starts the workflow in the background and returns immediately with workflow_id.
    Use GET /api/v1/workflows/{workflow_id} to check status and retrieve results.

    Args:
        request: Workflow request with task description and parameters
        background_tasks: FastAPI background tasks
        felix: Felix system instance (injected)

    Returns:
        WorkflowResponse: Workflow information with status "running"

    Raises:
        HTTPException: If workflow creation fails

    Example:
        ```json
        POST /api/v1/workflows
        {
          "task": "Explain quantum computing",
          "max_steps": 10
        }

        Response (202 Accepted):
        {
          "workflow_id": "wf_abc123",
          "status": "running",
          "task": "Explain quantum computing",
          "created_at": "2025-10-30T10:00:00Z",
          "agents_spawned": [],
          "synthesis": null
        }
        ```
    """
    try:
        # Generate workflow ID
        workflow_id = generate_workflow_id()

        # Create workflow record
        workflow_data = {
            "workflow_id": workflow_id,
            "status": WorkflowStatus.PENDING,
            "task": request.task,
            "created_at": datetime.now(),
            "completed_at": None,
            "agents_spawned": [],
            "synthesis": None,
            "performance_metrics": None,
            "error": None,
            "max_steps": request.max_steps,
            "parent_workflow_id": request.parent_workflow_id
        }

        # Store workflow
        workflows_db[workflow_id] = workflow_data

        # Start workflow in background
        # Use run_in_executor to run sync code in thread pool
        loop = asyncio.get_event_loop()
        loop.run_in_executor(
            executor,
            run_workflow_sync,
            felix,
            workflow_id,
            request.task,
            request.max_steps
        )

        # Return immediate response
        return WorkflowResponse(**workflow_data)

    except Exception as e:
        logger.exception("Error creating workflow")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    api_key: str = Depends(verify_api_key)
) -> WorkflowResponse:
    """
    Get workflow status and results.

    Retrieves the current status of a workflow, including:
    - Execution status (pending, running, completed, failed)
    - Spawned agents
    - Synthesis results (if completed)
    - Performance metrics

    Args:
        workflow_id: Unique workflow identifier
        api_key: API key for authentication

    Returns:
        WorkflowResponse: Complete workflow information

    Raises:
        HTTPException: If workflow not found

    Example:
        ```
        GET /api/v1/workflows/wf_abc123

        Response:
        {
          "workflow_id": "wf_abc123",
          "status": "completed",
          "task": "Explain quantum computing",
          "created_at": "2025-10-30T10:00:00Z",
          "completed_at": "2025-10-30T10:02:30Z",
          "agents_spawned": [...],
          "synthesis": {
            "content": "Quantum computing is...",
            "confidence": 0.87,
            "agents_synthesized": 3
          }
        }
        ```
    """
    if workflow_id not in workflows_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {workflow_id}"
        )

    workflow_data = workflows_db[workflow_id]
    return WorkflowResponse(**workflow_data)


@router.get("", response_model=WorkflowListResponse)
async def list_workflows(
    status_filter: Optional[str] = Query(None, description="Filter by status: pending, running, completed, failed"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results to return"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    api_key: str = Depends(verify_api_key)
) -> WorkflowListResponse:
    """
    List workflows with optional filtering.

    Returns a paginated list of workflows, optionally filtered by status.

    Args:
        status_filter: Optional status filter (pending, running, completed, failed)
        limit: Maximum number of results (1-500, default 50)
        offset: Number of results to skip for pagination (default 0)
        api_key: API key for authentication

    Returns:
        WorkflowListResponse: List of workflows with pagination info

    Example:
        ```
        GET /api/v1/workflows?status_filter=completed&limit=10

        Response:
        {
          "workflows": [
            {
              "workflow_id": "wf_abc123",
              "status": "completed",
              ...
            }
          ],
          "total": 42,
          "page": 0,
          "page_size": 10
        }
        ```
    """
    # Filter workflows
    filtered_workflows = []

    for workflow_data in workflows_db.values():
        # Apply status filter if provided
        if status_filter:
            try:
                status_enum = WorkflowStatus(status_filter)
                if workflow_data["status"] != status_enum:
                    continue
            except ValueError:
                # Invalid status filter - skip
                continue

        filtered_workflows.append(workflow_data)

    # Sort by created_at (most recent first)
    filtered_workflows.sort(
        key=lambda w: w["created_at"],
        reverse=True
    )

    # Apply pagination
    total = len(filtered_workflows)
    paginated = filtered_workflows[offset:offset + limit]

    # Convert to response models
    workflow_responses = [WorkflowResponse(**w) for w in paginated]

    return WorkflowListResponse(
        workflows=workflow_responses,
        total=total,
        page=offset // limit,
        page_size=limit
    )


@router.delete("/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_workflow(
    workflow_id: str,
    felix: FelixSystem = Depends(get_authenticated_felix)
) -> None:
    """
    Cancel a running workflow.

    Attempts to cancel a workflow that is currently running.
    Completed or failed workflows cannot be cancelled.

    Args:
        workflow_id: Unique workflow identifier
        felix: Felix system instance

    Returns:
        None (204 No Content)

    Raises:
        HTTPException: If workflow not found or cannot be cancelled

    Example:
        ```
        DELETE /api/v1/workflows/wf_abc123

        Response: 204 No Content
        ```
    """
    if workflow_id not in workflows_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow not found: {workflow_id}"
        )

    workflow_data = workflows_db[workflow_id]

    if workflow_data["status"] not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel workflow with status: {workflow_data['status']}"
        )

    # TODO: Implement actual workflow cancellation
    # For now, just mark as cancelled
    workflow_data["status"] = WorkflowStatus.CANCELLED
    workflow_data["completed_at"] = datetime.now()
    workflow_data["error"] = "Workflow cancelled by user"

    logger.info(f"Workflow {workflow_id} cancelled")
