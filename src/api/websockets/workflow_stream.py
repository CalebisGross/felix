"""
WebSocket endpoint for workflow streaming.

Provides real-time updates for workflow execution including:
- Workflow status changes
- Agent spawn events
- Agent output events
- Synthesis results
- Error events
"""

import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from fastapi.exceptions import HTTPException

from src.api.websockets.connection_manager import get_connection_manager
from src.api.dependencies import verify_api_key, optional_api_key
from src.api.routers.workflows import workflows_db

logger = logging.getLogger(__name__)

# Router for WebSocket endpoints
router = APIRouter(
    prefix="/api/v1/ws",
    tags=["WebSocket"]
)


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@router.websocket("/workflows/{workflow_id}")
async def workflow_stream(
    websocket: WebSocket,
    workflow_id: str,
    api_key: Optional[str] = Query(None, description="API key for authentication")
):
    """
    WebSocket endpoint for real-time workflow updates.

    Streams live events for a specific workflow including:
    - Workflow status changes (pending, running, completed, failed)
    - Agent spawn events (when new agents are created)
    - Agent output events (agent responses and confidence)
    - Synthesis events (final synthesis progress)
    - Error events (if workflow fails)

    **Connection:**
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/api/v1/ws/workflows/wf_abc123?api_key=your-key');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('Event:', data.type, data);
    };
    ```

    **Event Types:**

    1. **connected** - Initial connection confirmation
       ```json
       {
         "type": "connected",
         "workflow_id": "wf_abc123",
         "timestamp": "2025-10-30T10:00:00Z"
       }
       ```

    2. **workflow_status** - Workflow status change
       ```json
       {
         "type": "workflow_status",
         "workflow_id": "wf_abc123",
         "status": "running",
         "timestamp": "2025-10-30T10:00:01Z"
       }
       ```

    3. **agent_spawned** - New agent created
       ```json
       {
         "type": "agent_spawned",
         "workflow_id": "wf_abc123",
         "agent_id": "research_001",
         "agent_type": "research",
         "spawn_time": 0.1,
         "timestamp": "2025-10-30T10:00:02Z"
       }
       ```

    4. **agent_output** - Agent produced output
       ```json
       {
         "type": "agent_output",
         "workflow_id": "wf_abc123",
         "agent_id": "research_001",
         "content": "Research findings...",
         "confidence": 0.85,
         "timestamp": "2025-10-30T10:00:05Z"
       }
       ```

    5. **synthesis_started** - Synthesis phase beginning
       ```json
       {
         "type": "synthesis_started",
         "workflow_id": "wf_abc123",
         "agent_count": 3,
         "timestamp": "2025-10-30T10:02:00Z"
       }
       ```

    6. **workflow_complete** - Workflow finished successfully
       ```json
       {
         "type": "workflow_complete",
         "workflow_id": "wf_abc123",
         "status": "completed",
         "synthesis": {
           "content": "Final synthesis...",
           "confidence": 0.87,
           "agents_synthesized": 3
         },
         "timestamp": "2025-10-30T10:02:30Z"
       }
       ```

    7. **workflow_error** - Workflow failed
       ```json
       {
         "type": "workflow_error",
         "workflow_id": "wf_abc123",
         "error": "Error message",
         "timestamp": "2025-10-30T10:02:30Z"
       }
       ```

    Args:
        websocket: WebSocket connection
        workflow_id: Workflow ID to stream updates for
        api_key: Optional API key for authentication (query parameter)
    """
    manager = get_connection_manager()

    # Verify API key if authentication is enabled
    # Note: API key passed as query parameter for WebSocket
    # TODO: Implement proper WebSocket authentication
    if api_key:
        # Simplified auth check - in production, validate properly
        pass

    # Check if workflow exists
    if workflow_id not in workflows_db:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Workflow not found")
        return

    # Accept connection and register with manager
    await manager.connect(websocket, workflow_id)

    try:
        # Send initial connection confirmation
        await manager.send_personal(websocket, {
            "type": "connected",
            "workflow_id": workflow_id,
            "message": f"Connected to workflow {workflow_id} stream"
        })

        # Send current workflow status
        workflow = workflows_db[workflow_id]
        await manager.send_personal(websocket, {
            "type": "workflow_status",
            "workflow_id": workflow_id,
            "status": workflow["status"].value if hasattr(workflow["status"], "value") else str(workflow["status"]),
            "created_at": workflow["created_at"].isoformat(),
            "task": workflow["task"]
        })

        # If workflow is already completed, send completion event
        if workflow["status"].value in ["completed", "failed", "cancelled"]:
            if workflow["status"].value == "completed" and workflow.get("synthesis"):
                synthesis = workflow["synthesis"]
                await manager.send_personal(websocket, {
                    "type": "workflow_complete",
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "synthesis": {
                        "content": synthesis.content if hasattr(synthesis, "content") else str(synthesis),
                        "confidence": synthesis.confidence if hasattr(synthesis, "confidence") else 0.0,
                        "agents_synthesized": synthesis.agents_synthesized if hasattr(synthesis, "agents_synthesized") else 0
                    }
                })
            elif workflow["status"].value == "failed":
                await manager.send_personal(websocket, {
                    "type": "workflow_error",
                    "workflow_id": workflow_id,
                    "error": workflow.get("error", "Unknown error")
                })

        # Keep connection alive and wait for messages
        while True:
            try:
                # Receive messages from client (ping/pong, commands, etc.)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                # Handle client messages
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "status":
                    # Send current workflow status
                    workflow = workflows_db.get(workflow_id)
                    if workflow:
                        await manager.send_personal(websocket, {
                            "type": "workflow_status",
                            "workflow_id": workflow_id,
                            "status": workflow["status"].value if hasattr(workflow["status"], "value") else str(workflow["status"])
                        })

            except asyncio.TimeoutError:
                # No message received, send keepalive ping
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    # Connection closed
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from workflow {workflow_id}")

    except Exception as e:
        logger.exception(f"WebSocket error for workflow {workflow_id}")

    finally:
        # Cleanup connection
        await manager.disconnect(websocket)


@router.websocket("/system/events")
async def system_events(
    websocket: WebSocket,
    api_key: Optional[str] = Query(None, description="API key for authentication")
):
    """
    WebSocket endpoint for system-wide events.

    Streams all system events including:
    - Agent registration/deregistration
    - Knowledge brain events
    - Approval requests
    - System status changes

    **Connection:**
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/api/v1/ws/system/events?api_key=your-key');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('System Event:', data.type, data);
    };
    ```

    **Event Types:**

    1. **agent_registered** - New agent registered
    2. **agent_deregistered** - Agent removed
    3. **knowledge_ingested** - Document ingested
    4. **approval_requested** - Command requires approval
    5. **system_status_change** - System status changed

    Args:
        websocket: WebSocket connection
        api_key: Optional API key for authentication
    """
    manager = get_connection_manager()

    # Accept connection
    await manager.connect(websocket, workflow_id=None)

    try:
        # Send initial connection confirmation
        await manager.send_personal(websocket, {
            "type": "connected",
            "message": "Connected to system events stream",
            "connection_id": id(websocket)
        })

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "stats":
                    # Send connection statistics
                    stats = manager.get_connection_stats()
                    await manager.send_personal(websocket, {
                        "type": "connection_stats",
                        **stats
                    })

            except asyncio.TimeoutError:
                # Keepalive ping
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected from system events")

    except Exception as e:
        logger.exception("WebSocket error for system events")

    finally:
        await manager.disconnect(websocket)


# ============================================================================
# Helper Functions for Sending Events
# ============================================================================

async def send_workflow_event(workflow_id: str, event_type: str, data: Dict[str, Any]) -> int:
    """
    Helper function to send an event to all connections for a workflow.

    This should be called from workflow execution code to broadcast events.

    Args:
        workflow_id: Workflow ID
        event_type: Type of event (e.g., "agent_spawned", "agent_output")
        data: Event data

    Returns:
        Number of connections the event was sent to
    """
    manager = get_connection_manager()

    event = {
        "type": event_type,
        "workflow_id": workflow_id,
        **data
    }

    return await manager.send_to_workflow(workflow_id, event)


async def send_system_event(event_type: str, data: Dict[str, Any]) -> int:
    """
    Helper function to broadcast a system-wide event.

    Args:
        event_type: Type of event
        data: Event data

    Returns:
        Number of connections the event was sent to
    """
    manager = get_connection_manager()

    event = {
        "type": event_type,
        **data
    }

    return await manager.send_to_all(event)
