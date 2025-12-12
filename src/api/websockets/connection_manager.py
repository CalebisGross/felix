"""
WebSocket connection manager.

Manages multiple WebSocket connections and handles broadcasting events
to connected clients.
"""

import logging
import asyncio
from typing import Dict, Set, Optional, Any
from datetime import datetime
import json

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time event streaming.

    Handles:
    - Connection lifecycle (connect, disconnect)
    - Broadcasting events to specific workflows
    - Broadcasting system-wide events
    - Connection cleanup
    """

    def __init__(self):
        """Initialize connection manager."""
        # workflow_id -> set of WebSocket connections
        self._workflow_connections: Dict[str, Set[WebSocket]] = {}

        # All active connections (for system-wide events)
        self._all_connections: Set[WebSocket] = set()

        # Connection metadata: WebSocket -> dict
        self._connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        logger.info("ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, workflow_id: Optional[str] = None) -> None:
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to accept
            workflow_id: Optional workflow ID to associate with connection
        """
        await websocket.accept()

        async with self._lock:
            # Add to all connections
            self._all_connections.add(websocket)

            # Store metadata
            self._connection_metadata[websocket] = {
                "workflow_id": workflow_id,
                "connected_at": datetime.now(),
                "messages_sent": 0
            }

            # If workflow-specific, add to workflow connections
            if workflow_id:
                if workflow_id not in self._workflow_connections:
                    self._workflow_connections[workflow_id] = set()
                self._workflow_connections[workflow_id].add(websocket)

                logger.info(f"WebSocket connected for workflow {workflow_id}")
            else:
                logger.info("WebSocket connected (system-wide)")

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Unregister and cleanup a WebSocket connection.

        Args:
            websocket: WebSocket connection to disconnect
        """
        async with self._lock:
            # Get workflow_id before removing metadata
            metadata = self._connection_metadata.get(websocket, {})
            workflow_id = metadata.get("workflow_id")

            # Remove from all connections
            self._all_connections.discard(websocket)

            # Remove from workflow connections
            if workflow_id and workflow_id in self._workflow_connections:
                self._workflow_connections[workflow_id].discard(websocket)

                # Cleanup empty sets
                if not self._workflow_connections[workflow_id]:
                    del self._workflow_connections[workflow_id]

            # Remove metadata
            self._connection_metadata.pop(websocket, None)

            if workflow_id:
                logger.info(f"WebSocket disconnected from workflow {workflow_id}")
            else:
                logger.info("WebSocket disconnected (system-wide)")

    async def send_to_workflow(self, workflow_id: str, event: Dict[str, Any]) -> int:
        """
        Send event to all connections for a specific workflow.

        Args:
            workflow_id: Workflow ID to send event to
            event: Event data to send

        Returns:
            Number of connections the event was sent to
        """
        if workflow_id not in self._workflow_connections:
            return 0

        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()

        connections = list(self._workflow_connections[workflow_id])
        sent_count = 0

        for websocket in connections:
            try:
                await websocket.send_json(event)
                self._connection_metadata[websocket]["messages_sent"] += 1
                sent_count += 1
            except Exception as e:
                logger.error(f"Error sending to WebSocket: {e}")
                # Remove broken connection
                await self.disconnect(websocket)

        return sent_count

    async def send_to_all(self, event: Dict[str, Any]) -> int:
        """
        Broadcast event to all connected clients.

        Args:
            event: Event data to send

        Returns:
            Number of connections the event was sent to
        """
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()

        connections = list(self._all_connections)
        sent_count = 0

        for websocket in connections:
            try:
                await websocket.send_json(event)
                self._connection_metadata[websocket]["messages_sent"] += 1
                sent_count += 1
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                await self.disconnect(websocket)

        return sent_count

    async def send_personal(self, websocket: WebSocket, event: Dict[str, Any]) -> bool:
        """
        Send event to a specific WebSocket connection.

        Args:
            websocket: WebSocket to send to
            event: Event data to send

        Returns:
            True if sent successfully, False otherwise
        """
        if "timestamp" not in event:
            event["timestamp"] = datetime.now().isoformat()

        try:
            await websocket.send_json(event)
            if websocket in self._connection_metadata:
                self._connection_metadata[websocket]["messages_sent"] += 1
            return True
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            await self.disconnect(websocket)
            return False

    def get_workflow_connection_count(self, workflow_id: str) -> int:
        """
        Get number of active connections for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Number of active connections
        """
        return len(self._workflow_connections.get(workflow_id, set()))

    def get_total_connections(self) -> int:
        """
        Get total number of active connections.

        Returns:
            Total connection count
        """
        return len(self._all_connections)

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about active connections.

        Returns:
            Dictionary with connection statistics
        """
        return {
            "total_connections": len(self._all_connections),
            "workflow_connections": {
                wf_id: len(connections)
                for wf_id, connections in self._workflow_connections.items()
            },
            "system_connections": len(self._all_connections) - sum(
                len(connections) for connections in self._workflow_connections.values()
            )
        }


# Global connection manager instance
_connection_manager: Optional[ConnectionManager] = None


def get_connection_manager() -> ConnectionManager:
    """
    Get the global connection manager instance.

    Returns:
        ConnectionManager singleton
    """
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = ConnectionManager()
    return _connection_manager
