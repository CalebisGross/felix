"""
Streaming Coordinator for the Felix Framework.

Handles real-time streaming of agent thoughts and progress updates,
implementing the hybrid approach: real-time display with deferred synthesis.

Key Features:
- Time-batched streaming from agents
- Accumulated content tracking
- Real-time GUI event emission
- Streaming completion handling
- Callback registration for listeners
- Metadata tracking (agent_type, progress, tokens)

This module was extracted from CentralPost to improve separation of concerns
and maintainability while preserving all functionality.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Callable

# Set up logging
logger = logging.getLogger(__name__)


class StreamingCoordinator:
    """
    Manages real-time streaming of agent thoughts and progress.

    Responsibilities:
    - Receive partial thoughts from agents
    - Accumulate streaming content
    - Emit real-time events to GUI listeners
    - Handle streaming completion
    - Maintain streaming metadata
    """

    def __init__(self):
        """Initialize Streaming Coordinator."""
        # Streaming state tracking
        self._partial_thoughts: Dict[str, str] = {}  # agent_id -> accumulated content
        self._streaming_metadata: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata
        self._streaming_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        logger.info("✓ StreamingCoordinator initialized")

    def receive_partial_thought(
        self,
        agent_id: str,
        partial_content: str,
        accumulated: str,
        progress: float,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Receive time-batched streaming chunk from agent.

        Accumulates content for display but doesn't synthesize until complete.
        This is the HYBRID approach: real-time display, deferred synthesis.

        Args:
            agent_id: Agent sending the partial thought
            partial_content: New content since last batch
            accumulated: Full content accumulated so far
            progress: Agent's progress along helix (0.0-1.0)
            metadata: Additional metadata (agent_type, checkpoint, etc.)
        """
        # Update accumulated state
        self._partial_thoughts[agent_id] = accumulated
        self._streaming_metadata[agent_id] = metadata

        # Emit event for GUI (real-time display)
        self._emit_streaming_event({
            "type": "partial_thought",
            "agent_id": agent_id,
            "partial": partial_content,
            "accumulated": accumulated,
            "progress": progress,
            "agent_type": metadata.get("agent_type"),
            "checkpoint": metadata.get("checkpoint"),
            "tokens_so_far": metadata.get("tokens_so_far", 0),
            "timestamp": time.time()
        })

        # Note: Do NOT synthesize here (wait for completion per hybrid approach)

    def finalize_streaming_thought(
        self,
        agent_id: str,
        final_content: str,
        confidence: float
    ) -> None:
        """
        Finalize streaming thought when agent completes.

        Now we can synthesize with complete message (hybrid approach).

        Args:
            agent_id: Agent completing the thought
            final_content: Complete final content
            confidence: Agent's confidence score
        """
        # Clean up streaming state
        metadata = self._streaming_metadata.pop(agent_id, {})
        self._partial_thoughts.pop(agent_id, None)

        # Log completion
        logger.info(f"✓ Streaming thought complete: {agent_id} (confidence: {confidence:.2f})")

        # Emit completion event
        self._emit_streaming_event({
            "type": "thought_complete",
            "agent_id": agent_id,
            "final_content": final_content,
            "confidence": confidence,
            "agent_type": metadata.get("agent_type"),
            "timestamp": time.time()
        })

        # Now consider synthesis with complete message (hybrid approach)
        # Note: Actual synthesis logic handled by workflow/CentralPost

    def _emit_streaming_event(self, event: Dict[str, Any]) -> None:
        """
        Emit streaming event to registered callbacks (GUI listeners).

        Args:
            event: Event data to emit
        """
        for callback in self._streaming_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Streaming event callback failed: {e}")

    def register_streaming_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register callback for streaming events (for GUI updates).

        Args:
            callback: Function to call with streaming events
        """
        self._streaming_callbacks.append(callback)
        logger.info(f"Registered streaming callback (total: {len(self._streaming_callbacks)})")

    def get_active_streams(self) -> List[str]:
        """
        Get list of agent IDs with active streaming sessions.

        Returns:
            List of agent IDs currently streaming
        """
        return list(self._partial_thoughts.keys())

    def get_accumulated_content(self, agent_id: str) -> Optional[str]:
        """
        Get accumulated content for specific agent.

        Args:
            agent_id: Agent ID to query

        Returns:
            Accumulated content or None if not streaming
        """
        return self._partial_thoughts.get(agent_id)

    def get_streaming_metadata(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get streaming metadata for specific agent.

        Args:
            agent_id: Agent ID to query

        Returns:
            Metadata dict or None if not streaming
        """
        return self._streaming_metadata.get(agent_id)

    def clear_all_streams(self) -> None:
        """Clear all streaming state (for reset/shutdown)."""
        self._partial_thoughts.clear()
        self._streaming_metadata.clear()
        logger.info("Cleared all streaming state")

    def get_stream_count(self) -> int:
        """Get count of active streaming sessions."""
        return len(self._partial_thoughts)
