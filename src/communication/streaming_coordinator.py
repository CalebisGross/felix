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

    def __init__(self, stream_timeout_seconds: float = 300.0):
        """
        Initialize Streaming Coordinator.

        Args:
            stream_timeout_seconds: Timeout for stale stream cleanup (default 5 minutes)
        """
        # Streaming state tracking
        self._partial_thoughts: Dict[str, str] = {}  # agent_id -> accumulated content
        self._streaming_metadata: Dict[str, Dict[str, Any]] = {}  # agent_id -> metadata
        self._streaming_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        # Issue #4.4: Track stream start times for stale cleanup
        self._stream_start_times: Dict[str, float] = {}
        self._stream_timeout_seconds = stream_timeout_seconds

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
        # Issue #4.4: Track start time on first chunk for stale cleanup
        if agent_id not in self._stream_start_times:
            self._stream_start_times[agent_id] = time.time()

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
        self._stream_start_times.pop(agent_id, None)  # Issue #4.4: Clean up start time

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
        self._stream_start_times.clear()  # Issue #4.4
        logger.info("Cleared all streaming state")

    def get_stream_count(self) -> int:
        """Get count of active streaming sessions."""
        return len(self._partial_thoughts)

    # Issue #4.3: Stream cancellation support
    def cancel_stream(self, agent_id: str, reason: str = "user_cancelled") -> bool:
        """
        Cancel an active streaming session for an agent.

        Cleans up state and notifies callbacks of cancellation.

        Args:
            agent_id: The agent whose stream to cancel
            reason: Reason for cancellation (for logging/callbacks)

        Returns:
            True if stream was cancelled, False if no active stream
        """
        if agent_id not in self._partial_thoughts:
            return False

        # Capture state before cleanup
        accumulated = self._partial_thoughts.get(agent_id, "")
        metadata = self._streaming_metadata.get(agent_id, {})

        # Clean up state
        self._partial_thoughts.pop(agent_id, None)
        self._streaming_metadata.pop(agent_id, None)
        self._stream_start_times.pop(agent_id, None)

        # Notify callbacks of cancellation
        self._emit_streaming_event({
            "type": "stream_cancelled",
            "agent_id": agent_id,
            "partial_content": accumulated,
            "reason": reason,
            "agent_type": metadata.get("agent_type"),
            "timestamp": time.time()
        })

        logger.info(f"Cancelled streaming session for agent {agent_id} (reason: {reason})")
        return True

    def cancel_all_streams(self, reason: str = "bulk_cancel") -> int:
        """
        Cancel all active streams with callback notification.

        Unlike clear_all_streams(), this notifies callbacks of each cancellation.

        Args:
            reason: Reason for cancellation

        Returns:
            Number of streams cancelled
        """
        agent_ids = list(self._partial_thoughts.keys())
        for agent_id in agent_ids:
            self.cancel_stream(agent_id, reason)
        return len(agent_ids)

    # Issue #4.4: Stale stream cleanup
    def cleanup_stale_streams(self, max_age_seconds: Optional[float] = None) -> List[str]:
        """
        Remove streams older than max_age.

        Call periodically to prevent memory leaks from crashed agents.

        Args:
            max_age_seconds: Maximum stream age (uses default timeout if None)

        Returns:
            List of agent IDs whose streams were cleaned up
        """
        max_age = max_age_seconds if max_age_seconds is not None else self._stream_timeout_seconds
        now = time.time()
        stale = []

        for agent_id, start_time in list(self._stream_start_times.items()):
            age = now - start_time
            if age > max_age:
                stale.append(agent_id)
                self.cancel_stream(agent_id, reason=f"stale_timeout_{age:.1f}s")
                logger.warning(f"Cleaned up stale stream for agent {agent_id} (age: {age:.1f}s)")

        return stale

    def get_stream_age(self, agent_id: str) -> Optional[float]:
        """
        Get the age of a streaming session in seconds.

        Args:
            agent_id: Agent ID to query

        Returns:
            Age in seconds or None if not streaming
        """
        start_time = self._stream_start_times.get(agent_id)
        if start_time is None:
            return None
        return time.time() - start_time
