"""
Agent Performance Tracker - Records agent execution metrics for analytics.

Populates felix_agent_performance.db with checkpoint-level metrics for:
- Confidence evolution
- Token usage patterns
- Processing time analysis
- Helix position tracking
- Phase-based performance
"""

import sqlite3
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformanceRecord:
    """Single performance record for an agent at a checkpoint."""
    agent_id: str
    workflow_id: Optional[int]
    agent_type: str
    spawn_time: float
    checkpoint: float
    confidence: float
    tokens_used: int
    processing_time: float
    depth_ratio: float
    phase: str
    position_x: Optional[float]
    position_y: Optional[float]
    position_z: Optional[float]
    content_preview: Optional[str]
    timestamp: float


class AgentPerformanceTracker:
    """
    Tracks and stores agent performance metrics during workflow execution.

    Integrates with felix_workflow.py to record performance data at each
    agent checkpoint for later analysis and learning.
    """

    def __init__(self, db_path: Path = None):
        """
        Initialize performance tracker.

        Args:
            db_path: Path to felix_agent_performance.db
        """
        if db_path is None:
            db_path = Path("felix_agent_performance.db")

        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Ensure database exists."""
        if not self.db_path.exists():
            logger.warning(f"{self.db_path} does not exist - will be created")
            self.db_path.touch()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def record_agent_checkpoint(self,
                                agent_id: str,
                                agent_type: str,
                                spawn_time: float,
                                checkpoint: float,
                                confidence: float,
                                tokens_used: int,
                                processing_time: float,
                                depth_ratio: float,
                                phase: str,
                                workflow_id: Optional[int] = None,
                                position: Optional[tuple] = None,
                                content_preview: str = None) -> int:
        """
        Record agent performance at a checkpoint.

        Args:
            agent_id: Unique agent identifier
            agent_type: Type of agent (research, analysis, critic, synthesis)
            spawn_time: When agent was spawned (timestamp)
            checkpoint: Checkpoint number crossed
            confidence: Agent's confidence score
            tokens_used: Tokens consumed at this checkpoint
            processing_time: Time taken for this checkpoint (seconds)
            depth_ratio: Position along helix (0.0-1.0)
            phase: Current phase (exploration, analysis, synthesis)
            workflow_id: Optional workflow ID
            position: Optional (x, y, z) helix position
            content_preview: Optional preview of agent output

        Returns:
            Record ID of inserted performance entry
        """
        try:
            record = AgentPerformanceRecord(
                agent_id=agent_id,
                workflow_id=workflow_id,
                agent_type=agent_type,
                spawn_time=spawn_time,
                checkpoint=checkpoint,
                confidence=confidence,
                tokens_used=tokens_used,
                processing_time=processing_time,
                depth_ratio=depth_ratio,
                phase=phase,
                position_x=position[0] if position else None,
                position_y=position[1] if position else None,
                position_z=position[2] if position else None,
                content_preview=content_preview[:500] if content_preview else None,  # Limit preview
                timestamp=time.time()
            )

            conn = self._get_connection()
            cursor = conn.execute("""
                INSERT INTO agent_performance
                (agent_id, workflow_id, agent_type, spawn_time, checkpoint,
                 confidence, tokens_used, processing_time, depth_ratio, phase,
                 position_x, position_y, position_z, content_preview, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.agent_id,
                record.workflow_id,
                record.agent_type,
                record.spawn_time,
                record.checkpoint,
                record.confidence,
                record.tokens_used,
                record.processing_time,
                record.depth_ratio,
                record.phase,
                record.position_x,
                record.position_y,
                record.position_z,
                record.content_preview,
                record.timestamp
            ))

            record_id = cursor.lastrowid
            conn.commit()
            conn.close()

            logger.debug(f"Recorded performance for agent {agent_id[:8]}... at checkpoint {checkpoint}")
            return record_id

        except sqlite3.Error as e:
            logger.error(f"Failed to record agent performance: {e}")
            return -1

    def get_agent_performance_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Get performance history for a specific agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of performance records
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT * FROM agent_performance
                WHERE agent_id = ?
                ORDER BY checkpoint
            """, (agent_id,))

            records = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return records

        except sqlite3.Error as e:
            logger.error(f"Failed to get agent performance history: {e}")
            return []

    def get_workflow_agent_summary(self, workflow_id: int) -> Dict[str, Any]:
        """
        Get summary of agent performance for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Dictionary with performance summary
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            # Get aggregate metrics
            cursor = conn.execute("""
                SELECT
                    COUNT(DISTINCT agent_id) as agent_count,
                    AVG(confidence) as avg_confidence,
                    SUM(tokens_used) as total_tokens,
                    SUM(processing_time) as total_processing_time,
                    MAX(checkpoint) as max_checkpoint
                FROM agent_performance
                WHERE workflow_id = ?
            """, (workflow_id,))

            summary = dict(cursor.fetchone())

            # Get per-agent-type breakdown
            cursor = conn.execute("""
                SELECT
                    agent_type,
                    COUNT(*) as checkpoint_count,
                    AVG(confidence) as avg_confidence,
                    SUM(tokens_used) as tokens_used,
                    AVG(processing_time) as avg_processing_time
                FROM agent_performance
                WHERE workflow_id = ?
                GROUP BY agent_type
            """, (workflow_id,))

            summary['by_agent_type'] = [dict(row) for row in cursor.fetchall()]

            conn.close()
            return summary

        except sqlite3.Error as e:
            logger.error(f"Failed to get workflow agent summary: {e}")
            return {}

    def get_agent_type_statistics(self, agent_type: str, days: int = 30) -> Dict[str, Any]:
        """
        Get statistics for an agent type over time period.

        Args:
            agent_type: Type of agent
            days: Number of days to include

        Returns:
            Dictionary with statistics
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cutoff_time = time.time() - (days * 24 * 60 * 60)

            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_checkpoints,
                    COUNT(DISTINCT agent_id) as unique_agents,
                    AVG(confidence) as avg_confidence,
                    MIN(confidence) as min_confidence,
                    MAX(confidence) as max_confidence,
                    AVG(tokens_used) as avg_tokens,
                    AVG(processing_time) as avg_processing_time,
                    SUM(tokens_used) as total_tokens
                FROM agent_performance
                WHERE agent_type = ? AND timestamp > ?
            """, (agent_type, cutoff_time))

            stats = dict(cursor.fetchone())
            stats['agent_type'] = agent_type
            stats['days'] = days

            conn.close()
            return stats

        except sqlite3.Error as e:
            logger.error(f"Failed to get agent type statistics: {e}")
            return {}

    def get_phase_transition_analysis(self, workflow_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Analyze how agents transition through phases.

        Args:
            workflow_id: Optional workflow to filter by

        Returns:
            List of phase transition records
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            query = """
                SELECT
                    phase,
                    agent_type,
                    AVG(confidence) as avg_confidence,
                    AVG(depth_ratio) as avg_depth_ratio,
                    COUNT(*) as checkpoint_count
                FROM agent_performance
            """

            params = []
            if workflow_id is not None:
                query += " WHERE workflow_id = ?"
                params.append(workflow_id)

            query += " GROUP BY phase, agent_type ORDER BY phase, agent_type"

            cursor = conn.execute(query, params)
            records = [dict(row) for row in cursor.fetchall()]

            conn.close()
            return records

        except sqlite3.Error as e:
            logger.error(f"Failed to get phase transition analysis: {e}")
            return []

    def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """
        Clean up old performance records.

        Args:
            days_to_keep: Number of days to retain

        Returns:
            Number of records deleted
        """
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

            conn = self._get_connection()
            cursor = conn.execute("""
                DELETE FROM agent_performance
                WHERE timestamp < ?
            """, (cutoff_time,))

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Cleaned up {deleted_count} old performance records (older than {days_to_keep} days)")
            return deleted_count

        except sqlite3.Error as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0
