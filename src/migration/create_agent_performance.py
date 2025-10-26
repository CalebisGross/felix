"""
Create felix_agent_performance.db (AgentPerformanceStore).

This is a new database for tracking agent performance metrics
at checkpoint level for analytics and optimization.
"""

import sqlite3
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class AgentPerformanceMigration001(Migration):
    """Create agent performance tracking database."""

    version = 1
    description = "Create agent_performance table with indexes for analytics"

    def up(self, conn: sqlite3.Connection):
        """Create agent performance schema."""
        logger.info("Creating agent_performance table...")

        # Main performance tracking table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                workflow_id INTEGER,
                agent_type TEXT NOT NULL,
                spawn_time REAL NOT NULL,
                checkpoint REAL NOT NULL,
                confidence REAL NOT NULL,
                tokens_used INTEGER NOT NULL,
                processing_time REAL NOT NULL,
                depth_ratio REAL NOT NULL,
                phase TEXT NOT NULL,
                position_x REAL,
                position_y REAL,
                position_z REAL,
                content_preview TEXT,
                timestamp REAL NOT NULL
            )
        """)

        logger.info("✓ Created agent_performance table")

        # Create indexes for common queries
        logger.info("Creating indexes...")

        # Agent type analysis
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_type_confidence
            ON agent_performance(agent_type, confidence)
        """)

        # Workflow-specific queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_agents
            ON agent_performance(workflow_id)
        """)

        # Phase analysis (helix progression)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_phase_performance
            ON agent_performance(phase, confidence, tokens_used)
        """)

        # Time-based queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON agent_performance(timestamp DESC)
        """)

        # Agent-specific tracking
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_workflow_checkpoint
            ON agent_performance(agent_id, workflow_id, checkpoint)
        """)

        logger.info("✓ Created 5 indexes")

    def down(self, conn: sqlite3.Connection):
        """Remove agent performance schema."""
        conn.execute("DROP TABLE IF EXISTS agent_performance")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify table and indexes were created."""
        # Check table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='agent_performance'
        """)
        if cursor.fetchone() is None:
            return False

        # Check indexes exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_agent_type_confidence',
                'idx_workflow_agents',
                'idx_phase_performance',
                'idx_timestamp',
                'idx_agent_workflow_checkpoint'
            )
        """)
        indexes = cursor.fetchall()
        return len(indexes) == 5


def get_migrations() -> List[Migration]:
    """
    Get all agent performance migrations in order.

    Returns:
        List of Migration instances
    """
    return [
        AgentPerformanceMigration001()
    ]
