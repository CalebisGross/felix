"""
Add agent_lifecycle_events table to felix_system_actions.db.

This migration creates a table for tracking agent lifecycle events
(spawn, complete, fail) for full traceability and GUI observability.
"""

import sqlite3
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class AgentLifecycleEventsMigration(Migration):
    """Create agent lifecycle events table in system actions database."""

    version = 3  # Follows SystemActionsMigration002
    description = "Create agent_lifecycle_events table for tracking agent spawn/complete/fail events"

    def up(self, conn: sqlite3.Connection):
        """Create agent lifecycle events schema."""
        logger.info("Creating agent_lifecycle_events table...")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_lifecycle_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                event_type TEXT NOT NULL,
                workflow_id TEXT,
                timestamp REAL NOT NULL,
                duration REAL,
                metadata_json TEXT,
                error_message TEXT,
                exit_status TEXT
            )
        """)

        # Index for querying by agent
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lifecycle_agent
            ON agent_lifecycle_events(agent_id, timestamp DESC)
        """)

        # Index for querying by event type
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lifecycle_event_type
            ON agent_lifecycle_events(event_type, timestamp DESC)
        """)

        # Index for querying by workflow
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lifecycle_workflow
            ON agent_lifecycle_events(workflow_id, timestamp DESC)
        """)

        # Index for recent events (GUI queries)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_lifecycle_timestamp
            ON agent_lifecycle_events(timestamp DESC)
        """)

        logger.info("Created agent_lifecycle_events with 4 indexes")

    def down(self, conn: sqlite3.Connection):
        """Remove agent lifecycle events table."""
        conn.execute("DROP INDEX IF EXISTS idx_lifecycle_timestamp")
        conn.execute("DROP INDEX IF EXISTS idx_lifecycle_workflow")
        conn.execute("DROP INDEX IF EXISTS idx_lifecycle_event_type")
        conn.execute("DROP INDEX IF EXISTS idx_lifecycle_agent")
        conn.execute("DROP TABLE IF EXISTS agent_lifecycle_events")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify table and indexes were created."""
        # Check table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='agent_lifecycle_events'
        """)
        if not cursor.fetchone():
            logger.error("agent_lifecycle_events table not found")
            return False

        # Check indexes exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name LIKE 'idx_lifecycle_%'
        """)
        indexes = cursor.fetchall()
        if len(indexes) < 4:
            logger.error(f"Expected 4 indexes, found {len(indexes)}")
            return False

        return True


def get_migrations() -> List[Migration]:
    """Get all agent lifecycle migrations."""
    return [AgentLifecycleEventsMigration()]
