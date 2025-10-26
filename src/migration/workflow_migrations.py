"""
Migrations for felix_workflow_history.db (WorkflowHistory).

These migrations:
- Add composite indexes for common filter patterns
- Add full-text search for workflow tasks and synthesis results
"""

import sqlite3
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class WorkflowMigration001(Migration):
    """Add composite indexes for common filter patterns."""

    version = 1
    description = "Add composite indexes for efficient workflow queries"

    def up(self, conn: sqlite3.Connection):
        """Create composite indexes."""
        logger.info("Creating composite indexes on workflow_outputs...")

        # Common filter: status + date (e.g., "show completed workflows from last week")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status_created
            ON workflow_outputs(status, created_at DESC)
        """)

        # Conversation threading: thread + date
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_created
            ON workflow_outputs(conversation_thread_id, created_at DESC)
        """)

        # Quality filtering: confidence + date
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence_time
            ON workflow_outputs(confidence DESC, created_at DESC)
        """)

        logger.info("✓ Created 3 composite indexes")

    def down(self, conn: sqlite3.Connection):
        """Remove composite indexes."""
        conn.execute("DROP INDEX IF EXISTS idx_status_created")
        conn.execute("DROP INDEX IF EXISTS idx_thread_created")
        conn.execute("DROP INDEX IF EXISTS idx_confidence_time")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify indexes were created."""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_status_created',
                'idx_thread_created',
                'idx_confidence_time'
            )
        """)
        indexes = cursor.fetchall()
        return len(indexes) == 3


class WorkflowMigration002(Migration):
    """Add full-text search for workflow tasks and synthesis results."""

    version = 2
    description = "Add FTS5 virtual table for workflow search"

    def up(self, conn: sqlite3.Connection):
        """Create FTS5 table and triggers."""
        logger.info("Creating FTS5 virtual table for workflow_outputs...")

        # Create FTS5 virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS workflow_fts
            USING fts5(
                workflow_id UNINDEXED,
                task_input,
                final_synthesis,
                tokenize='porter unicode61'
            )
        """)

        logger.info("✓ Created workflow_fts virtual table")

        # Populate FTS table from existing data
        logger.info("Populating FTS table from existing workflows...")
        cursor = conn.execute("SELECT COUNT(*) FROM workflow_outputs")
        total_entries = cursor.fetchone()[0]

        if total_entries > 0:
            conn.execute("""
                INSERT INTO workflow_fts(workflow_id, task_input, final_synthesis)
                SELECT workflow_id, task_input, COALESCE(final_synthesis, '')
                FROM workflow_outputs
            """)
            logger.info(f"✓ Populated FTS table with {total_entries} entries")
        else:
            logger.info("No existing entries to populate")

        # Create triggers
        logger.info("Creating triggers to maintain FTS sync...")

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS workflow_fts_insert
            AFTER INSERT ON workflow_outputs BEGIN
                INSERT INTO workflow_fts(workflow_id, task_input, final_synthesis)
                VALUES (new.workflow_id, new.task_input, COALESCE(new.final_synthesis, ''));
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS workflow_fts_update
            AFTER UPDATE ON workflow_outputs BEGIN
                UPDATE workflow_fts
                SET task_input = new.task_input,
                    final_synthesis = COALESCE(new.final_synthesis, '')
                WHERE workflow_id = new.workflow_id;
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS workflow_fts_delete
            AFTER DELETE ON workflow_outputs BEGIN
                DELETE FROM workflow_fts WHERE workflow_id = old.workflow_id;
            END
        """)

        logger.info("✓ Created 3 triggers (insert, update, delete)")

    def down(self, conn: sqlite3.Connection):
        """Remove FTS5 table and triggers."""
        conn.execute("DROP TRIGGER IF EXISTS workflow_fts_insert")
        conn.execute("DROP TRIGGER IF EXISTS workflow_fts_update")
        conn.execute("DROP TRIGGER IF EXISTS workflow_fts_delete")
        conn.execute("DROP TABLE IF EXISTS workflow_fts")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify FTS table and triggers were created."""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='workflow_fts'
        """)
        if cursor.fetchone() is None:
            return False

        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name LIKE 'workflow_fts_%'
        """)
        triggers = cursor.fetchall()
        return len(triggers) == 3


def get_migrations() -> List[Migration]:
    """
    Get all workflow history migrations in order.

    Returns:
        List of Migration instances
    """
    return [
        WorkflowMigration001(),
        WorkflowMigration002()
    ]
