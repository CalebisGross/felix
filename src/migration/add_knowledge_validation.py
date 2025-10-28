"""
Add validation fields to knowledge_entries for quality control.

This migration extends the existing knowledge_entries table in
felix_knowledge.db with validation scoring and flagging capabilities.
"""

import sqlite3
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class KnowledgeValidationMigration001(Migration):
    """Add validation_score and validation_flags to knowledge_entries."""

    version = 1
    description = "Add validation fields to knowledge_entries for quality control"

    def up(self, conn: sqlite3.Connection):
        """Add validation fields to knowledge_entries."""
        logger.info("Adding validation fields to knowledge_entries...")

        # Add validation_score column (0.0 to 1.0)
        conn.execute("""
            ALTER TABLE knowledge_entries
            ADD COLUMN validation_score REAL DEFAULT 1.0
        """)
        logger.info("✓ Added validation_score column")

        # Add validation_flags column (JSON array of issue types)
        conn.execute("""
            ALTER TABLE knowledge_entries
            ADD COLUMN validation_flags TEXT DEFAULT '[]'
        """)
        logger.info("✓ Added validation_flags column")

        # Add validation_status column (trusted/flagged/review/quarantine)
        conn.execute("""
            ALTER TABLE knowledge_entries
            ADD COLUMN validation_status TEXT DEFAULT 'trusted'
                CHECK(validation_status IN ('trusted', 'flagged', 'review', 'quarantine'))
        """)
        logger.info("✓ Added validation_status column")

        # Add validated_at timestamp
        conn.execute("""
            ALTER TABLE knowledge_entries
            ADD COLUMN validated_at REAL
        """)
        logger.info("✓ Added validated_at column")

        # Create index for validation queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_validation_status
            ON knowledge_entries(validation_status, validation_score DESC)
        """)

        # Create index for flagged entries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_validation_score
            ON knowledge_entries(validation_score DESC)
            WHERE validation_status != 'trusted'
        """)

        logger.info("✓ Created 2 validation indexes")

        # Summary
        logger.info("\n" + "="*60)
        logger.info("KNOWLEDGE VALIDATION FIELDS ADDED")
        logger.info("="*60)
        logger.info("New columns:")
        logger.info("  - validation_score (0.0-1.0)")
        logger.info("  - validation_flags (JSON)")
        logger.info("  - validation_status (trusted/flagged/review/quarantine)")
        logger.info("  - validated_at (timestamp)")
        logger.info("New indexes: 2")
        logger.info("="*60)

    def down(self, conn: sqlite3.Connection):
        """
        Remove validation fields.

        Note: SQLite doesn't support DROP COLUMN easily.
        This would require recreating the table.
        """
        logger.warning("Downgrade not fully supported - would require table recreation")
        # Drop indexes only
        conn.execute("DROP INDEX IF EXISTS idx_validation_status")
        conn.execute("DROP INDEX IF EXISTS idx_validation_score")
        logger.info("✓ Dropped validation indexes (columns remain)")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify validation fields were added."""
        # Check columns exist
        cursor = conn.execute("PRAGMA table_info(knowledge_entries)")
        columns = {row[1]: row for row in cursor.fetchall()}

        required_columns = ['validation_score', 'validation_flags', 'validation_status', 'validated_at']
        for col in required_columns:
            if col not in columns:
                logger.error(f"Column {col} not found in knowledge_entries")
                return False

        # Check indexes exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_validation_status',
                'idx_validation_score'
            )
        """)
        indexes = cursor.fetchall()
        if len(indexes) != 2:
            logger.error(f"Expected 2 validation indexes, found {len(indexes)}")
            return False

        logger.info("✓ All validation fields and indexes verified")
        return True


def get_migrations() -> List[Migration]:
    """
    Get all knowledge validation migrations in order.

    Returns:
        List of Migration instances
    """
    return [
        KnowledgeValidationMigration001()
    ]
