"""
Migration: Add Knowledge Audit Log Table

Creates a comprehensive audit log table to track all CRUD operations on
knowledge entries. This enables full traceability of changes, supports
compliance requirements, and helps debug data integrity issues.

Features:
- Records INSERT, UPDATE, DELETE, MERGE operations
- Captures before/after state for updates
- Tracks user/agent performing operation
- Transaction-level grouping
- Indexed for fast querying
"""

import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def migrate_up(db_path: str = "felix_knowledge.db") -> bool:
    """
    Create the knowledge_audit_log table with indexes.

    Schema:
    - audit_id: Auto-incrementing primary key
    - timestamp: When operation occurred (REAL for SQLite datetime)
    - operation: INSERT, UPDATE, DELETE, MERGE, CLEANUP
    - knowledge_id: ID of affected entry (indexed)
    - user_agent: Who performed operation (e.g., "ResearchAgent", "GUI User")
    - old_values_json: Previous state (NULL for INSERT)
    - new_values_json: New state (NULL for DELETE)
    - details: Human-readable description
    - transaction_id: Groups related operations (e.g., merge operations)

    Args:
        db_path: Path to knowledge database

    Returns:
        True if migration successful, False otherwise
    """
    logger.info("Running migration: add_audit_log_table")

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # Check if table already exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='knowledge_audit_log'
        """)

        if cursor.fetchone():
            logger.info("knowledge_audit_log table already exists, skipping migration")
            conn.close()
            return True

        # Create audit log table
        logger.info("Creating knowledge_audit_log table...")
        cursor.execute("""
            CREATE TABLE knowledge_audit_log (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                operation TEXT NOT NULL,
                knowledge_id TEXT,
                user_agent TEXT,
                old_values_json TEXT,
                new_values_json TEXT,
                details TEXT,
                transaction_id TEXT
            )
        """)

        # Create indexes for fast querying
        logger.info("Creating indexes...")
        cursor.execute("""
            CREATE INDEX idx_audit_timestamp ON knowledge_audit_log(timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX idx_audit_knowledge_id ON knowledge_audit_log(knowledge_id)
        """)

        cursor.execute("""
            CREATE INDEX idx_audit_operation ON knowledge_audit_log(operation)
        """)

        cursor.execute("""
            CREATE INDEX idx_audit_transaction_id ON knowledge_audit_log(transaction_id)
        """)

        conn.commit()
        logger.info("Successfully created knowledge_audit_log table with indexes")

        # Insert initial audit entry to mark migration
        cursor.execute("""
            INSERT INTO knowledge_audit_log (
                timestamp, operation, user_agent, details
            ) VALUES (?, ?, ?, ?)
        """, (
            datetime.now().timestamp(),
            'SYSTEM',
            'Migration',
            'Audit log table created - tracking enabled'
        ))
        conn.commit()

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


def migrate_down(db_path: str = "felix_knowledge.db") -> bool:
    """
    Remove the knowledge_audit_log table and all indexes.

    WARNING: This will permanently delete all audit history!

    Args:
        db_path: Path to knowledge database

    Returns:
        True if rollback successful, False otherwise
    """
    logger.info("Rolling back migration: add_audit_log_table")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop indexes first
        cursor.execute("DROP INDEX IF EXISTS idx_audit_timestamp")
        cursor.execute("DROP INDEX IF EXISTS idx_audit_knowledge_id")
        cursor.execute("DROP INDEX IF EXISTS idx_audit_operation")
        cursor.execute("DROP INDEX IF EXISTS idx_audit_transaction_id")

        # Drop table
        cursor.execute("DROP TABLE IF EXISTS knowledge_audit_log")

        conn.commit()
        conn.close()
        logger.info("Successfully removed knowledge_audit_log table")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


def verify_migration(db_path: str = "felix_knowledge.db") -> bool:
    """
    Verify that the migration was successful.

    Checks:
    1. Table exists with correct schema
    2. All indexes exist
    3. Initial audit entry was created

    Args:
        db_path: Path to knowledge database

    Returns:
        True if verification passed, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='knowledge_audit_log'
        """)

        if not cursor.fetchone():
            logger.error("Verification failed: knowledge_audit_log table not found")
            conn.close()
            return False

        # Check schema has required columns
        cursor.execute("PRAGMA table_info(knowledge_audit_log)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {
            'audit_id', 'timestamp', 'operation', 'knowledge_id',
            'user_agent', 'old_values_json', 'new_values_json',
            'details', 'transaction_id'
        }

        if not required_columns.issubset(columns):
            missing = required_columns - columns
            logger.error(f"Verification failed: missing columns {missing}")
            conn.close()
            return False

        # Check indexes exist
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='knowledge_audit_log'
        """)

        indexes = {row[0] for row in cursor.fetchall()}
        required_indexes = {
            'idx_audit_timestamp',
            'idx_audit_knowledge_id',
            'idx_audit_operation',
            'idx_audit_transaction_id'
        }

        # SQLite auto-creates index for primary key, so we exclude it
        if not required_indexes.issubset(indexes):
            missing = required_indexes - indexes
            logger.error(f"Verification failed: missing indexes {missing}")
            conn.close()
            return False

        # Check initial audit entry exists
        cursor.execute("""
            SELECT COUNT(*) FROM knowledge_audit_log
            WHERE operation='SYSTEM' AND details LIKE '%Audit log table created%'
        """)

        if cursor.fetchone()[0] == 0:
            logger.warning("Initial audit entry not found (may have been deleted)")

        conn.close()
        logger.info("Migration verification passed")
        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


if __name__ == "__main__":
    # Run migration
    import sys

    logging.basicConfig(level=logging.INFO)

    db_path = sys.argv[1] if len(sys.argv) > 1 else "felix_knowledge.db"

    print(f"Running migration on {db_path}...")

    if migrate_up(db_path):
        print("Migration successful!")

        if verify_migration(db_path):
            print("Verification passed!")
            print("\nAudit logging is now enabled. All CRUD operations on")
            print("knowledge entries will be tracked in knowledge_audit_log table.")
        else:
            print("Verification failed!")
            sys.exit(1)
    else:
        print("Migration failed!")
        sys.exit(1)
