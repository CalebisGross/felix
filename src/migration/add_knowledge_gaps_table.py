"""
Migration: Add Knowledge Gaps Table

Creates a table to track knowledge gaps discovered during workflows.
This enables epistemic self-awareness (Phase 6 - Knowledge Gap Cartography).

Features:
- Tracks domains and concepts where knowledge is lacking
- Records gap severity and occurrence count
- Correlates gaps with workflow outcomes
- Enables gap-directed learning (proactive knowledge acquisition)
"""

import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def migrate_up(db_path: str = "felix_knowledge.db") -> bool:
    """
    Create the knowledge_gaps table with indexes.

    Schema:
    - gap_id: Primary key (TEXT for UUID)
    - domain: Domain where gap was identified (indexed)
    - concept: Specific concept lacking (optional)
    - first_seen: When gap was first detected
    - last_seen: Most recent occurrence
    - occurrence_count: How many times gap affected workflows
    - impact_severity_avg: Running average of gap impact (0.0-1.0)
    - resolved: Whether gap has been filled
    - resolution_method: How gap was resolved (web_search, manual, etc.)
    - resolution_timestamp: When gap was resolved

    Args:
        db_path: Path to knowledge database

    Returns:
        True if migration successful, False otherwise
    """
    logger.info("Running migration: add_knowledge_gaps_table")

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # Check if table already exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='knowledge_gaps'
        """)

        if cursor.fetchone():
            logger.info("knowledge_gaps table already exists, skipping migration")
            conn.close()
            return True

        # Create knowledge gaps table
        logger.info("Creating knowledge_gaps table...")
        cursor.execute("""
            CREATE TABLE knowledge_gaps (
                gap_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                concept TEXT,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                occurrence_count INTEGER DEFAULT 1,
                impact_severity_avg REAL DEFAULT 0.5,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_method TEXT,
                resolution_timestamp REAL
            )
        """)

        # Create gap-workflow correlation table
        logger.info("Creating gap_workflow_correlation table...")
        cursor.execute("""
            CREATE TABLE gap_workflow_correlation (
                correlation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                gap_id TEXT NOT NULL,
                workflow_id TEXT NOT NULL,
                detected_at REAL NOT NULL,
                workflow_confidence REAL,
                impact_severity REAL,
                FOREIGN KEY (gap_id) REFERENCES knowledge_gaps(gap_id) ON DELETE CASCADE
            )
        """)

        # Create indexes for fast querying
        logger.info("Creating indexes...")
        cursor.execute("""
            CREATE INDEX idx_gaps_domain ON knowledge_gaps(domain)
        """)

        cursor.execute("""
            CREATE INDEX idx_gaps_resolved ON knowledge_gaps(resolved)
        """)

        cursor.execute("""
            CREATE INDEX idx_gaps_severity ON knowledge_gaps(impact_severity_avg DESC)
        """)

        cursor.execute("""
            CREATE INDEX idx_gaps_occurrences ON knowledge_gaps(occurrence_count DESC)
        """)

        cursor.execute("""
            CREATE INDEX idx_gap_workflow_gap_id ON gap_workflow_correlation(gap_id)
        """)

        cursor.execute("""
            CREATE INDEX idx_gap_workflow_workflow_id ON gap_workflow_correlation(workflow_id)
        """)

        conn.commit()
        logger.info("Successfully created knowledge_gaps and gap_workflow_correlation tables")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


def migrate_down(db_path: str = "felix_knowledge.db") -> bool:
    """
    Remove the knowledge_gaps tables and indexes.

    WARNING: This will permanently delete all gap tracking history!

    Args:
        db_path: Path to knowledge database

    Returns:
        True if rollback successful, False otherwise
    """
    logger.info("Rolling back migration: add_knowledge_gaps_table")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop correlation table first (has foreign key)
        cursor.execute("DROP TABLE IF EXISTS gap_workflow_correlation")

        # Drop indexes (automatically dropped with table, but be explicit)
        cursor.execute("DROP INDEX IF EXISTS idx_gaps_domain")
        cursor.execute("DROP INDEX IF EXISTS idx_gaps_resolved")
        cursor.execute("DROP INDEX IF EXISTS idx_gaps_severity")
        cursor.execute("DROP INDEX IF EXISTS idx_gaps_occurrences")

        # Drop main table
        cursor.execute("DROP TABLE IF EXISTS knowledge_gaps")

        conn.commit()
        conn.close()
        logger.info("Successfully removed knowledge_gaps tables")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


def verify_migration(db_path: str = "felix_knowledge.db") -> bool:
    """
    Verify that the migration was successful.

    Args:
        db_path: Path to knowledge database

    Returns:
        True if verification passed, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check main table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='knowledge_gaps'
        """)

        if not cursor.fetchone():
            logger.error("Verification failed: knowledge_gaps table not found")
            conn.close()
            return False

        # Check correlation table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='gap_workflow_correlation'
        """)

        if not cursor.fetchone():
            logger.error("Verification failed: gap_workflow_correlation table not found")
            conn.close()
            return False

        # Check schema has required columns
        cursor.execute("PRAGMA table_info(knowledge_gaps)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {
            'gap_id', 'domain', 'concept', 'first_seen', 'last_seen',
            'occurrence_count', 'impact_severity_avg', 'resolved'
        }

        if not required_columns.issubset(columns):
            missing = required_columns - columns
            logger.error(f"Verification failed: missing columns {missing}")
            conn.close()
            return False

        conn.close()
        logger.info("Migration verification passed")
        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    db_path = sys.argv[1] if len(sys.argv) > 1 else "felix_knowledge.db"

    print(f"Running migration on {db_path}...")

    if migrate_up(db_path):
        print("Migration successful!")

        if verify_migration(db_path):
            print("Verification passed!")
            print("\nKnowledge gap tracking is now enabled.")
        else:
            print("Verification failed!")
            sys.exit(1)
    else:
        print("Migration failed!")
        sys.exit(1)
