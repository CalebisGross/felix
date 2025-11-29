"""
Migration: Add watch_directories table

Adds a table to track watched directories with metadata and statistics.
This enables directory-level operations and better visibility into the knowledge base.
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate_up(db_path: str = "felix_knowledge.db") -> bool:
    """
    Add watch_directories table to track watched directories.

    Schema:
        - watch_id: Primary key
        - directory_path: Absolute path to watched directory (unique)
        - added_at: Unix timestamp when directory was added
        - enabled: Boolean indicating if directory is actively watched
        - last_scan: Unix timestamp of last scan
        - document_count: Number of documents from this directory
        - entry_count: Number of knowledge entries from this directory
        - notes: Optional user notes about this directory

    Args:
        db_path: Path to knowledge database

    Returns:
        True if migration successful, False otherwise
    """
    logger.info("Running migration: add_watch_directories_table")

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check if table already exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='watch_directories'
            """)

            if cursor.fetchone():
                logger.info("watch_directories table already exists, skipping migration")
                return True

            # Create watch_directories table
            cursor.execute("""
                CREATE TABLE watch_directories (
                    watch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    directory_path TEXT NOT NULL UNIQUE,
                    added_at REAL NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    last_scan REAL,
                    document_count INTEGER DEFAULT 0,
                    entry_count INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)

            # Create index on directory_path for fast lookups
            cursor.execute("""
                CREATE INDEX idx_watch_directories_path
                ON watch_directories(directory_path)
            """)

            # Create index on enabled for filtering
            cursor.execute("""
                CREATE INDEX idx_watch_directories_enabled
                ON watch_directories(enabled)
            """)

            conn.commit()
            logger.info("Successfully created watch_directories table")

            # Populate with existing watch directories from document_sources
            _populate_initial_directories(conn)

            return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def _populate_initial_directories(conn: sqlite3.Connection):
    """
    Populate watch_directories with existing directories from document_sources.

    Args:
        conn: Database connection
    """
    try:
        cursor = conn.cursor()

        # Get unique directory paths from document_sources
        cursor.execute("""
            SELECT DISTINCT
                SUBSTR(file_path, 1,
                    CASE
                        WHEN INSTR(SUBSTR(file_path, 2), '/') > 0
                        THEN INSTR(SUBSTR(file_path, 2), '/') + 1
                        ELSE LENGTH(file_path)
                    END
                ) as dir_path,
                COUNT(*) as doc_count,
                MIN(added_at) as first_added
            FROM document_sources
            GROUP BY dir_path
            HAVING doc_count > 0
        """)

        directories = cursor.fetchall()

        if not directories:
            logger.info("No existing directories to populate")
            return

        # Insert directories
        for dir_path, doc_count, first_added in directories:
            # Try to extract a reasonable directory path
            # This is a best-effort since we don't have the original watch directory
            path = Path(dir_path).parent if dir_path else None

            if not path or not path.exists():
                continue

            try:
                # Get entry count for this directory
                cursor.execute("""
                    SELECT COUNT(DISTINCT ke.knowledge_id)
                    FROM knowledge_entries ke
                    JOIN document_sources ds ON ke.source_doc_id = ds.doc_id
                    WHERE ds.file_path LIKE ?
                """, (f"{path}%",))

                entry_count = cursor.fetchone()[0]

                # Insert watch directory
                cursor.execute("""
                    INSERT OR IGNORE INTO watch_directories
                    (directory_path, added_at, enabled, document_count, entry_count, notes)
                    VALUES (?, ?, 1, ?, ?, ?)
                """, (
                    str(path.absolute()),
                    first_added or 0,
                    doc_count,
                    entry_count,
                    "Auto-populated from existing documents"
                ))

            except Exception as e:
                logger.warning(f"Could not populate directory {path}: {e}")
                continue

        conn.commit()
        logger.info(f"Populated {len(directories)} existing directories")

    except Exception as e:
        logger.error(f"Error populating initial directories: {e}")


def migrate_down(db_path: str = "felix_knowledge.db") -> bool:
    """
    Remove watch_directories table.

    Args:
        db_path: Path to knowledge database

    Returns:
        True if rollback successful, False otherwise
    """
    logger.info("Rolling back migration: add_watch_directories_table")

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Drop indexes
            cursor.execute("DROP INDEX IF EXISTS idx_watch_directories_path")
            cursor.execute("DROP INDEX IF EXISTS idx_watch_directories_enabled")

            # Drop table
            cursor.execute("DROP TABLE IF EXISTS watch_directories")

            conn.commit()
            logger.info("Successfully removed watch_directories table")
            return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
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
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Check table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='watch_directories'
            """)

            if not cursor.fetchone():
                logger.error("Verification failed: watch_directories table not found")
                return False

            # Check schema
            cursor.execute("PRAGMA table_info(watch_directories)")
            columns = {row[1] for row in cursor.fetchall()}

            expected_columns = {
                'watch_id', 'directory_path', 'added_at', 'enabled',
                'last_scan', 'document_count', 'entry_count', 'notes'
            }

            if not expected_columns.issubset(columns):
                missing = expected_columns - columns
                logger.error(f"Verification failed: missing columns {missing}")
                return False

            # Check indexes
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='watch_directories'
            """)
            indexes = {row[0] for row in cursor.fetchall()}

            expected_indexes = {
                'idx_watch_directories_path',
                'idx_watch_directories_enabled'
            }

            if not expected_indexes.issubset(indexes):
                missing = expected_indexes - indexes
                logger.error(f"Verification failed: missing indexes {missing}")
                return False

            logger.info("Migration verification passed")
            return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
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
        else:
            print("Verification failed!")
            sys.exit(1)
    else:
        print("Migration failed!")
        sys.exit(1)
