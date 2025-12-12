"""
Migration: Add FTS5 Auto-Sync Triggers

Adds database triggers to automatically keep the FTS5 full-text search index
synchronized with the knowledge_entries table. This ensures that any INSERT,
UPDATE, or DELETE operations on knowledge_entries are immediately reflected
in the knowledge_fts virtual table.

Background:
- FTS5 index was previously populated only during initial migration
- Old triggers were removed by fix_fts_triggers.py due to schema mismatch
- This migration creates correct triggers using current schema (content_json)
"""

import sqlite3
import logging

logger = logging.getLogger(__name__)


def migrate_up(db_path: str = "felix_knowledge.db") -> bool:
    """
    Create triggers to automatically sync knowledge_fts with knowledge_entries.

    Creates three triggers:
    - knowledge_entries_ai: AFTER INSERT - adds new entries to FTS5
    - knowledge_entries_au: AFTER UPDATE - updates FTS5 on content changes
    - knowledge_entries_ad: AFTER DELETE - removes entries from FTS5

    Args:
        db_path: Path to knowledge database

    Returns:
        True if migration successful, False otherwise
    """
    logger.info("Running migration: add_fts5_triggers")

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # Check if triggers already exist
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name IN (
                'knowledge_entries_ai',
                'knowledge_entries_au',
                'knowledge_entries_ad'
            )
        """)

        existing_triggers = {row[0] for row in cursor.fetchall()}

        if len(existing_triggers) == 3:
            logger.info("FTS5 triggers already exist, skipping migration")
            conn.close()
            return True

        # Drop any existing triggers first (cleanup from old migrations)
        logger.info("Dropping any existing FTS5 triggers...")
        for trigger_name in ['knowledge_entries_ai', 'knowledge_entries_au', 'knowledge_entries_ad']:
            cursor.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")

        # Create AFTER INSERT trigger
        logger.info("Creating AFTER INSERT trigger...")
        cursor.execute("""
            CREATE TRIGGER knowledge_entries_ai AFTER INSERT ON knowledge_entries
            BEGIN
                INSERT INTO knowledge_fts(knowledge_id, content, domain, tags)
                VALUES(
                    new.knowledge_id,
                    CASE
                        WHEN json_valid(new.content_json) THEN
                            COALESCE(json_extract(new.content_json, '$.concept'), '') || ' ' ||
                            COALESCE(json_extract(new.content_json, '$.definition'), '') || ' ' ||
                            COALESCE(json_extract(new.content_json, '$.summary'), '')
                        ELSE COALESCE(new.content_json, '')
                    END,
                    COALESCE(new.domain, ''),
                    COALESCE(new.tags_json, '')
                );
            END;
        """)

        # Create AFTER UPDATE trigger
        logger.info("Creating AFTER UPDATE trigger...")
        cursor.execute("""
            CREATE TRIGGER knowledge_entries_au AFTER UPDATE ON knowledge_entries
            BEGIN
                UPDATE knowledge_fts
                SET
                    content = CASE
                        WHEN json_valid(new.content_json) THEN
                            COALESCE(json_extract(new.content_json, '$.concept'), '') || ' ' ||
                            COALESCE(json_extract(new.content_json, '$.definition'), '') || ' ' ||
                            COALESCE(json_extract(new.content_json, '$.summary'), '')
                        ELSE COALESCE(new.content_json, '')
                    END,
                    domain = COALESCE(new.domain, ''),
                    tags = COALESCE(new.tags_json, '')
                WHERE knowledge_id = old.knowledge_id;
            END;
        """)

        # Create AFTER DELETE trigger
        logger.info("Creating AFTER DELETE trigger...")
        cursor.execute("""
            CREATE TRIGGER knowledge_entries_ad AFTER DELETE ON knowledge_entries
            BEGIN
                DELETE FROM knowledge_fts WHERE knowledge_id = old.knowledge_id;
            END;
        """)

        conn.commit()
        logger.info("Successfully created FTS5 auto-sync triggers")

        # Verify FTS5 is in sync (rebuild if needed)
        cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
        entries_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge_fts")
        fts_count = cursor.fetchone()[0]

        if entries_count != fts_count:
            logger.warning(
                f"FTS5 out of sync: {entries_count} entries vs {fts_count} FTS rows. "
                f"Rebuilding FTS5 index..."
            )
            _rebuild_fts5_index(conn)

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


def _rebuild_fts5_index(conn: sqlite3.Connection):
    """
    Rebuild the FTS5 index from scratch.

    Args:
        conn: Database connection
    """
    try:
        cursor = conn.cursor()

        # Clear existing FTS5 data
        cursor.execute("DELETE FROM knowledge_fts")

        # Repopulate from knowledge_entries
        cursor.execute("""
            INSERT INTO knowledge_fts (knowledge_id, content, domain, tags)
            SELECT
                knowledge_id,
                CASE
                    WHEN json_valid(content_json) THEN
                        COALESCE(json_extract(content_json, '$.concept'), '') || ' ' ||
                        COALESCE(json_extract(content_json, '$.definition'), '') || ' ' ||
                        COALESCE(json_extract(content_json, '$.summary'), '')
                    ELSE COALESCE(content_json, '')
                END,
                COALESCE(domain, ''),
                COALESCE(tags_json, '')
            FROM knowledge_entries
            WHERE content_json IS NOT NULL
        """)

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM knowledge_fts")
        new_count = cursor.fetchone()[0]
        logger.info(f"FTS5 index rebuilt with {new_count} entries")

    except Exception as e:
        logger.error(f"Failed to rebuild FTS5 index: {e}")
        raise


def migrate_down(db_path: str = "felix_knowledge.db") -> bool:
    """
    Remove FTS5 auto-sync triggers.

    Args:
        db_path: Path to knowledge database

    Returns:
        True if rollback successful, False otherwise
    """
    logger.info("Rolling back migration: add_fts5_triggers")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop all three triggers
        cursor.execute("DROP TRIGGER IF EXISTS knowledge_entries_ai")
        cursor.execute("DROP TRIGGER IF EXISTS knowledge_entries_au")
        cursor.execute("DROP TRIGGER IF EXISTS knowledge_entries_ad")

        conn.commit()
        conn.close()
        logger.info("Successfully removed FTS5 auto-sync triggers")
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
    1. All three triggers exist
    2. Triggers have correct definitions
    3. FTS5 index is in sync with knowledge_entries

    Args:
        db_path: Path to knowledge database

    Returns:
        True if verification passed, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check triggers exist
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name IN (
                'knowledge_entries_ai',
                'knowledge_entries_au',
                'knowledge_entries_ad'
            )
        """)

        triggers = {row[0] for row in cursor.fetchall()}
        expected_triggers = {'knowledge_entries_ai', 'knowledge_entries_au', 'knowledge_entries_ad'}

        if triggers != expected_triggers:
            missing = expected_triggers - triggers
            logger.error(f"Verification failed: missing triggers {missing}")
            conn.close()
            return False

        # Check FTS5 is in sync
        cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
        entries_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge_fts")
        fts_count = cursor.fetchone()[0]

        if entries_count != fts_count:
            logger.error(
                f"Verification failed: FTS5 out of sync "
                f"({entries_count} entries vs {fts_count} FTS rows)"
            )
            conn.close()
            return False

        # Test INSERT trigger by checking trigger definition
        cursor.execute("""
            SELECT sql FROM sqlite_master
            WHERE type='trigger' AND name='knowledge_entries_ai'
        """)
        trigger_sql = cursor.fetchone()[0]

        if 'knowledge_fts' not in trigger_sql or 'INSERT' not in trigger_sql:
            logger.error("Verification failed: INSERT trigger has incorrect definition")
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
    # Run migration
    import sys

    logging.basicConfig(level=logging.INFO)

    db_path = sys.argv[1] if len(sys.argv) > 1 else "felix_knowledge.db"

    print(f"Running migration on {db_path}...")

    if migrate_up(db_path):
        print("Migration successful!")

        if verify_migration(db_path):
            print("Verification passed!")
            print("\nFTS5 triggers are now active. The full-text search index")
            print("will automatically stay in sync with knowledge_entries.")
        else:
            print("Verification failed!")
            sys.exit(1)
    else:
        print("Migration failed!")
        sys.exit(1)
