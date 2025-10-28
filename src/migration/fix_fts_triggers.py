"""
Migration: Fix FTS5 Trigger Conflicts

Removes old database triggers that reference 'content_text' column which no longer
exists in the knowledge_fts table (now uses 'content' column).

This fixes the error: "no such column: content_text" that occurs when knowledge
entries are accessed in the GUI.

Old triggers to remove:
- knowledge_fts_insert
- knowledge_fts_update
- knowledge_fts_delete

These triggers were created by an earlier migration but conflict with the new
Knowledge Brain schema.
"""

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_trigger_info(conn, trigger_name: str) -> bool:
    """Check if a trigger exists."""
    cursor = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type='trigger' AND name=?
    """, (trigger_name,))
    return cursor.fetchone() is not None


def apply_migration(db_path: str = "felix_knowledge.db"):
    """
    Apply migration to fix FTS5 trigger conflicts.

    Args:
        db_path: Path to the knowledge database
    """
    logger.info("=" * 80)
    logger.info("FTS5 TRIGGER FIX MIGRATION")
    logger.info("=" * 80)

    db_file = Path(db_path)
    if not db_file.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Please ensure Felix has been run at least once to create the database.")
        return False

    logger.info(f"✓ Using existing database: {db_path}")
    logger.info("")

    try:
        with sqlite3.connect(db_path) as conn:
            # Check which old triggers exist
            old_triggers = [
                'knowledge_fts_insert',
                'knowledge_fts_update',
                'knowledge_fts_delete'
            ]

            existing_triggers = []
            for trigger_name in old_triggers:
                if get_trigger_info(conn, trigger_name):
                    existing_triggers.append(trigger_name)

            if not existing_triggers:
                logger.info("✓ No old triggers found - database already clean")
                logger.info("")
                return True

            logger.info(f"Found {len(existing_triggers)} old trigger(s) to remove:")
            for trigger in existing_triggers:
                logger.info(f"  - {trigger}")
            logger.info("")

            # Drop old triggers
            logger.info("Dropping old triggers...")
            for trigger_name in existing_triggers:
                try:
                    conn.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
                    logger.info(f"  ✓ Dropped {trigger_name}")
                except sqlite3.Error as e:
                    logger.error(f"  ✗ Failed to drop {trigger_name}: {e}")
                    return False

            conn.commit()
            logger.info("")

            # Verify FTS5 table schema
            logger.info("Verifying knowledge_fts table schema...")
            try:
                cursor = conn.execute("PRAGMA table_info(knowledge_fts)")
                columns = cursor.fetchall()

                if columns:
                    logger.info("  FTS5 table columns:")
                    for col in columns:
                        logger.info(f"    - {col[1]} ({col[2]})")

                    # Check if 'content' column exists (not 'content_text')
                    col_names = [col[1] for col in columns]
                    if 'content' in col_names:
                        logger.info("  ✓ FTS5 table uses 'content' column (correct)")
                    elif 'content_text' in col_names:
                        logger.warning("  ⚠ FTS5 table still uses 'content_text' column!")
                        logger.warning("    You may need to run the Knowledge Brain migration.")
                else:
                    logger.warning("  ⚠ knowledge_fts table not found")
                    logger.warning("    This is OK if you haven't run the Knowledge Brain migration yet.")
            except sqlite3.Error as e:
                logger.debug(f"Could not verify FTS5 schema: {e}")

            logger.info("")
            logger.info("=" * 80)
            logger.info("✓ MIGRATION COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info("")
            logger.info("The GUI Dashboard Memory tab should now load without errors.")
            logger.info("")

            return True

    except sqlite3.Error as e:
        logger.error(f"✗ Database error: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Apply migration
    success = apply_migration()

    if not success:
        exit(1)
