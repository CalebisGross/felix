"""
Migration: Add CASCADE DELETE to knowledge_relationships

Adds CASCADE DELETE foreign key constraints to the knowledge_relationships table.
This ensures that when a knowledge entry is deleted, all relationships involving
that entry are automatically deleted, preventing orphaned relationships.

Background:
- Currently knowledge_relationships has NO foreign key constraints
- Orphaned relationships accumulate when entries are deleted
- Manual cleanup has been required via graph builder

Challenge:
- SQLite doesn't support ALTER TABLE for foreign keys
- Must recreate table with new constraints
- Must clean orphaned rows first

Process:
1. Delete orphaned relationships (where source_id or target_id don't exist)
2. Create new table with CASCADE DELETE constraints
3. Copy all valid data from old table
4. Drop old table
5. Rename new table
6. Recreate indexes
"""

import sqlite3
import logging
import time

logger = logging.getLogger(__name__)


def migrate_up(db_path: str = "felix_knowledge.db") -> bool:
    """
    Add CASCADE DELETE foreign keys to knowledge_relationships.

    Args:
        db_path: Path to knowledge database

    Returns:
        True if migration successful, False otherwise
    """
    logger.info("Running migration: add_relationships_cascade")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = OFF")  # Disable during migration
        conn.execute("BEGIN TRANSACTION")

        # Check if table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='knowledge_relationships'
        """)
        if not cursor.fetchone():
            logger.info("knowledge_relationships table doesn't exist, nothing to migrate")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.close()
            return True

        # Check if migration already applied
        cursor = conn.execute("PRAGMA foreign_key_list(knowledge_relationships)")
        fk_list = cursor.fetchall()

        for fk in fk_list:
            # fk format: (id, seq, table, from, to, on_update, on_delete, match)
            if fk[2] == 'knowledge_entries' and fk[6] == 'CASCADE':
                logger.info("CASCADE DELETE constraint already exists, skipping migration")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.close()
                return True

        # Count initial relationships
        cursor = conn.execute("SELECT COUNT(*) FROM knowledge_relationships")
        initial_count = cursor.fetchone()[0]
        logger.info(f"Found {initial_count} relationships before cleanup")

        # Step 1: Clean orphaned relationships FIRST
        logger.info("Cleaning orphaned relationships...")

        # Delete relationships where source_id doesn't exist
        cursor = conn.execute("""
            DELETE FROM knowledge_relationships
            WHERE source_id NOT IN (SELECT knowledge_id FROM knowledge_entries)
        """)
        orphaned_source = cursor.rowcount
        logger.info(f"  Deleted {orphaned_source} relationships with missing source_id")

        # Delete relationships where target_id doesn't exist
        cursor = conn.execute("""
            DELETE FROM knowledge_relationships
            WHERE target_id NOT IN (SELECT knowledge_id FROM knowledge_entries)
        """)
        orphaned_target = cursor.rowcount
        logger.info(f"  Deleted {orphaned_target} relationships with missing target_id")

        # Count remaining
        cursor = conn.execute("SELECT COUNT(*) FROM knowledge_relationships")
        clean_count = cursor.fetchone()[0]
        total_deleted = initial_count - clean_count
        logger.info(f"Cleanup complete: removed {total_deleted} orphaned relationships, {clean_count} remain")

        # Step 2: Create new table with CASCADE DELETE constraints
        logger.info("Creating new knowledge_relationships table with CASCADE DELETE...")

        conn.execute("""
            CREATE TABLE knowledge_relationships_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(source_id, target_id, relationship_type),
                FOREIGN KEY(source_id) REFERENCES knowledge_entries(knowledge_id) ON DELETE CASCADE,
                FOREIGN KEY(target_id) REFERENCES knowledge_entries(knowledge_id) ON DELETE CASCADE
            )
        """)

        # Step 3: Copy all data from old table
        logger.info("Copying data from old table...")
        conn.execute("""
            INSERT INTO knowledge_relationships_new
            (id, source_id, target_id, relationship_type, confidence, created_at)
            SELECT id, source_id, target_id, relationship_type, confidence, created_at
            FROM knowledge_relationships
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM knowledge_relationships_new").fetchone()[0]
        logger.info(f"Copied {row_count} relationships")

        # Step 4: Drop old table
        logger.info("Dropping old table...")
        conn.execute("DROP TABLE knowledge_relationships")

        # Step 5: Rename new table
        logger.info("Renaming new table...")
        conn.execute("ALTER TABLE knowledge_relationships_new RENAME TO knowledge_relationships")

        # Step 6: Recreate indexes
        logger.info("Recreating indexes...")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kr_source
            ON knowledge_relationships(source_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kr_target
            ON knowledge_relationships(target_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_kr_type
            ON knowledge_relationships(relationship_type)
        """)

        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")  # Re-enable
        conn.close()

        logger.info("Successfully added CASCADE DELETE foreign key constraints to knowledge_relationships")
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if conn:
            conn.rollback()
            conn.execute("PRAGMA foreign_keys = ON")
            conn.close()
        return False


def migrate_down(db_path: str = "felix_knowledge.db") -> bool:
    """
    Remove CASCADE DELETE foreign keys (recreate table without constraints).

    Args:
        db_path: Path to knowledge database

    Returns:
        True if rollback successful, False otherwise
    """
    logger.info("Rolling back migration: add_relationships_cascade")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN TRANSACTION")

        # Check if table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='knowledge_relationships'
        """)
        if not cursor.fetchone():
            logger.info("knowledge_relationships table doesn't exist, nothing to rollback")
            conn.execute("PRAGMA foreign_keys = ON")
            conn.close()
            return True

        logger.info("Creating knowledge_relationships table without CASCADE DELETE...")

        # Create table without CASCADE DELETE
        conn.execute("""
            CREATE TABLE knowledge_relationships_rollback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                UNIQUE(source_id, target_id, relationship_type)
            )
        """)

        # Copy data
        conn.execute("""
            INSERT INTO knowledge_relationships_rollback
            (id, source_id, target_id, relationship_type, confidence, created_at)
            SELECT id, source_id, target_id, relationship_type, confidence, created_at
            FROM knowledge_relationships
        """)

        # Drop and rename
        conn.execute("DROP TABLE knowledge_relationships")
        conn.execute("ALTER TABLE knowledge_relationships_rollback RENAME TO knowledge_relationships")

        # Recreate indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_source ON knowledge_relationships(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_target ON knowledge_relationships(target_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kr_type ON knowledge_relationships(relationship_type)")

        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()

        logger.info("Successfully removed CASCADE DELETE constraints")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        if conn:
            conn.rollback()
            conn.execute("PRAGMA foreign_keys = ON")
            conn.close()
        return False


def verify_migration(db_path: str = "felix_knowledge.db") -> bool:
    """
    Verify that the migration was successful.

    Checks:
    1. knowledge_relationships table exists
    2. CASCADE DELETE foreign keys exist for both source_id and target_id
    3. All indexes recreated
    4. No orphaned relationships exist

    Args:
        db_path: Path to knowledge database

    Returns:
        True if verification passed, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='knowledge_relationships'
        """)

        if not cursor.fetchone():
            logger.error("Verification failed: knowledge_relationships table not found")
            conn.close()
            return False

        # Check CASCADE DELETE foreign keys
        cursor.execute("PRAGMA foreign_key_list(knowledge_relationships)")
        fk_list = cursor.fetchall()

        source_cascade = False
        target_cascade = False

        for fk in fk_list:
            # fk format: (id, seq, table, from, to, on_update, on_delete, match)
            if fk[2] == 'knowledge_entries' and fk[6] == 'CASCADE':
                if fk[3] == 'source_id':
                    source_cascade = True
                elif fk[3] == 'target_id':
                    target_cascade = True

        if not source_cascade:
            logger.error("Verification failed: CASCADE DELETE constraint on source_id not found")
            conn.close()
            return False

        if not target_cascade:
            logger.error("Verification failed: CASCADE DELETE constraint on target_id not found")
            conn.close()
            return False

        # Check indexes
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='knowledge_relationships'
        """)
        indexes = {row[0] for row in cursor.fetchall()}

        expected_indexes = {'idx_kr_source', 'idx_kr_target', 'idx_kr_type'}

        if not expected_indexes.issubset(indexes):
            missing = expected_indexes - indexes
            logger.error(f"Verification failed: missing indexes {missing}")
            conn.close()
            return False

        # Check for orphaned relationships
        cursor.execute("""
            SELECT COUNT(*) FROM knowledge_relationships
            WHERE source_id NOT IN (SELECT knowledge_id FROM knowledge_entries)
               OR target_id NOT IN (SELECT knowledge_id FROM knowledge_entries)
        """)
        orphan_count = cursor.fetchone()[0]

        if orphan_count > 0:
            logger.warning(f"Found {orphan_count} orphaned relationships (FK constraints should prevent new ones)")

        # Get stats
        cursor.execute("SELECT COUNT(*) FROM knowledge_relationships")
        rel_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
        entry_count = cursor.fetchone()[0]

        conn.close()
        logger.info("Migration verification passed")
        logger.info(f"  ✓ CASCADE DELETE constraint on source_id active")
        logger.info(f"  ✓ CASCADE DELETE constraint on target_id active")
        logger.info(f"  ✓ All indexes present")
        logger.info(f"  ✓ {rel_count} relationships linking {entry_count} entries")
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
    print("\n⚠️  WARNING: This migration will recreate the knowledge_relationships table.")
    print("   Orphaned relationships will be deleted.")
    print("   Ensure you have a backup before proceeding!")
    print()

    if migrate_up(db_path):
        print("✓ Migration successful!")

        if verify_migration(db_path):
            print("✓ Verification passed!")
            print("\nCASCADE DELETE is now active. When you delete a knowledge entry,")
            print("all relationships involving that entry will be automatically deleted.")
        else:
            print("✗ Verification failed!")
            sys.exit(1)
    else:
        print("✗ Migration failed!")
        sys.exit(1)
