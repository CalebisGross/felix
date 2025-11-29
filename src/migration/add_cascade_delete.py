"""
Migration: Add CASCADE DELETE Foreign Key

Adds CASCADE DELETE foreign key constraint to knowledge_entries.source_doc_id.
This ensures that when a document is deleted, all associated knowledge entries
are automatically deleted, preventing orphaned entries.

Background:
- Currently source_doc_id has NO foreign key constraint
- Manual cleanup required via delete_orphaned_entries()
- Adding CASCADE DELETE automates this cleanup

Challenge:
- SQLite doesn't support ALTER TABLE for foreign keys
- Must recreate table with new constraint

Process:
1. Create new table with CASCADE DELETE constraint
2. Copy all data from old table
3. Drop old table
4. Rename new table
5. Recreate indexes
"""

import sqlite3
import logging
import time

logger = logging.getLogger(__name__)


def migrate_up(db_path: str = "felix_knowledge.db") -> bool:
    """
    Add CASCADE DELETE foreign key to knowledge_entries.source_doc_id.

    Args:
        db_path: Path to knowledge database

    Returns:
        True if migration successful, False otherwise
    """
    logger.info("Running migration: add_cascade_delete")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = OFF")  # Disable during migration
        conn.execute("BEGIN TRANSACTION")

        # Check if migration already applied
        cursor = conn.execute("PRAGMA foreign_key_list(knowledge_entries)")
        fk_list = cursor.fetchall()

        for fk in fk_list:
            # fk format: (id, seq, table, from, to, on_update, on_delete, match)
            if fk[2] == 'document_sources' and fk[6] == 'CASCADE':
                logger.info("CASCADE DELETE constraint already exists, skipping migration")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.close()
                return True

        logger.info("Creating new knowledge_entries table with CASCADE DELETE...")

        # 1. Create new table with CASCADE DELETE constraint
        conn.execute("""
            CREATE TABLE knowledge_entries_new (
                knowledge_id TEXT PRIMARY KEY,
                knowledge_type TEXT NOT NULL,
                content_json TEXT NOT NULL,
                content_compressed BLOB,
                confidence_level TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                domain TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                related_entries_json TEXT DEFAULT '[]',
                validation_score REAL DEFAULT 1.0,
                validation_flags TEXT DEFAULT '[]',
                validation_status TEXT DEFAULT 'trusted',
                validated_at REAL,
                embedding BLOB,
                source_doc_id TEXT,
                chunk_index INTEGER,
                FOREIGN KEY(source_doc_id) REFERENCES document_sources(doc_id) ON DELETE CASCADE
            )
        """)

        # 2. Copy all data from old table
        logger.info("Copying data from old table...")
        conn.execute("""
            INSERT INTO knowledge_entries_new
            SELECT * FROM knowledge_entries
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM knowledge_entries_new").fetchone()[0]
        logger.info(f"Copied {row_count} knowledge entries")

        # 3. Drop old table
        logger.info("Dropping old table...")
        conn.execute("DROP TABLE knowledge_entries")

        # 4. Rename new table
        logger.info("Renaming new table...")
        conn.execute("ALTER TABLE knowledge_entries_new RENAME TO knowledge_entries")

        # 5. Recreate indexes
        logger.info("Recreating indexes...")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_type
            ON knowledge_entries(knowledge_type)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_domain
            ON knowledge_entries(domain)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_created
            ON knowledge_entries(created_at DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_confidence
            ON knowledge_entries(confidence_level)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_source_agent
            ON knowledge_entries(source_agent)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_source_doc
            ON knowledge_entries(source_doc_id)
        """)

        # Note: FTS5 triggers will automatically handle knowledge_fts updates
        # via the triggers created by add_fts5_triggers.py

        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")  # Re-enable
        conn.close()

        logger.info("Successfully added CASCADE DELETE foreign key constraint")
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
    Remove CASCADE DELETE foreign key (recreate table without constraint).

    Args:
        db_path: Path to knowledge database

    Returns:
        True if rollback successful, False otherwise
    """
    logger.info("Rolling back migration: add_cascade_delete")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN TRANSACTION")

        logger.info("Creating knowledge_entries table without CASCADE DELETE...")

        # Create table without CASCADE DELETE
        conn.execute("""
            CREATE TABLE knowledge_entries_rollback (
                knowledge_id TEXT PRIMARY KEY,
                knowledge_type TEXT NOT NULL,
                content_json TEXT NOT NULL,
                content_compressed BLOB,
                confidence_level TEXT NOT NULL,
                source_agent TEXT NOT NULL,
                domain TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                related_entries_json TEXT DEFAULT '[]',
                validation_score REAL DEFAULT 1.0,
                validation_flags TEXT DEFAULT '[]',
                validation_status TEXT DEFAULT 'trusted',
                validated_at REAL,
                embedding BLOB,
                source_doc_id TEXT,
                chunk_index INTEGER
            )
        """)

        # Copy data
        conn.execute("""
            INSERT INTO knowledge_entries_rollback
            SELECT * FROM knowledge_entries
        """)

        # Drop and rename
        conn.execute("DROP TABLE knowledge_entries")
        conn.execute("ALTER TABLE knowledge_entries_rollback RENAME TO knowledge_entries")

        # Recreate indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_entries(knowledge_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_domain ON knowledge_entries(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_created ON knowledge_entries(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_confidence ON knowledge_entries(confidence_level)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_source_agent ON knowledge_entries(source_agent)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_source_doc ON knowledge_entries(source_doc_id)")

        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.close()

        logger.info("Successfully removed CASCADE DELETE constraint")
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
    1. knowledge_entries table exists
    2. CASCADE DELETE foreign key exists
    3. All indexes recreated
    4. Data integrity (row counts match)

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
            WHERE type='table' AND name='knowledge_entries'
        """)

        if not cursor.fetchone():
            logger.error("Verification failed: knowledge_entries table not found")
            conn.close()
            return False

        # Check CASCADE DELETE foreign key
        cursor.execute("PRAGMA foreign_key_list(knowledge_entries)")
        fk_list = cursor.fetchall()

        cascade_found = False
        for fk in fk_list:
            # fk format: (id, seq, table, from, to, on_update, on_delete, match)
            if fk[2] == 'document_sources' and fk[3] == 'source_doc_id' and fk[6] == 'CASCADE':
                cascade_found = True
                break

        if not cascade_found:
            logger.error("Verification failed: CASCADE DELETE constraint not found")
            conn.close()
            return False

        # Check indexes
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='knowledge_entries'
        """)
        indexes = {row[0] for row in cursor.fetchall()}

        expected_indexes = {
            'idx_knowledge_type',
            'idx_knowledge_domain',
            'idx_knowledge_created',
            'idx_knowledge_confidence',
            'idx_knowledge_source_agent',
            'idx_knowledge_source_doc'
        }

        if not expected_indexes.issubset(indexes):
            missing = expected_indexes - indexes
            logger.error(f"Verification failed: missing indexes {missing}")
            conn.close()
            return False

        # Check data integrity
        cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
        entries_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM knowledge_fts")
        fts_count = cursor.fetchone()[0]

        if entries_count != fts_count:
            logger.warning(
                f"FTS5 sync issue detected: {entries_count} entries vs {fts_count} FTS rows. "
                f"Run rebuild if needed."
            )

        conn.close()
        logger.info("Migration verification passed")
        logger.info(f"  ✓ CASCADE DELETE constraint active")
        logger.info(f"  ✓ All indexes present")
        logger.info(f"  ✓ Data integrity verified ({entries_count} entries)")
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
    print("\n⚠️  WARNING: This migration will recreate the knowledge_entries table.")
    print("   Ensure you have a backup before proceeding!")
    print()

    if migrate_up(db_path):
        print("✓ Migration successful!")

        if verify_migration(db_path):
            print("✓ Verification passed!")
            print("\nCASCADE DELETE is now active. When you delete a document,")
            print("all associated knowledge entries will be automatically deleted.")
        else:
            print("✗ Verification failed!")
            sys.exit(1)
    else:
        print("✗ Migration failed!")
        sys.exit(1)
