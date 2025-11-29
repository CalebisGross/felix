"""
Knowledge Relationships Table Migration

Adds the knowledge_relationships table for storing relationships
between knowledge entries (explicit mentions, similarity, co-occurrence).

This table is used by:
- KnowledgeGraphBuilder for relationship discovery
- GUI panels for relationship visualization
- Backup/restore for data portability
"""

import sqlite3
import logging
from .base_migration import Migration, MigrationError

logger = logging.getLogger(__name__)


class AddKnowledgeRelationshipsTable(Migration):
    """Add knowledge_relationships table."""

    version = 105  # After knowledge brain migration (100)
    description = "Add knowledge_relationships table for knowledge graph"

    def up(self, conn: sqlite3.Connection) -> None:
        """Apply the migration."""
        logger.info(f"Applying migration {self.version}: {self.description}")

        try:
            # Create knowledge_relationships table
            logger.info("Creating knowledge_relationships table...")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    created_at REAL DEFAULT (strftime('%s', 'now')),
                    UNIQUE(source_id, target_id, relationship_type)
                )
            """)

            # Create indexes for efficient queries
            logger.info("Creating indexes on knowledge_relationships...")
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
            logger.info("Migration completed successfully")

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise MigrationError(f"Failed to create knowledge_relationships table: {e}")

    def down(self, conn: sqlite3.Connection) -> None:
        """Rollback the migration."""
        logger.info(f"Rolling back migration {self.version}")

        try:
            conn.execute("DROP TABLE IF EXISTS knowledge_relationships")
            conn.commit()
            logger.info("Rollback completed successfully")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise MigrationError(f"Failed to rollback: {e}")


def run_migration(db_path: str = "felix_knowledge.db") -> bool:
    """
    Run this migration directly.

    Args:
        db_path: Path to the knowledge database

    Returns:
        True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        migration = AddKnowledgeRelationshipsTable()
        migration.up(conn)
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    db_path = sys.argv[1] if len(sys.argv) > 1 else "felix_knowledge.db"
    print(f"Running migration on {db_path}...")

    if run_migration(db_path):
        print("Migration completed successfully!")
    else:
        print("Migration failed!")
        sys.exit(1)
