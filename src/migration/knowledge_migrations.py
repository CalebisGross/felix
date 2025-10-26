"""
Migrations for felix_knowledge.db (KnowledgeStore).

These migrations add performance optimizations including:
- Composite indexes for common query patterns
- Full-text search (FTS5) for content searching
- Additional single-column indexes for agent attribution
"""

import sqlite3
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class KnowledgeMigration001(Migration):
    """Add composite indexes for common query patterns."""

    version = 1
    description = "Add composite indexes for efficient filtering and sorting"

    def up(self, conn: sqlite3.Connection):
        """Create composite indexes."""
        logger.info("Creating composite indexes on knowledge_entries...")

        # Composite index for filtered browsing: domain + confidence + time
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_domain_confidence_time
            ON knowledge_entries(domain, confidence_level, created_at DESC)
        """)

        # Composite index for agent-specific knowledge: source_agent + domain
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_agent_domain
            ON knowledge_entries(source_agent, domain)
        """)

        # Composite index for quality-based queries: confidence + success_rate
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence_success
            ON knowledge_entries(confidence_level, success_rate DESC)
        """)

        # Single-column index for agent attribution (was missing)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_agent
            ON knowledge_entries(source_agent)
        """)

        logger.info("✓ Created 4 composite/additional indexes")

    def down(self, conn: sqlite3.Connection):
        """Remove composite indexes."""
        conn.execute("DROP INDEX IF EXISTS idx_domain_confidence_time")
        conn.execute("DROP INDEX IF EXISTS idx_source_agent_domain")
        conn.execute("DROP INDEX IF EXISTS idx_confidence_success")
        conn.execute("DROP INDEX IF EXISTS idx_source_agent")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify indexes were created."""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_domain_confidence_time',
                'idx_source_agent_domain',
                'idx_confidence_success',
                'idx_source_agent'
            )
        """)
        indexes = cursor.fetchall()
        return len(indexes) == 4


class KnowledgeMigration002(Migration):
    """Add full-text search (FTS5) for content searching."""

    version = 2
    description = "Add FTS5 virtual table for fast content search"

    def up(self, conn: sqlite3.Connection):
        """Create FTS5 table and triggers."""
        logger.info("Creating FTS5 virtual table for knowledge_entries...")

        # Create FTS5 virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
            USING fts5(
                knowledge_id UNINDEXED,
                content_text,
                domain,
                tags,
                tokenize='porter unicode61'
            )
        """)

        logger.info("✓ Created knowledge_fts virtual table")

        # Populate FTS table from existing data
        logger.info("Populating FTS table from existing knowledge entries...")
        cursor = conn.execute("SELECT COUNT(*) FROM knowledge_entries")
        total_entries = cursor.fetchone()[0]

        if total_entries > 0:
            conn.execute("""
                INSERT INTO knowledge_fts(knowledge_id, content_text, domain, tags)
                SELECT knowledge_id, content_json, domain, tags_json
                FROM knowledge_entries
            """)
            logger.info(f"✓ Populated FTS table with {total_entries} entries")
        else:
            logger.info("No existing entries to populate")

        # Create triggers to keep FTS in sync
        logger.info("Creating triggers to maintain FTS sync...")

        # Trigger for INSERT
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS knowledge_fts_insert
            AFTER INSERT ON knowledge_entries BEGIN
                INSERT INTO knowledge_fts(knowledge_id, content_text, domain, tags)
                VALUES (new.knowledge_id, new.content_json, new.domain, new.tags_json);
            END
        """)

        # Trigger for UPDATE
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS knowledge_fts_update
            AFTER UPDATE ON knowledge_entries BEGIN
                UPDATE knowledge_fts
                SET content_text = new.content_json,
                    domain = new.domain,
                    tags = new.tags_json
                WHERE knowledge_id = new.knowledge_id;
            END
        """)

        # Trigger for DELETE
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS knowledge_fts_delete
            AFTER DELETE ON knowledge_entries BEGIN
                DELETE FROM knowledge_fts WHERE knowledge_id = old.knowledge_id;
            END
        """)

        logger.info("✓ Created 3 triggers (insert, update, delete)")

    def down(self, conn: sqlite3.Connection):
        """Remove FTS5 table and triggers."""
        conn.execute("DROP TRIGGER IF EXISTS knowledge_fts_insert")
        conn.execute("DROP TRIGGER IF EXISTS knowledge_fts_update")
        conn.execute("DROP TRIGGER IF EXISTS knowledge_fts_delete")
        conn.execute("DROP TABLE IF EXISTS knowledge_fts")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify FTS table and triggers were created."""
        # Check FTS table exists
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='knowledge_fts'
        """)
        if cursor.fetchone() is None:
            return False

        # Check triggers exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name IN (
                'knowledge_fts_insert',
                'knowledge_fts_update',
                'knowledge_fts_delete'
            )
        """)
        triggers = cursor.fetchall()
        return len(triggers) == 3


def get_migrations() -> List[Migration]:
    """
    Get all knowledge store migrations in order.

    Returns:
        List of Migration instances
    """
    return [
        KnowledgeMigration001(),
        KnowledgeMigration002()
    ]
