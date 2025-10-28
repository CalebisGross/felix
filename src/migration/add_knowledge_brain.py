"""
Knowledge Brain Database Migration

Adds tables and columns for autonomous document ingestion and knowledge retrieval:
- document_sources: Tracks ingested documents
- knowledge extensions: embedding, source_doc_id, chunk_index columns
- knowledge_fts: FTS5 virtual table for keyword search
- knowledge_usage: Meta-learning tracking

Backwards compatible with existing knowledge entries.
"""

import sqlite3
import logging
from .base_migration import Migration, MigrationError

logger = logging.getLogger(__name__)


class AddKnowledgeBrain(Migration):
    """Add Knowledge Brain tables and extensions."""

    version = 100  # High number to avoid conflicts with existing migrations
    description = "Add Knowledge Brain system for document ingestion and retrieval"

    def up(self, conn: sqlite3.Connection) -> None:
        """Apply the migration."""
        logger.info(f"Applying migration {self.version}: {self.description}")

        try:
            # 1. Create document_sources table
            logger.info("Creating document_sources table...")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_sources (
                    doc_id TEXT PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_hash TEXT NOT NULL,
                    page_count INTEGER,
                    title TEXT,
                    author TEXT,
                    created_date REAL,
                    modified_date REAL,
                    encoding TEXT DEFAULT 'utf-8',
                    ingestion_status TEXT DEFAULT 'pending',
                    ingestion_started REAL,
                    ingestion_completed REAL,
                    chunk_count INTEGER DEFAULT 0,
                    concept_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    added_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)

            # Create index on file_path for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_sources_path
                ON document_sources(file_path)
            """)

            # Create index on ingestion_status for filtering
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_sources_status
                ON document_sources(ingestion_status)
            """)

            # 2. Add columns to existing knowledge_entries table (if they don't exist)
            logger.info("Extending knowledge_entries table...")

            # Check if columns already exist
            cursor = conn.execute("PRAGMA table_info(knowledge_entries)")
            existing_columns = {row[1] for row in cursor.fetchall()}

            if 'embedding' not in existing_columns:
                # SQLite doesn't support ADD COLUMN for BLOB directly in some versions
                # Use ALTER TABLE ADD COLUMN
                conn.execute("ALTER TABLE knowledge_entries ADD COLUMN embedding BLOB")
                logger.info("Added 'embedding' column to knowledge_entries table")

            if 'source_doc_id' not in existing_columns:
                conn.execute("ALTER TABLE knowledge_entries ADD COLUMN source_doc_id TEXT")
                logger.info("Added 'source_doc_id' column to knowledge_entries table")

            if 'chunk_index' not in existing_columns:
                conn.execute("ALTER TABLE knowledge_entries ADD COLUMN chunk_index INTEGER")
                logger.info("Added 'chunk_index' column to knowledge_entries table")

            # Create index on source_doc_id for fast lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_entries_source_doc
                ON knowledge_entries(source_doc_id)
            """)

            # 3. Create FTS5 virtual table for full-text search
            logger.info("Creating knowledge_fts virtual table...")

            # Drop existing FTS table if it exists (for re-running migration)
            conn.execute("DROP TABLE IF EXISTS knowledge_fts")

            # Create FTS5 virtual table
            conn.execute("""
                CREATE VIRTUAL TABLE knowledge_fts USING fts5(
                    knowledge_id UNINDEXED,
                    content,
                    domain,
                    tags,
                    tokenize='porter unicode61'
                )
            """)

            # Populate FTS5 table with existing knowledge
            logger.info("Populating FTS5 table with existing knowledge...")
            # Use CASE to handle potential JSON errors gracefully
            conn.execute("""
                INSERT INTO knowledge_fts (knowledge_id, content, domain, tags)
                SELECT
                    knowledge_id,
                    CASE
                        WHEN json_valid(content_json) THEN
                            COALESCE(json_extract(content_json, '$.concept'), '') || ' ' ||
                            COALESCE(json_extract(content_json, '$.definition'), '') || ' ' ||
                            COALESCE(json_extract(content_json, '$.summary'), '')
                        ELSE content_json
                    END,
                    domain,
                    COALESCE(tags_json, '')
                FROM knowledge_entries
                WHERE content_json IS NOT NULL
            """)

            # 4. Create knowledge_usage table for meta-learning
            logger.info("Creating knowledge_usage table...")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_usage (
                    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    knowledge_id TEXT NOT NULL,
                    task_type TEXT,
                    task_complexity TEXT,
                    useful_score REAL DEFAULT 0.0,
                    retrieval_method TEXT,
                    recorded_at REAL DEFAULT (strftime('%s', 'now')),
                    FOREIGN KEY (knowledge_id) REFERENCES knowledge(knowledge_id) ON DELETE CASCADE
                )
            """)

            # Create indices for meta-learning queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_usage_workflow
                ON knowledge_usage(workflow_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_usage_knowledge
                ON knowledge_usage(knowledge_id)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_usage_task_type
                ON knowledge_usage(task_type)
            """)

            # 5. Create knowledge_usage_summary materialized view
            # (Periodically refreshed to track which knowledge is most useful)
            logger.info("Creating knowledge_usage_summary view...")
            conn.execute("DROP VIEW IF EXISTS knowledge_usage_summary")
            conn.execute("""
                CREATE VIEW knowledge_usage_summary AS
                SELECT
                    knowledge_id,
                    task_type,
                    COUNT(*) as usage_count,
                    AVG(useful_score) as avg_usefulness,
                    MAX(recorded_at) as last_used
                FROM knowledge_usage
                GROUP BY knowledge_id, task_type
                HAVING usage_count > 0
                ORDER BY avg_usefulness DESC, usage_count DESC
            """)

            conn.commit()
            logger.info(f"Migration {self.version} applied successfully")

        except sqlite3.Error as e:
            conn.rollback()
            raise MigrationError(f"Failed to apply migration {self.version}: {e}")

    def down(self, conn: sqlite3.Connection) -> None:
        """Rollback the migration."""
        logger.info(f"Rolling back migration {self.version}: {self.description}")

        try:
            # Drop created tables and views
            conn.execute("DROP VIEW IF EXISTS knowledge_usage_summary")
            conn.execute("DROP TABLE IF EXISTS knowledge_usage")
            conn.execute("DROP TABLE IF EXISTS knowledge_fts")
            conn.execute("DROP TABLE IF EXISTS document_sources")

            # Drop indices
            conn.execute("DROP INDEX IF EXISTS idx_knowledge_entries_source_doc")

            # Note: SQLite doesn't support DROP COLUMN directly
            # To remove columns from knowledge_entries table, would need to:
            # 1. Create new table without those columns
            # 2. Copy data
            # 3. Drop old table
            # 4. Rename new table
            # This is complex and risky, so we just leave the columns

            logger.warning(
                "Note: Columns (embedding, source_doc_id, chunk_index) "
                "were NOT removed from knowledge_entries table (SQLite limitation). "
                "They will remain but be unused."
            )

            conn.commit()
            logger.info(f"Migration {self.version} rolled back successfully")

        except sqlite3.Error as e:
            conn.rollback()
            raise MigrationError(f"Failed to rollback migration {self.version}: {e}")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify that the migration was applied successfully."""
        try:
            # Check document_sources table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='document_sources'
            """)
            if cursor.fetchone() is None:
                logger.error("document_sources table not found")
                return False

            # Check knowledge_entries table has new columns
            cursor = conn.execute("PRAGMA table_info(knowledge_entries)")
            columns = {row[1] for row in cursor.fetchall()}

            required_columns = {'embedding', 'source_doc_id', 'chunk_index'}
            if not required_columns.issubset(columns):
                missing = required_columns - columns
                logger.error(f"Missing columns in knowledge_entries table: {missing}")
                return False

            # Check knowledge_fts virtual table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_fts'
            """)
            if cursor.fetchone() is None:
                logger.error("knowledge_fts virtual table not found")
                return False

            # Check knowledge_usage table exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_usage'
            """)
            if cursor.fetchone() is None:
                logger.error("knowledge_usage table not found")
                return False

            # Check view exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='view' AND name='knowledge_usage_summary'
            """)
            if cursor.fetchone() is None:
                logger.error("knowledge_usage_summary view not found")
                return False

            logger.info(f"Migration {self.version} verified successfully")
            return True

        except sqlite3.Error as e:
            logger.error(f"Verification failed: {e}")
            return False
