"""
Migrations for felix_task_memory.db (TaskMemory).

These migrations:
- Add composite indexes for analytics queries
- Normalize JSON arrays into relational tables
- Add full-text search for task descriptions
"""

import sqlite3
import json
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class TaskMigration001(Migration):
    """Add composite indexes for analytics queries."""

    version = 1
    description = "Add composite indexes for performance analytics"

    def up(self, conn: sqlite3.Connection):
        """Create composite indexes."""
        logger.info("Creating composite indexes on task tables...")

        # Analytics: success rate by task type over time
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type_outcome_time
            ON task_executions(task_type, outcome, created_at DESC)
        """)

        # Performance analysis: duration by complexity
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_complexity_duration
            ON task_executions(complexity, duration)
        """)

        # Failure analysis: duration by outcome
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_outcome_duration
            ON task_executions(outcome, duration)
        """)

        logger.info("✓ Created 3 composite indexes")

    def down(self, conn: sqlite3.Connection):
        """Remove composite indexes."""
        conn.execute("DROP INDEX IF EXISTS idx_type_outcome_time")
        conn.execute("DROP INDEX IF EXISTS idx_complexity_duration")
        conn.execute("DROP INDEX IF EXISTS idx_outcome_duration")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify indexes were created."""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_type_outcome_time',
                'idx_complexity_duration',
                'idx_outcome_duration'
            )
        """)
        indexes = cursor.fetchall()
        return len(indexes) == 3


class TaskMigration002(Migration):
    """Normalize patterns_matched_json into relational table."""

    version = 2
    description = "Create task_pattern_matches table and migrate JSON data"

    def up(self, conn: sqlite3.Connection):
        """Create normalized pattern matches table."""
        logger.info("Creating task_pattern_matches table...")

        # Create normalized table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_pattern_matches (
                execution_id TEXT NOT NULL,
                pattern_id TEXT NOT NULL,
                match_confidence REAL DEFAULT 1.0,
                PRIMARY KEY (execution_id, pattern_id),
                FOREIGN KEY (execution_id) REFERENCES task_executions(execution_id) ON DELETE CASCADE,
                FOREIGN KEY (pattern_id) REFERENCES task_patterns(pattern_id) ON DELETE CASCADE
            )
        """)

        # Create indexes
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_executions
            ON task_pattern_matches(pattern_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_patterns
            ON task_pattern_matches(execution_id)
        """)

        logger.info("✓ Created task_pattern_matches table with 2 indexes")

        # Migrate existing data from JSON
        logger.info("Migrating patterns_matched_json data...")
        cursor = conn.execute("""
            SELECT execution_id, patterns_matched_json
            FROM task_executions
            WHERE patterns_matched_json != '[]' AND patterns_matched_json IS NOT NULL
        """)

        migrated = 0
        failed = 0

        for execution_id, patterns_json in cursor.fetchall():
            try:
                patterns = json.loads(patterns_json) if patterns_json else []
                for pattern_id in patterns:
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO task_pattern_matches (execution_id, pattern_id)
                            VALUES (?, ?)
                        """, (execution_id, pattern_id))
                        migrated += 1
                    except sqlite3.Error as e:
                        logger.warning(f"Could not migrate pattern {pattern_id} for {execution_id}: {e}")
                        failed += 1
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not parse patterns for execution {execution_id}: {e}")
                failed += 1

        logger.info(f"✓ Migrated {migrated} pattern matches ({failed} failed)")

    def down(self, conn: sqlite3.Connection):
        """Remove normalized table."""
        conn.execute("DROP TABLE IF EXISTS task_pattern_matches")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify table and indexes were created."""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='task_pattern_matches'
        """)
        return cursor.fetchone() is not None


class TaskMigration003(Migration):
    """Normalize agents_used_json into relational table."""

    version = 3
    description = "Create task_execution_agents table and migrate JSON data"

    def up(self, conn: sqlite3.Connection):
        """Create normalized agents table."""
        logger.info("Creating task_execution_agents table...")

        # Create normalized table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_execution_agents (
                execution_id TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                agent_count INTEGER DEFAULT 1,
                PRIMARY KEY (execution_id, agent_type),
                FOREIGN KEY (execution_id) REFERENCES task_executions(execution_id) ON DELETE CASCADE
            )
        """)

        # Create index for agent-based queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_type_executions
            ON task_execution_agents(agent_type)
        """)

        logger.info("✓ Created task_execution_agents table with index")

        # Migrate existing data from JSON
        logger.info("Migrating agents_used_json data...")
        cursor = conn.execute("""
            SELECT execution_id, agents_used_json
            FROM task_executions
            WHERE agents_used_json != '[]' AND agents_used_json IS NOT NULL
        """)

        migrated = 0
        failed = 0

        for execution_id, agents_json in cursor.fetchall():
            try:
                agents = json.loads(agents_json) if agents_json else []

                # Count occurrences of each agent type
                agent_counts = {}
                for agent_type in agents:
                    agent_counts[agent_type] = agent_counts.get(agent_type, 0) + 1

                # Insert with counts
                for agent_type, count in agent_counts.items():
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO task_execution_agents (execution_id, agent_type, agent_count)
                            VALUES (?, ?, ?)
                        """, (execution_id, agent_type, count))
                        migrated += 1
                    except sqlite3.Error as e:
                        logger.warning(f"Could not migrate agent {agent_type} for {execution_id}: {e}")
                        failed += 1
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Could not parse agents for execution {execution_id}: {e}")
                failed += 1

        logger.info(f"✓ Migrated {migrated} agent entries ({failed} failed)")

    def down(self, conn: sqlite3.Connection):
        """Remove normalized table."""
        conn.execute("DROP TABLE IF EXISTS task_execution_agents")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify table and index were created."""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='task_execution_agents'
        """)
        return cursor.fetchone() is not None


class TaskMigration004(Migration):
    """Add full-text search for task descriptions."""

    version = 4
    description = "Add FTS5 virtual table for task search"

    def up(self, conn: sqlite3.Connection):
        """Create FTS5 table and triggers."""
        logger.info("Creating FTS5 virtual table for task_executions...")

        # Create FTS5 virtual table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS task_executions_fts
            USING fts5(
                execution_id UNINDEXED,
                task_description,
                error_messages,
                tokenize='porter unicode61'
            )
        """)

        logger.info("✓ Created task_executions_fts virtual table")

        # Populate FTS table from existing data
        logger.info("Populating FTS table from existing task executions...")
        cursor = conn.execute("SELECT COUNT(*) FROM task_executions")
        total_entries = cursor.fetchone()[0]

        if total_entries > 0:
            conn.execute("""
                INSERT INTO task_executions_fts(execution_id, task_description, error_messages)
                SELECT execution_id, task_description, error_messages_json
                FROM task_executions
            """)
            logger.info(f"✓ Populated FTS table with {total_entries} entries")
        else:
            logger.info("No existing entries to populate")

        # Create triggers
        logger.info("Creating triggers to maintain FTS sync...")

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS task_executions_fts_insert
            AFTER INSERT ON task_executions BEGIN
                INSERT INTO task_executions_fts(execution_id, task_description, error_messages)
                VALUES (new.execution_id, new.task_description, new.error_messages_json);
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS task_executions_fts_update
            AFTER UPDATE ON task_executions BEGIN
                UPDATE task_executions_fts
                SET task_description = new.task_description,
                    error_messages = new.error_messages_json
                WHERE execution_id = new.execution_id;
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS task_executions_fts_delete
            AFTER DELETE ON task_executions BEGIN
                DELETE FROM task_executions_fts WHERE execution_id = old.execution_id;
            END
        """)

        logger.info("✓ Created 3 triggers (insert, update, delete)")

    def down(self, conn: sqlite3.Connection):
        """Remove FTS5 table and triggers."""
        conn.execute("DROP TRIGGER IF EXISTS task_executions_fts_insert")
        conn.execute("DROP TRIGGER IF EXISTS task_executions_fts_update")
        conn.execute("DROP TRIGGER IF EXISTS task_executions_fts_delete")
        conn.execute("DROP TABLE IF EXISTS task_executions_fts")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify FTS table and triggers were created."""
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='task_executions_fts'
        """)
        if cursor.fetchone() is None:
            return False

        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name LIKE 'task_executions_fts_%'
        """)
        triggers = cursor.fetchall()
        return len(triggers) == 3


def get_migrations() -> List[Migration]:
    """
    Get all task memory migrations in order.

    Returns:
        List of Migration instances
    """
    return [
        TaskMigration001(),
        TaskMigration002(),
        TaskMigration003(),
        TaskMigration004()
    ]
