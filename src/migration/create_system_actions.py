"""
Create felix_system_actions.db (SystemActionsStore).

This is a new database for tracking system commands, patterns,
and approvals for the system autonomy feature.
"""

import sqlite3
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class SystemActionsMigration001(Migration):
    """Create system actions database with all tables and indexes."""

    version = 1
    description = "Create command executions, patterns, and approval tables"

    def up(self, conn: sqlite3.Connection):
        """Create system actions schema."""
        logger.info("Creating system actions database schema...")

        # Main command executions table
        logger.info("Creating command_executions table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS command_executions (
                execution_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id INTEGER,
                agent_id TEXT NOT NULL,
                agent_type TEXT,
                command TEXT NOT NULL,
                command_hash TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                approved_by TEXT,
                approval_timestamp REAL,
                executed BOOLEAN NOT NULL DEFAULT 0,
                execution_timestamp REAL,
                exit_code INTEGER,
                duration REAL,
                stdout_preview TEXT,
                stderr_preview TEXT,
                output_size INTEGER,
                context TEXT,
                cwd TEXT,
                env_snapshot TEXT,
                venv_active BOOLEAN,
                success BOOLEAN,
                error_category TEXT,
                timestamp REAL NOT NULL
            )
        """)

        # Create indexes for command_executions
        logger.info("Creating indexes for command_executions...")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_commands
            ON command_executions(agent_id, timestamp DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_command_hash
            ON command_executions(command_hash)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trust_level
            ON command_executions(trust_level, executed)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_success_rate
            ON command_executions(success, trust_level)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_commands
            ON command_executions(workflow_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_execution_time
            ON command_executions(execution_timestamp DESC)
        """)

        logger.info("✓ Created command_executions with 6 indexes")

        # FTS for command search
        logger.info("Creating FTS5 for command search...")
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS command_fts
            USING fts5(
                execution_id UNINDEXED,
                command,
                context,
                stdout_preview,
                stderr_preview,
                tokenize='porter unicode61'
            )
        """)

        # Create triggers to keep FTS in sync
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS command_fts_insert
            AFTER INSERT ON command_executions BEGIN
                INSERT INTO command_fts(execution_id, command, context, stdout_preview, stderr_preview)
                VALUES (new.execution_id, new.command, COALESCE(new.context, ''),
                        COALESCE(new.stdout_preview, ''), COALESCE(new.stderr_preview, ''));
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS command_fts_update
            AFTER UPDATE ON command_executions BEGIN
                UPDATE command_fts
                SET command = new.command,
                    context = COALESCE(new.context, ''),
                    stdout_preview = COALESCE(new.stdout_preview, ''),
                    stderr_preview = COALESCE(new.stderr_preview, '')
                WHERE execution_id = new.execution_id;
            END
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS command_fts_delete
            AFTER DELETE ON command_executions BEGIN
                DELETE FROM command_fts WHERE execution_id = old.execution_id;
            END
        """)

        logger.info("✓ Created command_fts with 3 triggers")

        # Command patterns table (learned sequences)
        logger.info("Creating command_patterns table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS command_patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT UNIQUE NOT NULL,
                command_sequence TEXT NOT NULL,
                task_category TEXT,
                success_rate REAL DEFAULT 0.0,
                usage_count INTEGER DEFAULT 0,
                avg_duration REAL DEFAULT 0.0,
                preconditions TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_category
            ON command_patterns(task_category, success_rate DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_usage
            ON command_patterns(usage_count DESC, success_rate DESC)
        """)

        logger.info("✓ Created command_patterns with 2 indexes")

        # Pattern usage tracking
        logger.info("Creating command_pattern_usage table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS command_pattern_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id INTEGER NOT NULL,
                workflow_id INTEGER,
                agent_id TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                duration REAL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (pattern_id) REFERENCES command_patterns(pattern_id) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_success
            ON command_pattern_usage(pattern_id, success)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_timestamp
            ON command_pattern_usage(timestamp DESC)
        """)

        logger.info("✓ Created command_pattern_usage with 2 indexes")

        # Approval queue table
        logger.info("Creating pending_approvals table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pending_approvals (
                approval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_id INTEGER NOT NULL,
                agent_id TEXT NOT NULL,
                command TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                context TEXT,
                risk_assessment TEXT,
                requested_at REAL NOT NULL,
                expires_at REAL,
                approved BOOLEAN,
                approved_by TEXT,
                approved_at REAL,
                denial_reason TEXT,
                FOREIGN KEY (execution_id) REFERENCES command_executions(execution_id) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pending_status
            ON pending_approvals(approved, requested_at DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_approval_expiry
            ON pending_approvals(expires_at)
        """)

        logger.info("✓ Created pending_approvals with 2 indexes")

        # Summary
        logger.info("\n" + "="*60)
        logger.info("SYSTEM ACTIONS DATABASE CREATED")
        logger.info("="*60)
        logger.info("Tables created:")
        logger.info("  - command_executions (with FTS)")
        logger.info("  - command_patterns")
        logger.info("  - command_pattern_usage")
        logger.info("  - pending_approvals")
        logger.info("Total indexes: 14")
        logger.info("FTS triggers: 3")
        logger.info("="*60)

    def down(self, conn: sqlite3.Connection):
        """Remove all system actions tables."""
        conn.execute("DROP TRIGGER IF EXISTS command_fts_insert")
        conn.execute("DROP TRIGGER IF EXISTS command_fts_update")
        conn.execute("DROP TRIGGER IF EXISTS command_fts_delete")
        conn.execute("DROP TABLE IF EXISTS command_fts")
        conn.execute("DROP TABLE IF EXISTS pending_approvals")
        conn.execute("DROP TABLE IF EXISTS command_pattern_usage")
        conn.execute("DROP TABLE IF EXISTS command_patterns")
        conn.execute("DROP TABLE IF EXISTS command_executions")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify all tables and indexes were created."""
        # Check tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN (
                'command_executions',
                'command_fts',
                'command_patterns',
                'command_pattern_usage',
                'pending_approvals'
            )
        """)
        tables = cursor.fetchall()
        if len(tables) != 5:
            logger.error(f"Expected 5 tables, found {len(tables)}")
            return False

        # Check key indexes exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_agent_commands',
                'idx_command_hash',
                'idx_pattern_category',
                'idx_pending_status'
            )
        """)
        indexes = cursor.fetchall()
        if len(indexes) != 4:
            logger.error(f"Expected at least 4 key indexes, found {len(indexes)}")
            return False

        # Check triggers exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name LIKE 'command_fts_%'
        """)
        triggers = cursor.fetchall()
        if len(triggers) != 3:
            logger.error(f"Expected 3 FTS triggers, found {len(triggers)}")
            return False

        return True


class SystemActionsMigration002(Migration):
    """Add status field for tracking active commands."""

    version = 2
    description = "Add status field to command_executions for Terminal Output Tab"

    def up(self, conn: sqlite3.Connection):
        """Add status field and indexes."""
        logger.info("Adding status field to command_executions...")

        # Add status column
        conn.execute("""
            ALTER TABLE command_executions
            ADD COLUMN status TEXT DEFAULT 'completed'
        """)

        # Add index for fast active command queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_command_status
            ON command_executions(status, timestamp DESC)
        """)

        # Add composite index for workflow filtering
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_timestamp
            ON command_executions(workflow_id, timestamp DESC)
        """)

        # Backfill existing records based on success field
        conn.execute("""
            UPDATE command_executions
            SET status = CASE
                WHEN success = 1 THEN 'completed'
                WHEN success = 0 THEN 'failed'
                ELSE 'completed'
            END
            WHERE status = 'completed'
        """)

        logger.info("✓ Added status field with 2 indexes")
        logger.info("✓ Backfilled existing records")

    def down(self, conn: sqlite3.Connection):
        """Remove status field (SQLite doesn't support DROP COLUMN easily)."""
        logger.warning("Downgrade not fully supported - would require table recreation")
        # In SQLite, dropping columns requires recreating the table
        # For now, just drop the indexes
        conn.execute("DROP INDEX IF EXISTS idx_command_status")
        conn.execute("DROP INDEX IF EXISTS idx_workflow_timestamp")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify status field and indexes were added."""
        # Check status column exists
        cursor = conn.execute("PRAGMA table_info(command_executions)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'status' not in columns:
            logger.error("status column not found in command_executions")
            return False

        # Check indexes exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_command_status',
                'idx_workflow_timestamp'
            )
        """)
        indexes = cursor.fetchall()
        if len(indexes) != 2:
            logger.error(f"Expected 2 new indexes, found {len(indexes)}")
            return False

        # Check backfill worked (count records with status)
        cursor = conn.execute("""
            SELECT COUNT(*) FROM command_executions
            WHERE status IS NOT NULL
        """)
        count = cursor.fetchone()[0]
        logger.info(f"✓ Found {count} records with status field")

        return True


def get_migrations() -> List[Migration]:
    """
    Get all system actions migrations in order.

    Returns:
        List of Migration instances
    """
    from .add_agent_lifecycle_events import AgentLifecycleEventsMigration

    return [
        SystemActionsMigration001(),
        SystemActionsMigration002(),
        AgentLifecycleEventsMigration()
    ]
