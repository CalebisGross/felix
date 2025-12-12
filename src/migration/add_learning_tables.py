"""
Add learning system tables to felix_task_memory.db.

This migration adds tables for:
- Learned thresholds per task type
- Confidence calibration factors per agent type
- Pattern recommendation tracking
"""

import sqlite3
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class LearningSystemMigration001(Migration):
    """Create learned thresholds and confidence calibration tables."""

    version = 1
    description = "Create learned_thresholds and confidence_calibration tables"

    def up(self, conn: sqlite3.Connection):
        """Create learning system schema."""
        logger.info("Creating learning system database schema...")

        # Learned thresholds table
        logger.info("Creating learned_thresholds table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS learned_thresholds (
                threshold_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                threshold_name TEXT NOT NULL,
                learned_value REAL NOT NULL,
                default_value REAL NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                sample_size INTEGER NOT NULL DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                last_updated REAL NOT NULL,
                UNIQUE(task_type, threshold_name)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_learned_thresholds_task
            ON learned_thresholds(task_type, threshold_name)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_learned_thresholds_confidence
            ON learned_thresholds(confidence DESC, sample_size DESC)
        """)

        logger.info("✓ Created learned_thresholds with 2 indexes")

        # Confidence calibration table
        logger.info("Creating confidence_calibration table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS confidence_calibration (
                calibration_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                task_complexity TEXT NOT NULL,
                calibration_factor REAL NOT NULL DEFAULT 1.0,
                sample_size INTEGER NOT NULL DEFAULT 0,
                avg_predicted_confidence REAL DEFAULT 0.0,
                avg_actual_success REAL DEFAULT 0.0,
                calibration_error REAL DEFAULT 0.0,
                last_updated REAL NOT NULL,
                UNIQUE(agent_type, task_complexity)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibration_agent
            ON confidence_calibration(agent_type, task_complexity)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_calibration_error
            ON confidence_calibration(calibration_error ASC)
        """)

        logger.info("✓ Created confidence_calibration with 2 indexes")

        # Pattern recommendation tracking (success rate of recommendations)
        logger.info("Creating pattern_recommendations table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pattern_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                pattern_id TEXT,
                task_type TEXT NOT NULL,
                recommended_strategies TEXT NOT NULL,
                recommended_agents TEXT NOT NULL,
                applied BOOLEAN NOT NULL DEFAULT 0,
                workflow_success BOOLEAN,
                execution_time REAL,
                created_at REAL NOT NULL,
                completed_at REAL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendations_pattern
            ON pattern_recommendations(pattern_id, applied)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendations_success
            ON pattern_recommendations(applied, workflow_success)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_recommendations_workflow
            ON pattern_recommendations(workflow_id)
        """)

        logger.info("✓ Created pattern_recommendations with 3 indexes")

        # Summary
        logger.info("\n" + "="*60)
        logger.info("LEARNING SYSTEM DATABASE CREATED")
        logger.info("="*60)
        logger.info("Tables created:")
        logger.info("  - learned_thresholds (adaptive threshold optimization)")
        logger.info("  - confidence_calibration (agent confidence adjustment)")
        logger.info("  - pattern_recommendations (recommendation tracking)")
        logger.info("Total indexes: 7")
        logger.info("="*60)

    def down(self, conn: sqlite3.Connection):
        """Remove all learning system tables."""
        conn.execute("DROP TABLE IF EXISTS pattern_recommendations")
        conn.execute("DROP TABLE IF EXISTS confidence_calibration")
        conn.execute("DROP TABLE IF EXISTS learned_thresholds")
        logger.info("✓ Removed all learning system tables")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify all tables and indexes were created."""
        # Check tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN (
                'learned_thresholds',
                'confidence_calibration',
                'pattern_recommendations'
            )
        """)
        tables = cursor.fetchall()
        if len(tables) != 3:
            logger.error(f"Expected 3 tables, found {len(tables)}")
            return False

        # Check key indexes exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_learned_thresholds_task',
                'idx_calibration_agent',
                'idx_recommendations_pattern'
            )
        """)
        indexes = cursor.fetchall()
        if len(indexes) != 3:
            logger.error(f"Expected at least 3 key indexes, found {len(indexes)}")
            return False

        logger.info("✓ All learning tables and indexes verified")
        return True


def get_migrations() -> List[Migration]:
    """
    Get all learning system migrations in order.

    Returns:
        List of Migration instances
    """
    return [
        LearningSystemMigration001()
    ]
