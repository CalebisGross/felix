"""
Corrective migration to fix Phase 2 learning schema mismatches.

Fixes:
1. learned_thresholds: Remove default_value, change to INTEGER PRIMARY KEY AUTOINCREMENT
2. pattern_recommendations: Change created_at → recorded_at, remove completed_at
3. confidence_calibration: Change to INTEGER PRIMARY KEY AUTOINCREMENT

Strategy: Create new tables, copy data, drop old, rename new.
"""

import sqlite3
import logging
import time
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class FixLearningSchemaV2(Migration):
    """Fix learning schema to match Phase 2 code expectations."""

    version = 2
    description = "Fix learning table schemas (PRIMARY KEY types, column names)"

    def up(self, conn: sqlite3.Connection):
        """Apply corrective schema changes."""
        logger.info("=" * 60)
        logger.info("APPLYING CORRECTIVE SCHEMA MIGRATION")
        logger.info("=" * 60)

        # Check if correction is needed
        if not self._needs_correction(conn):
            logger.info("✓ Schema already correct, skipping migration")
            return

        # Part 1: Fix learned_thresholds
        logger.info("\n1. Fixing learned_thresholds table...")
        self._fix_learned_thresholds(conn)

        # Part 2: Fix pattern_recommendations
        logger.info("\n2. Fixing pattern_recommendations table...")
        self._fix_pattern_recommendations(conn)

        # Part 3: Fix confidence_calibration
        logger.info("\n3. Fixing confidence_calibration table...")
        self._fix_confidence_calibration(conn)

        logger.info("\n" + "=" * 60)
        logger.info("✓ CORRECTIVE MIGRATION COMPLETE")
        logger.info("=" * 60)

    def _needs_correction(self, conn: sqlite3.Connection) -> bool:
        """Check if tables need correction."""
        try:
            # Check if learned_thresholds has default_value column (old schema)
            cursor = conn.execute("PRAGMA table_info(learned_thresholds)")
            columns = [row[1] for row in cursor.fetchall()]
            has_default_value = "default_value" in columns

            if has_default_value:
                logger.info("✓ Detected old schema (has default_value), correction needed")
                return True

            # Check if pattern_recommendations has created_at instead of recorded_at
            cursor = conn.execute("PRAGMA table_info(pattern_recommendations)")
            columns = [row[1] for row in cursor.fetchall()]
            has_created_at = "created_at" in columns
            has_recorded_at = "recorded_at" in columns

            if has_created_at and not has_recorded_at:
                logger.info("✓ Detected old schema (has created_at), correction needed")
                return True

            logger.info("✓ Schema already corrected")
            return False

        except sqlite3.OperationalError:
            # Tables don't exist, no correction needed
            return False

    def _fix_learned_thresholds(self, conn: sqlite3.Connection):
        """Fix learned_thresholds schema."""
        # Create new table with correct schema
        logger.info("  Creating learned_thresholds_new...")
        conn.execute("""
            CREATE TABLE learned_thresholds_new (
                threshold_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                threshold_name TEXT NOT NULL,
                learned_value REAL NOT NULL,
                confidence REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                success_rate REAL NOT NULL,
                last_updated REAL NOT NULL,
                UNIQUE(task_type, threshold_name)
            )
        """)

        # Copy existing data (if any)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM learned_thresholds")
            count = cursor.fetchone()[0]

            if count > 0:
                logger.info(f"  Copying {count} existing records...")
                conn.execute("""
                    INSERT INTO learned_thresholds_new
                    (task_type, threshold_name, learned_value, confidence,
                     sample_size, success_rate, last_updated)
                    SELECT task_type, threshold_name, learned_value, confidence,
                           sample_size, success_rate, last_updated
                    FROM learned_thresholds
                """)
                logger.info(f"  ✓ Copied {count} records")
            else:
                logger.info("  No existing data to copy")
        except sqlite3.Error as e:
            logger.warning(f"  Could not copy data: {e}")

        # Drop old table
        logger.info("  Dropping old table...")
        conn.execute("DROP TABLE IF EXISTS learned_thresholds")

        # Rename new table
        logger.info("  Renaming new table...")
        conn.execute("ALTER TABLE learned_thresholds_new RENAME TO learned_thresholds")

        # Recreate indexes
        logger.info("  Recreating indexes...")
        conn.execute("""
            CREATE INDEX idx_learned_thresholds_task
            ON learned_thresholds(task_type, threshold_name)
        """)
        conn.execute("""
            CREATE INDEX idx_learned_thresholds_confidence
            ON learned_thresholds(confidence DESC, sample_size DESC)
        """)

        logger.info("  ✓ learned_thresholds fixed")

    def _fix_pattern_recommendations(self, conn: sqlite3.Connection):
        """Fix pattern_recommendations schema."""
        # Create new table with correct schema
        logger.info("  Creating pattern_recommendations_new...")
        conn.execute("""
            CREATE TABLE pattern_recommendations_new (
                recommendation_id TEXT PRIMARY KEY,
                workflow_id TEXT,
                pattern_id TEXT,
                recommended_strategies TEXT,
                recommended_agents TEXT,
                applied INTEGER DEFAULT 0,
                workflow_success INTEGER DEFAULT 0,
                actual_duration REAL,
                user_notes TEXT,
                recorded_at REAL
            )
        """)

        # Copy existing data, mapping created_at → recorded_at
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM pattern_recommendations")
            count = cursor.fetchone()[0]

            if count > 0:
                logger.info(f"  Copying {count} existing records...")
                conn.execute("""
                    INSERT INTO pattern_recommendations_new
                    (recommendation_id, workflow_id, pattern_id, recommended_strategies,
                     recommended_agents, applied, workflow_success, recorded_at)
                    SELECT recommendation_id, workflow_id, pattern_id, recommended_strategies,
                           recommended_agents, applied, workflow_success, created_at
                    FROM pattern_recommendations
                """)
                logger.info(f"  ✓ Copied {count} records (created_at → recorded_at)")
            else:
                logger.info("  No existing data to copy")
        except sqlite3.Error as e:
            logger.warning(f"  Could not copy data: {e}")

        # Drop old table
        logger.info("  Dropping old table...")
        conn.execute("DROP TABLE IF EXISTS pattern_recommendations")

        # Rename new table
        logger.info("  Renaming new table...")
        conn.execute("ALTER TABLE pattern_recommendations_new RENAME TO pattern_recommendations")

        # Recreate indexes
        logger.info("  Recreating indexes...")
        conn.execute("""
            CREATE INDEX idx_recommendations_pattern
            ON pattern_recommendations(pattern_id, applied)
        """)
        conn.execute("""
            CREATE INDEX idx_recommendations_success
            ON pattern_recommendations(applied, workflow_success)
        """)
        conn.execute("""
            CREATE INDEX idx_recommendations_workflow
            ON pattern_recommendations(workflow_id)
        """)

        logger.info("  ✓ pattern_recommendations fixed")

    def _fix_confidence_calibration(self, conn: sqlite3.Connection):
        """Fix confidence_calibration schema."""
        # Create new table with correct schema
        logger.info("  Creating confidence_calibration_new...")
        conn.execute("""
            CREATE TABLE confidence_calibration_new (
                calibration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                task_complexity TEXT NOT NULL,
                avg_predicted_confidence REAL NOT NULL,
                avg_actual_success REAL NOT NULL,
                calibration_factor REAL NOT NULL,
                calibration_error REAL NOT NULL,
                sample_size INTEGER NOT NULL,
                last_updated REAL NOT NULL,
                UNIQUE(agent_type, task_complexity)
            )
        """)

        # Copy existing data (if any)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM confidence_calibration")
            count = cursor.fetchone()[0]

            if count > 0:
                logger.info(f"  Copying {count} existing records...")
                conn.execute("""
                    INSERT INTO confidence_calibration_new
                    (agent_type, task_complexity, avg_predicted_confidence,
                     avg_actual_success, calibration_factor, calibration_error,
                     sample_size, last_updated)
                    SELECT agent_type, task_complexity, avg_predicted_confidence,
                           avg_actual_success, calibration_factor, calibration_error,
                           sample_size, last_updated
                    FROM confidence_calibration
                """)
                logger.info(f"  ✓ Copied {count} records")
            else:
                logger.info("  No existing data to copy")
        except sqlite3.Error as e:
            logger.warning(f"  Could not copy data: {e}")

        # Drop old table
        logger.info("  Dropping old table...")
        conn.execute("DROP TABLE IF EXISTS confidence_calibration")

        # Rename new table
        logger.info("  Renaming new table...")
        conn.execute("ALTER TABLE confidence_calibration_new RENAME TO confidence_calibration")

        # Recreate indexes
        logger.info("  Recreating indexes...")
        conn.execute("""
            CREATE INDEX idx_calibration_agent
            ON confidence_calibration(agent_type, task_complexity)
        """)
        conn.execute("""
            CREATE INDEX idx_calibration_error
            ON confidence_calibration(calibration_error ASC)
        """)

        logger.info("  ✓ confidence_calibration fixed")

    def down(self, conn: sqlite3.Connection):
        """Rollback corrective changes (restore original schema)."""
        logger.warning("Rollback not fully implemented - original schema was flawed")
        logger.warning("If you need to rollback, delete felix_task_memory.db and re-run original migrations")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify corrected schema."""
        try:
            # Verify learned_thresholds
            cursor = conn.execute("PRAGMA table_info(learned_thresholds)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            if "default_value" in columns:
                logger.error("✗ learned_thresholds still has default_value column")
                return False

            if "threshold_id" not in columns:
                logger.error("✗ learned_thresholds missing threshold_id")
                return False

            # Verify pattern_recommendations
            cursor = conn.execute("PRAGMA table_info(pattern_recommendations)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            if "created_at" in columns:
                logger.error("✗ pattern_recommendations still has created_at")
                return False

            if "recorded_at" not in columns:
                logger.error("✗ pattern_recommendations missing recorded_at")
                return False

            # Verify confidence_calibration
            cursor = conn.execute("PRAGMA table_info(confidence_calibration)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            if "calibration_id" not in columns:
                logger.error("✗ confidence_calibration missing calibration_id")
                return False

            logger.info("✓ All schemas verified correct")
            return True

        except sqlite3.Error as e:
            logger.error(f"✗ Verification failed: {e}")
            return False


def get_migrations() -> List[Migration]:
    """Get corrective migration."""
    return [
        FixLearningSchemaV2()
    ]
