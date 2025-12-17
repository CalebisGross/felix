"""
Migration: Add Synthesis Audit Table

Creates a comprehensive audit table for synthesis operations. This enables
full traceability of synthesis decisions, confidence calculations, prompts,
and agent contributions - addressing the "black box" issue in the synthesis
engine.

Features:
- Records all synthesis operations with full context
- Captures system/user prompts (critical for auditability)
- Tracks per-agent contributions and confidence calculations
- Stores validation results and flags
- Records degradation reasons and fallback usage
"""

import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def migrate_up(db_path: str = "felix_knowledge.db") -> bool:
    """
    Create the synthesis_audit table with indexes.

    Schema:
    - audit_id: Auto-incrementing primary key
    - workflow_id: ID of the workflow that triggered synthesis
    - timestamp: When synthesis occurred
    - task_description: Original task
    - task_complexity: SIMPLE_FACTUAL, MEDIUM, or COMPLEX

    Agent Contributions:
    - agent_count: Number of agents in synthesis
    - agent_outputs_json: Per-agent data (id, type, confidence, content_preview)
    - reasoning_weights_json: Per-agent reasoning weights applied

    Confidence Calculation:
    - raw_confidences_json: List of individual agent confidences
    - weighted_avg: Weighted average confidence
    - confidence_std: Standard deviation (for disagreement detection)
    - synthesis_confidence: Final synthesis confidence score

    Validation:
    - validation_called: Whether validation functions were invoked
    - validation_score: Validation score (0.0-1.0)
    - validation_flags_json: List of validation issues detected

    Prompts (critical for auditability):
    - system_prompt: Full system prompt sent to LLM
    - user_prompt: Full user prompt with agent outputs

    Output:
    - synthesis_content: Final synthesized output
    - tokens_used: Tokens consumed
    - synthesis_time: Processing time in seconds
    - used_fallback: Whether fallback synthesis was used
    - degraded: Whether output is degraded
    - degraded_reasons_json: List of degradation reasons

    Args:
        db_path: Path to knowledge database

    Returns:
        True if migration successful, False otherwise
    """
    logger.info("Running migration: add_synthesis_audit")

    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # Check if table already exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='synthesis_audit'
        """)

        if cursor.fetchone():
            logger.info("synthesis_audit table already exists, skipping migration")
            conn.close()
            return True

        # Create synthesis audit table
        logger.info("Creating synthesis_audit table...")
        cursor.execute("""
            CREATE TABLE synthesis_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT,
                timestamp REAL NOT NULL,
                task_description TEXT,
                task_complexity TEXT,

                -- Agent contributions
                agent_count INTEGER,
                agent_outputs_json TEXT,
                reasoning_weights_json TEXT,

                -- Confidence calculation
                raw_confidences_json TEXT,
                weighted_avg REAL,
                confidence_std REAL,
                synthesis_confidence REAL,

                -- Validation
                validation_called INTEGER DEFAULT 0,
                validation_score REAL,
                validation_flags_json TEXT,

                -- Prompts (critical for auditability)
                system_prompt TEXT,
                user_prompt TEXT,

                -- Output
                synthesis_content TEXT,
                tokens_used INTEGER,
                synthesis_time REAL,
                used_fallback INTEGER DEFAULT 0,
                degraded INTEGER DEFAULT 0,
                degraded_reasons_json TEXT
            )
        """)

        # Create indexes for fast querying
        logger.info("Creating indexes...")

        # Index on timestamp for recent query lookups
        cursor.execute("""
            CREATE INDEX idx_synthesis_timestamp ON synthesis_audit(timestamp DESC)
        """)

        # Index on workflow_id for correlation with workflow history
        cursor.execute("""
            CREATE INDEX idx_synthesis_workflow ON synthesis_audit(workflow_id)
        """)

        # Index on confidence for finding low-confidence syntheses
        cursor.execute("""
            CREATE INDEX idx_synthesis_confidence ON synthesis_audit(synthesis_confidence)
        """)

        # Index on degraded for finding problematic syntheses
        cursor.execute("""
            CREATE INDEX idx_synthesis_degraded ON synthesis_audit(degraded)
        """)

        # Index on task complexity for analysis
        cursor.execute("""
            CREATE INDEX idx_synthesis_complexity ON synthesis_audit(task_complexity)
        """)

        conn.commit()
        logger.info("Successfully created synthesis_audit table with indexes")

        # Insert initial audit entry to mark migration
        cursor.execute("""
            INSERT INTO synthesis_audit (
                timestamp, task_description, task_complexity, agent_count,
                synthesis_confidence, validation_called, synthesis_content
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().timestamp(),
            'SYSTEM_MIGRATION',
            'SYSTEM',
            0,
            1.0,
            0,
            'Synthesis audit table created - full auditability enabled'
        ))
        conn.commit()

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


def migrate_down(db_path: str = "felix_knowledge.db") -> bool:
    """
    Remove the synthesis_audit table and all indexes.

    WARNING: This will permanently delete all synthesis audit history!

    Args:
        db_path: Path to knowledge database

    Returns:
        True if rollback successful, False otherwise
    """
    logger.info("Rolling back migration: add_synthesis_audit")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop indexes first
        cursor.execute("DROP INDEX IF EXISTS idx_synthesis_timestamp")
        cursor.execute("DROP INDEX IF EXISTS idx_synthesis_workflow")
        cursor.execute("DROP INDEX IF EXISTS idx_synthesis_confidence")
        cursor.execute("DROP INDEX IF EXISTS idx_synthesis_degraded")
        cursor.execute("DROP INDEX IF EXISTS idx_synthesis_complexity")

        # Drop table
        cursor.execute("DROP TABLE IF EXISTS synthesis_audit")

        conn.commit()
        conn.close()
        logger.info("Successfully removed synthesis_audit table")
        return True

    except Exception as e:
        logger.error(f"Rollback failed: {e}")
        if 'conn' in locals():
            conn.close()
        return False


def verify_migration(db_path: str = "felix_knowledge.db") -> bool:
    """
    Verify that the migration was successful.

    Checks:
    1. Table exists with correct schema
    2. All indexes exist
    3. Initial audit entry was created

    Args:
        db_path: Path to knowledge database

    Returns:
        True if verification passed, False otherwise
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='synthesis_audit'
        """)

        if not cursor.fetchone():
            logger.error("Verification failed: synthesis_audit table not found")
            conn.close()
            return False

        # Check schema has required columns
        cursor.execute("PRAGMA table_info(synthesis_audit)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {
            'audit_id', 'workflow_id', 'timestamp', 'task_description',
            'task_complexity', 'agent_count', 'agent_outputs_json',
            'reasoning_weights_json', 'raw_confidences_json', 'weighted_avg',
            'confidence_std', 'synthesis_confidence', 'validation_called',
            'validation_score', 'validation_flags_json', 'system_prompt',
            'user_prompt', 'synthesis_content', 'tokens_used', 'synthesis_time',
            'used_fallback', 'degraded', 'degraded_reasons_json'
        }

        if not required_columns.issubset(columns):
            missing = required_columns - columns
            logger.error(f"Verification failed: missing columns {missing}")
            conn.close()
            return False

        # Check indexes exist
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND tbl_name='synthesis_audit'
        """)

        indexes = {row[0] for row in cursor.fetchall()}
        required_indexes = {
            'idx_synthesis_timestamp',
            'idx_synthesis_workflow',
            'idx_synthesis_confidence',
            'idx_synthesis_degraded',
            'idx_synthesis_complexity'
        }

        if not required_indexes.issubset(indexes):
            missing = required_indexes - indexes
            logger.error(f"Verification failed: missing indexes {missing}")
            conn.close()
            return False

        # Check initial audit entry exists
        cursor.execute("""
            SELECT COUNT(*) FROM synthesis_audit
            WHERE task_description='SYSTEM_MIGRATION'
        """)

        if cursor.fetchone()[0] == 0:
            logger.warning("Initial audit entry not found (may have been deleted)")

        conn.close()
        logger.info("Migration verification passed")
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

    print(f"Running synthesis audit migration on {db_path}...")

    if migrate_up(db_path):
        print("Migration successful!")

        if verify_migration(db_path):
            print("Verification passed!")
            print("\nSynthesis audit logging is now enabled. All synthesis")
            print("operations will be tracked with full context including:")
            print("  - System and user prompts")
            print("  - Per-agent contributions and confidence")
            print("  - Validation results and flags")
            print("  - Degradation reasons")
        else:
            print("Verification failed!")
            sys.exit(1)
    else:
        print("Migration failed!")
        sys.exit(1)
