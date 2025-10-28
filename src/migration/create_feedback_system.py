"""
Create feedback system tables for learning and quality control.

This migration adds tables to felix_workflow_history.db for:
- Quick workflow ratings (thumbs up/down)
- Detailed workflow feedback (accuracy, relevance, completeness)
- Knowledge-level feedback (correct/incorrect/unsure)
- Feedback patterns for source reliability tracking
"""

import sqlite3
import logging
from typing import List
from .base_migration import Migration

logger = logging.getLogger(__name__)


class FeedbackSystemMigration001(Migration):
    """Create all feedback system tables with indexes."""

    version = 1
    description = "Create workflow ratings, feedback, and knowledge feedback tables"

    def up(self, conn: sqlite3.Connection):
        """Create feedback system schema."""
        logger.info("Creating feedback system database schema...")

        # Tier 1: Quick workflow ratings
        logger.info("Creating workflow_ratings table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_ratings (
                rating_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                rating TEXT NOT NULL CHECK(rating IN ('positive', 'negative')),
                created_at REAL NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflow_outputs(workflow_id) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_ratings
            ON workflow_ratings(workflow_id, created_at DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rating_type
            ON workflow_ratings(rating, created_at DESC)
        """)

        logger.info("✓ Created workflow_ratings with 2 indexes")

        # Tier 2: Detailed workflow feedback
        logger.info("Creating workflow_feedback_detailed table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_feedback_detailed (
                feedback_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                accuracy_rating INTEGER CHECK(accuracy_rating BETWEEN 1 AND 5),
                relevance_rating INTEGER CHECK(relevance_rating BETWEEN 1 AND 5),
                completeness_rating INTEGER CHECK(completeness_rating BETWEEN 1 AND 5),
                user_comments TEXT,
                created_at REAL NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflow_outputs(workflow_id) ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_detailed_feedback_workflow
            ON workflow_feedback_detailed(workflow_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_detailed_feedback_quality
            ON workflow_feedback_detailed(
                accuracy_rating DESC,
                relevance_rating DESC,
                completeness_rating DESC
            )
        """)

        logger.info("✓ Created workflow_feedback_detailed with 2 indexes")

        # Tier 3: Knowledge-level feedback
        logger.info("Creating knowledge_feedback table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_feedback (
                feedback_id TEXT PRIMARY KEY,
                knowledge_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL CHECK(feedback_type IN ('correct', 'incorrect', 'unsure')),
                reason_category TEXT CHECK(reason_category IN (
                    'factually_wrong',
                    'outdated',
                    'irrelevant',
                    'missing_context',
                    'other'
                )),
                reason_detail TEXT,
                correction_suggestion TEXT,
                created_at REAL NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_feedback_entry
            ON knowledge_feedback(knowledge_id, created_at DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_feedback_type
            ON knowledge_feedback(feedback_type, created_at DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_feedback_category
            ON knowledge_feedback(reason_category)
        """)

        logger.info("✓ Created knowledge_feedback with 3 indexes")

        # Feedback patterns for source reliability
        logger.info("Creating feedback_patterns table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL CHECK(pattern_type IN (
                    'source_reliability',
                    'agent_accuracy',
                    'domain_quality',
                    'task_type_success'
                )),
                pattern_key TEXT NOT NULL,
                positive_count INTEGER DEFAULT 0,
                negative_count INTEGER DEFAULT 0,
                reliability_score REAL DEFAULT 0.5,
                sample_size INTEGER DEFAULT 0,
                last_updated REAL NOT NULL,
                UNIQUE(pattern_type, pattern_key)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_patterns_type
            ON feedback_patterns(pattern_type, reliability_score DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_feedback_patterns_reliability
            ON feedback_patterns(reliability_score DESC, sample_size DESC)
        """)

        logger.info("✓ Created feedback_patterns with 2 indexes")

        # Summary
        logger.info("\n" + "="*60)
        logger.info("FEEDBACK SYSTEM DATABASE CREATED")
        logger.info("="*60)
        logger.info("Tables created:")
        logger.info("  - workflow_ratings (quick feedback)")
        logger.info("  - workflow_feedback_detailed (detailed ratings)")
        logger.info("  - knowledge_feedback (entry-level feedback)")
        logger.info("  - feedback_patterns (reliability tracking)")
        logger.info("Total indexes: 9")
        logger.info("="*60)

    def down(self, conn: sqlite3.Connection):
        """Remove all feedback system tables."""
        conn.execute("DROP TABLE IF EXISTS feedback_patterns")
        conn.execute("DROP TABLE IF EXISTS knowledge_feedback")
        conn.execute("DROP TABLE IF EXISTS workflow_feedback_detailed")
        conn.execute("DROP TABLE IF EXISTS workflow_ratings")
        logger.info("✓ Removed all feedback system tables")

    def verify(self, conn: sqlite3.Connection) -> bool:
        """Verify all tables and indexes were created."""
        # Check tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name IN (
                'workflow_ratings',
                'workflow_feedback_detailed',
                'knowledge_feedback',
                'feedback_patterns'
            )
        """)
        tables = cursor.fetchall()
        if len(tables) != 4:
            logger.error(f"Expected 4 tables, found {len(tables)}")
            return False

        # Check key indexes exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='index' AND name IN (
                'idx_workflow_ratings',
                'idx_detailed_feedback_workflow',
                'idx_knowledge_feedback_entry',
                'idx_feedback_patterns_type'
            )
        """)
        indexes = cursor.fetchall()
        if len(indexes) != 4:
            logger.error(f"Expected at least 4 key indexes, found {len(indexes)}")
            return False

        logger.info("✓ All feedback tables and indexes verified")
        return True


def get_migrations() -> List[Migration]:
    """
    Get all feedback system migrations in order.

    Returns:
        List of Migration instances
    """
    return [
        FeedbackSystemMigration001()
    ]
