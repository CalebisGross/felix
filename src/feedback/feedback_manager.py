"""
Feedback Manager for Felix learning and quality control system.

Provides three-tier feedback collection:
- Tier 1: Quick workflow ratings (positive/negative)
- Tier 2: Detailed workflow feedback (accuracy, relevance, completeness)
- Tier 3: Knowledge-level feedback (correct/incorrect/unsure with corrections)

Aggregates feedback into patterns for source reliability and agent performance tracking.
"""

import sqlite3
import time
import uuid
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of knowledge feedback."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNSURE = "unsure"


class ReasonCategory(Enum):
    """Categories for incorrect knowledge feedback."""
    FACTUALLY_WRONG = "factually_wrong"
    OUTDATED = "outdated"
    IRRELEVANT = "irrelevant"
    MISSING_CONTEXT = "missing_context"
    OTHER = "other"


class PatternType(Enum):
    """Types of feedback patterns."""
    SOURCE_RELIABILITY = "source_reliability"
    AGENT_ACCURACY = "agent_accuracy"
    DOMAIN_QUALITY = "domain_quality"
    TASK_TYPE_SUCCESS = "task_type_success"


@dataclass
class WorkflowRating:
    """Quick workflow rating (Tier 1)."""
    rating_id: str
    workflow_id: str
    rating: str  # 'positive' or 'negative'
    created_at: float

    @classmethod
    def create(cls, workflow_id: str, positive: bool):
        """Create a new workflow rating."""
        return cls(
            rating_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            rating='positive' if positive else 'negative',
            created_at=time.time()
        )


@dataclass
class DetailedFeedback:
    """Detailed workflow feedback (Tier 2)."""
    feedback_id: str
    workflow_id: str
    accuracy_rating: Optional[int]  # 1-5
    relevance_rating: Optional[int]  # 1-5
    completeness_rating: Optional[int]  # 1-5
    user_comments: Optional[str]
    created_at: float

    @classmethod
    def create(cls, workflow_id: str, accuracy: int = None,
               relevance: int = None, completeness: int = None,
               comments: str = None):
        """Create detailed feedback."""
        return cls(
            feedback_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            accuracy_rating=accuracy,
            relevance_rating=relevance,
            completeness_rating=completeness,
            user_comments=comments,
            created_at=time.time()
        )


@dataclass
class KnowledgeFeedback:
    """Knowledge-level feedback (Tier 3)."""
    feedback_id: str
    knowledge_id: str
    feedback_type: str  # 'correct', 'incorrect', 'unsure'
    reason_category: Optional[str]
    reason_detail: Optional[str]
    correction_suggestion: Optional[str]
    created_at: float

    @classmethod
    def create(cls, knowledge_id: str, feedback_type: FeedbackType,
               reason_category: Optional[ReasonCategory] = None,
               reason_detail: str = None,
               correction_suggestion: str = None):
        """Create knowledge feedback."""
        return cls(
            feedback_id=str(uuid.uuid4()),
            knowledge_id=knowledge_id,
            feedback_type=feedback_type.value,
            reason_category=reason_category.value if reason_category else None,
            reason_detail=reason_detail,
            correction_suggestion=correction_suggestion,
            created_at=time.time()
        )


class FeedbackManager:
    """
    Manages feedback collection and aggregation for Felix learning system.

    Responsibilities:
    - Store feedback across three tiers
    - Aggregate feedback into reliability patterns
    - Update related systems (knowledge success rates, confidence calibration)
    - Provide feedback statistics and insights
    """

    def __init__(self, db_path: Path = None):
        """
        Initialize feedback manager.

        Args:
            db_path: Path to felix_workflow_history.db (contains feedback tables)
        """
        if db_path is None:
            db_path = Path("felix_workflow_history.db")

        self.db_path = db_path
        self._ensure_database()

    def _ensure_database(self):
        """Ensure database and tables exist."""
        if not self.db_path.exists():
            logger.warning(f"{self.db_path} does not exist - will be created")
            self.db_path.touch()

        # Ensure required tables exist
        self._ensure_tables()

    def _ensure_tables(self):
        """Create required tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Tier 1: Quick workflow ratings
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_ratings (
                    rating_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK(rating IN ('positive', 'negative')),
                    created_at REAL NOT NULL
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

            # Tier 2: Detailed workflow feedback
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflow_feedback_detailed (
                    feedback_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    accuracy_rating INTEGER CHECK(accuracy_rating BETWEEN 1 AND 5),
                    relevance_rating INTEGER CHECK(relevance_rating BETWEEN 1 AND 5),
                    completeness_rating INTEGER CHECK(completeness_rating BETWEEN 1 AND 5),
                    user_comments TEXT,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_detailed_feedback_workflow
                ON workflow_feedback_detailed(workflow_id)
            """)

            # Tier 3: Knowledge-level feedback
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_feedback (
                    feedback_id TEXT PRIMARY KEY,
                    knowledge_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL CHECK(feedback_type IN ('correct', 'incorrect', 'unsure')),
                    reason_category TEXT CHECK(reason_category IN (
                        'factually_wrong', 'outdated', 'irrelevant', 'missing_context', 'other'
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

            # Feedback patterns for reliability tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL CHECK(pattern_type IN (
                        'source_reliability', 'agent_accuracy', 'domain_quality', 'task_type_success'
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

            conn.commit()
            logger.debug("Feedback system tables ensured")
        except Exception as e:
            logger.error(f"Failed to ensure feedback tables: {e}")
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # Tier 1: Quick Workflow Ratings

    def submit_workflow_rating(self, workflow_id: str, positive: bool) -> WorkflowRating:
        """
        Submit quick workflow rating (thumbs up/down).

        Args:
            workflow_id: Workflow to rate
            positive: True for positive, False for negative

        Returns:
            WorkflowRating instance
        """
        rating = WorkflowRating.create(workflow_id, positive)

        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO workflow_ratings
                (rating_id, workflow_id, rating, created_at)
                VALUES (?, ?, ?, ?)
            """, (rating.rating_id, rating.workflow_id, rating.rating, rating.created_at))

            conn.commit()
            logger.info(f"Stored workflow rating: {workflow_id} = {rating.rating}")

            # Update task pattern based on rating
            self._update_task_pattern_from_rating(workflow_id, positive)

            # Update feedback patterns
            self._update_pattern(conn, PatternType.TASK_TYPE_SUCCESS, workflow_id, positive)

            conn.close()
            return rating

        except sqlite3.Error as e:
            logger.error(f"Failed to store workflow rating: {e}")
            conn.close()
            raise

    def get_workflow_rating(self, workflow_id: str) -> Optional[WorkflowRating]:
        """
        Get workflow rating if exists.

        Args:
            workflow_id: Workflow ID

        Returns:
            WorkflowRating or None
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM workflow_ratings
            WHERE workflow_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (workflow_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return WorkflowRating(**dict(row))
        return None

    # Tier 2: Detailed Workflow Feedback

    def submit_detailed_feedback(self, workflow_id: str,
                                 accuracy: int = None,
                                 relevance: int = None,
                                 completeness: int = None,
                                 comments: str = None) -> DetailedFeedback:
        """
        Submit detailed workflow feedback.

        Args:
            workflow_id: Workflow to rate
            accuracy: Accuracy rating (1-5)
            relevance: Relevance rating (1-5)
            completeness: Completeness rating (1-5)
            comments: Optional user comments

        Returns:
            DetailedFeedback instance
        """
        feedback = DetailedFeedback.create(
            workflow_id, accuracy, relevance, completeness, comments
        )

        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO workflow_feedback_detailed
                (feedback_id, workflow_id, accuracy_rating, relevance_rating,
                 completeness_rating, user_comments, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (feedback.feedback_id, feedback.workflow_id,
                  feedback.accuracy_rating, feedback.relevance_rating,
                  feedback.completeness_rating, feedback.user_comments,
                  feedback.created_at))

            conn.commit()
            logger.info(f"Stored detailed feedback for workflow: {workflow_id}")
            conn.close()

            return feedback

        except sqlite3.Error as e:
            logger.error(f"Failed to store detailed feedback: {e}")
            conn.close()
            raise

    def get_detailed_feedback(self, workflow_id: str) -> Optional[DetailedFeedback]:
        """
        Get detailed feedback for workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            DetailedFeedback or None
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM workflow_feedback_detailed
            WHERE workflow_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (workflow_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return DetailedFeedback(**dict(row))
        return None

    # Tier 3: Knowledge-Level Feedback

    def submit_knowledge_feedback(self, knowledge_id: str,
                                   feedback_type: FeedbackType,
                                   reason_category: Optional[ReasonCategory] = None,
                                   reason_detail: str = None,
                                   correction_suggestion: str = None) -> KnowledgeFeedback:
        """
        Submit knowledge-level feedback.

        Args:
            knowledge_id: Knowledge entry ID
            feedback_type: Correct, incorrect, or unsure
            reason_category: Category if incorrect
            reason_detail: Detailed reason
            correction_suggestion: Suggested correction

        Returns:
            KnowledgeFeedback instance
        """
        feedback = KnowledgeFeedback.create(
            knowledge_id, feedback_type, reason_category,
            reason_detail, correction_suggestion
        )

        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO knowledge_feedback
                (feedback_id, knowledge_id, feedback_type, reason_category,
                 reason_detail, correction_suggestion, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (feedback.feedback_id, feedback.knowledge_id,
                  feedback.feedback_type, feedback.reason_category,
                  feedback.reason_detail, feedback.correction_suggestion,
                  feedback.created_at))

            conn.commit()
            logger.info(f"Stored knowledge feedback: {knowledge_id} = {feedback_type.value}")

            # Update feedback patterns
            is_positive = (feedback_type == FeedbackType.CORRECT)
            self._update_pattern(conn, PatternType.AGENT_ACCURACY, knowledge_id, is_positive)

            conn.close()
            return feedback

        except sqlite3.Error as e:
            logger.error(f"Failed to store knowledge feedback: {e}")
            conn.close()
            raise

    def get_knowledge_feedback(self, knowledge_id: str) -> List[KnowledgeFeedback]:
        """
        Get all feedback for a knowledge entry.

        Args:
            knowledge_id: Knowledge entry ID

        Returns:
            List of KnowledgeFeedback instances
        """
        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT * FROM knowledge_feedback
            WHERE knowledge_id = ?
            ORDER BY created_at DESC
        """, (knowledge_id,))

        rows = cursor.fetchall()
        conn.close()

        return [KnowledgeFeedback(**dict(row)) for row in rows]

    # Pattern Aggregation

    def _update_pattern(self, conn: sqlite3.Connection,
                       pattern_type: PatternType,
                       pattern_key: str,
                       is_positive: bool):
        """
        Update feedback pattern with new feedback.

        Args:
            conn: Database connection
            pattern_type: Type of pattern
            pattern_key: Key for pattern (e.g., source URL, agent ID)
            is_positive: Whether feedback is positive
        """
        pattern_id = f"{pattern_type.value}:{pattern_key}"

        # Get or create pattern
        cursor = conn.execute("""
            SELECT * FROM feedback_patterns
            WHERE pattern_id = ?
        """, (pattern_id,))

        row = cursor.fetchone()

        if row:
            # Update existing pattern
            positive_count = row['positive_count'] + (1 if is_positive else 0)
            negative_count = row['negative_count'] + (0 if is_positive else 1)
            sample_size = positive_count + negative_count
            reliability_score = positive_count / sample_size if sample_size > 0 else 0.5

            conn.execute("""
                UPDATE feedback_patterns
                SET positive_count = ?,
                    negative_count = ?,
                    sample_size = ?,
                    reliability_score = ?,
                    last_updated = ?
                WHERE pattern_id = ?
            """, (positive_count, negative_count, sample_size,
                  reliability_score, time.time(), pattern_id))
        else:
            # Create new pattern
            positive_count = 1 if is_positive else 0
            negative_count = 0 if is_positive else 1
            sample_size = 1
            reliability_score = 1.0 if is_positive else 0.0

            conn.execute("""
                INSERT INTO feedback_patterns
                (pattern_id, pattern_type, pattern_key, positive_count,
                 negative_count, sample_size, reliability_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (pattern_id, pattern_type.value, pattern_key,
                  positive_count, negative_count, sample_size,
                  reliability_score, time.time()))

        conn.commit()

    def get_pattern_reliability(self, pattern_type: PatternType,
                                pattern_key: str) -> Tuple[float, int]:
        """
        Get reliability score for a pattern.

        Args:
            pattern_type: Type of pattern
            pattern_key: Pattern key

        Returns:
            Tuple of (reliability_score, sample_size)
        """
        pattern_id = f"{pattern_type.value}:{pattern_key}"

        conn = self._get_connection()
        cursor = conn.execute("""
            SELECT reliability_score, sample_size
            FROM feedback_patterns
            WHERE pattern_id = ?
        """, (pattern_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return (row['reliability_score'], row['sample_size'])
        return (0.5, 0)  # Default: neutral score, no samples

    def _update_task_pattern_from_rating(self, workflow_id: str, positive: bool):
        """
        Update task memory pattern based on workflow rating.

        This integrates with TaskMemory to update success rates.

        Args:
            workflow_id: Workflow that was rated
            positive: Whether rating was positive
        """
        # This will be integrated with TaskMemory in Phase 2
        # For now, just log the update
        logger.info(f"Task pattern update: workflow {workflow_id} rated {positive}")

    # Statistics and Insights

    def get_workflow_feedback_stats(self, workflow_id: str) -> Dict:
        """
        Get aggregated feedback statistics for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Dictionary with feedback statistics
        """
        conn = self._get_connection()

        # Get quick rating
        cursor = conn.execute("""
            SELECT rating FROM workflow_ratings
            WHERE workflow_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (workflow_id,))
        rating_row = cursor.fetchone()

        # Get detailed feedback
        cursor = conn.execute("""
            SELECT accuracy_rating, relevance_rating, completeness_rating
            FROM workflow_feedback_detailed
            WHERE workflow_id = ?
            ORDER BY created_at DESC LIMIT 1
        """, (workflow_id,))
        detailed_row = cursor.fetchone()

        conn.close()

        stats = {
            'workflow_id': workflow_id,
            'has_rating': rating_row is not None,
            'rating': rating_row['rating'] if rating_row else None,
            'has_detailed': detailed_row is not None,
            'accuracy': detailed_row['accuracy_rating'] if detailed_row else None,
            'relevance': detailed_row['relevance_rating'] if detailed_row else None,
            'completeness': detailed_row['completeness_rating'] if detailed_row else None
        }

        return stats

    def get_global_feedback_stats(self) -> Dict:
        """
        Get global feedback statistics across all workflows.

        Returns:
            Dictionary with global statistics
        """
        conn = self._get_connection()

        # Total ratings
        cursor = conn.execute("SELECT COUNT(*) FROM workflow_ratings")
        total_ratings = cursor.fetchone()[0]

        # Positive ratings
        cursor = conn.execute("""
            SELECT COUNT(*) FROM workflow_ratings WHERE rating = 'positive'
        """)
        positive_ratings = cursor.fetchone()[0]

        # Average detailed ratings
        cursor = conn.execute("""
            SELECT
                AVG(accuracy_rating) as avg_accuracy,
                AVG(relevance_rating) as avg_relevance,
                AVG(completeness_rating) as avg_completeness
            FROM workflow_feedback_detailed
        """)
        avg_row = cursor.fetchone()

        # Knowledge feedback counts
        cursor = conn.execute("""
            SELECT
                feedback_type,
                COUNT(*) as count
            FROM knowledge_feedback
            GROUP BY feedback_type
        """)
        knowledge_counts = {row['feedback_type']: row['count'] for row in cursor.fetchall()}

        conn.close()

        positive_rate = positive_ratings / total_ratings if total_ratings > 0 else 0

        return {
            'total_ratings': total_ratings,
            'positive_ratings': positive_ratings,
            'negative_ratings': total_ratings - positive_ratings,
            'positive_rate': positive_rate,
            'avg_accuracy': avg_row['avg_accuracy'] if avg_row else None,
            'avg_relevance': avg_row['avg_relevance'] if avg_row else None,
            'avg_completeness': avg_row['avg_completeness'] if avg_row else None,
            'knowledge_correct': knowledge_counts.get('correct', 0),
            'knowledge_incorrect': knowledge_counts.get('incorrect', 0),
            'knowledge_unsure': knowledge_counts.get('unsure', 0)
        }
