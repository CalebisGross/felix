"""
PatternLearner - Activates TaskMemory recommendations for workflow optimization.

This module bridges TaskMemory.recommend_strategy() with workflow execution,
determining when to auto-apply high-confidence recommendations vs showing
suggestions to users.

Key Logic:
- Auto-apply: success_probability ≥ 0.95 AND patterns_used ≥ 20
- Recommend: success_probability ≥ 0.80 AND patterns_used ≥ 10
- Ignore: Below thresholds
"""

import sqlite3
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from .db_utils import get_connection_with_wal, retry_on_locked
from src.memory.task_memory import TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class WorkflowRecommendation:
    """Recommendation for workflow optimization."""
    recommendation_id: str
    task_description: str
    task_type: str
    task_complexity: str
    recommended_strategies: List[str]
    recommended_agents: List[str]
    estimated_duration: Optional[float]
    success_probability: float
    patterns_used: int
    potential_issues: List[str]
    confidence_level: str  # 'high', 'medium', 'low'
    should_auto_apply: bool
    reason: str
    timestamp: float


class PatternLearner:
    """
    Bridges TaskMemory recommendations with workflow execution.

    Analyzes historical patterns to provide actionable recommendations
    for agent spawning, strategy selection, and workflow configuration.
    """

    def __init__(self,
                 task_memory,
                 db_path: Path = None,
                 min_samples: int = 10,
                 confidence_threshold: float = 0.8,
                 auto_apply_threshold: float = 0.95,
                 auto_apply_min_samples: int = 20):
        """
        Initialize PatternLearner.

        Args:
            task_memory: TaskMemory instance for pattern queries
            db_path: Path to felix_task_memory.db (default: auto-detect)
            min_samples: Minimum pattern samples to recommend (default: 10)
            confidence_threshold: Minimum success probability for recommendations (default: 0.8)
            auto_apply_threshold: Success probability for auto-apply (default: 0.95)
            auto_apply_min_samples: Minimum samples for auto-apply (default: 20)
        """
        self.task_memory = task_memory

        if db_path is None:
            db_path = Path("felix_task_memory.db")
        self.db_path = db_path

        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        self.auto_apply_threshold = auto_apply_threshold
        self.auto_apply_min_samples = auto_apply_min_samples

        # Ensure database table exists
        self._ensure_database()

        logger.info(f"PatternLearner initialized (min_samples={min_samples}, "
                   f"confidence_threshold={confidence_threshold}, "
                   f"auto_apply_threshold={auto_apply_threshold})")

    def _ensure_database(self):
        """Ensure database table exists."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_recommendations (
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
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with WAL mode enabled."""
        return get_connection_with_wal(self.db_path, timeout=30.0)

    def get_workflow_recommendations(self,
                                     task_description: str,
                                     task_type: str = "general",
                                     task_complexity: TaskComplexity = TaskComplexity.MODERATE) -> Optional[WorkflowRecommendation]:
        """
        Get workflow recommendations based on historical patterns.

        Args:
            task_description: Description of the task to optimize
            task_type: Type of task (research, coding, analysis, etc.)
            task_complexity: Complexity level (TaskComplexity enum)

        Returns:
            WorkflowRecommendation if patterns found, None otherwise
        """
        try:
            # Query TaskMemory for strategy recommendation
            recommendation_data = self.task_memory.recommend_strategy(
                task_description=task_description,
                task_type=task_type,
                complexity=task_complexity
            )

            if not recommendation_data:
                logger.info("No historical patterns found for this task type")
                return None

            # Extract recommendation components
            strategies = recommendation_data.get('strategies', [])
            agents = recommendation_data.get('agents', [])
            estimated_duration = recommendation_data.get('estimated_duration')
            success_probability = recommendation_data.get('success_probability', 0.0)
            potential_issues = recommendation_data.get('potential_issues', [])
            patterns_used = recommendation_data.get('patterns_used', 0)

            # Determine if recommendation meets thresholds
            should_auto_apply, reason = self._should_apply_recommendation(
                success_probability=success_probability,
                patterns_used=patterns_used
            )

            # Assign confidence level
            if success_probability >= self.auto_apply_threshold and patterns_used >= self.auto_apply_min_samples:
                confidence_level = 'high'
            elif success_probability >= self.confidence_threshold and patterns_used >= self.min_samples:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'

            # Create recommendation object
            recommendation = WorkflowRecommendation(
                recommendation_id=str(uuid.uuid4()),
                task_description=task_description,
                task_type=task_type,
                task_complexity=task_complexity.value,
                recommended_strategies=strategies,
                recommended_agents=agents,
                estimated_duration=estimated_duration,
                success_probability=success_probability,
                patterns_used=patterns_used,
                potential_issues=potential_issues,
                confidence_level=confidence_level,
                should_auto_apply=should_auto_apply,
                reason=reason,
                timestamp=time.time()
            )

            logger.info(f"Generated recommendation (confidence={confidence_level}, "
                       f"auto_apply={should_auto_apply}, "
                       f"success_prob={success_probability:.2f}, "
                       f"patterns={patterns_used})")

            return recommendation

        except Exception as e:
            logger.error(f"Failed to get workflow recommendations: {e}")
            return None

    def _should_apply_recommendation(self,
                                     success_probability: float,
                                     patterns_used: int) -> Tuple[bool, str]:
        """
        Determine if recommendation should be auto-applied.

        Args:
            success_probability: Historical success rate (0.0-1.0)
            patterns_used: Number of historical patterns used

        Returns:
            Tuple of (should_auto_apply: bool, reason: str)
        """
        # Auto-apply if very high confidence with many samples
        if success_probability >= self.auto_apply_threshold and patterns_used >= self.auto_apply_min_samples:
            return (True, f"High confidence: {success_probability:.1%} success over {patterns_used} patterns")

        # Recommend but don't auto-apply if moderate confidence
        if success_probability >= self.confidence_threshold and patterns_used >= self.min_samples:
            return (False, f"Moderate confidence: {success_probability:.1%} success over {patterns_used} patterns")

        # Don't recommend if below thresholds
        if patterns_used < self.min_samples:
            return (False, f"Insufficient data: only {patterns_used} patterns (need {self.min_samples})")

        return (False, f"Low success rate: {success_probability:.1%} (need {self.confidence_threshold:.1%})")

    def apply_to_spawning(self,
                          recommendation: WorkflowRecommendation,
                          agent_factory) -> Dict[str, Any]:
        """
        Apply recommendation to agent spawning configuration.

        Modifies agent factory parameters based on high-confidence recommendations.

        Args:
            recommendation: WorkflowRecommendation to apply
            agent_factory: AgentFactory instance to modify

        Returns:
            Dictionary of applied changes
        """
        if not recommendation.should_auto_apply:
            logger.warning("Attempted to apply recommendation that shouldn't be auto-applied")
            return {}

        try:
            applied_changes = {}

            # Apply recommended agent types
            if recommendation.recommended_agents:
                # This would require extending AgentFactory with a preference system
                # For now, we log the recommendation
                applied_changes['preferred_agent_types'] = recommendation.recommended_agents
                logger.info(f"Recommendation: Prefer agent types {recommendation.recommended_agents}")

            # Apply recommended strategies
            if recommendation.recommended_strategies:
                applied_changes['preferred_strategies'] = recommendation.recommended_strategies
                logger.info(f"Recommendation: Use strategies {recommendation.recommended_strategies}")

            # Adjust spawn timing based on complexity
            if recommendation.task_complexity == 'complex':
                applied_changes['spawn_timing'] = 'early_and_frequent'
                logger.info("Recommendation: Spawn agents early and frequently for complex task")
            elif recommendation.task_complexity == 'simple':
                applied_changes['spawn_timing'] = 'minimal'
                logger.info("Recommendation: Use minimal spawning for simple task")

            # Warn about potential issues
            if recommendation.potential_issues:
                applied_changes['warnings'] = recommendation.potential_issues
                for issue in recommendation.potential_issues:
                    logger.warning(f"Historical pattern warning: {issue}")

            logger.info(f"Applied {len(applied_changes)} recommendation changes")
            return applied_changes

        except Exception as e:
            logger.error(f"Failed to apply recommendation to spawning: {e}")
            return {}

    @retry_on_locked(max_attempts=3)
    def record_recommendation_outcome(self,
                                      recommendation_id: str,
                                      workflow_id: str,
                                      was_applied: bool,
                                      workflow_success: bool,
                                      actual_duration: Optional[float] = None,
                                      user_notes: Optional[str] = None) -> bool:
        """
        Record outcome of a recommendation for learning.

        Stores recommendation tracking in pattern_recommendations table
        for future optimization of recommendation logic.

        Args:
            recommendation_id: ID of the recommendation
            workflow_id: ID of the workflow that used the recommendation
            was_applied: Whether recommendation was actually applied
            workflow_success: Whether workflow succeeded
            actual_duration: Actual workflow duration (seconds)
            user_notes: Optional user feedback

        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            conn = self._get_connection()

            # Fetch recommendation details
            cursor = conn.execute("""
                SELECT recommended_strategies, recommended_agents
                FROM pattern_recommendations
                WHERE recommendation_id = ?
            """, (recommendation_id,))

            row = cursor.fetchone()
            if row:
                # Update existing record
                conn.execute("""
                    UPDATE pattern_recommendations
                    SET workflow_id = ?,
                        applied = ?,
                        workflow_success = ?,
                        actual_duration = ?,
                        user_notes = ?,
                        recorded_at = ?
                    WHERE recommendation_id = ?
                """, (workflow_id, was_applied, workflow_success, actual_duration,
                      user_notes, time.time(), recommendation_id))
            else:
                # Insert new record (if recommendation wasn't pre-stored)
                logger.warning(f"Recommendation {recommendation_id} not found - creating new record")
                conn.execute("""
                    INSERT INTO pattern_recommendations
                    (recommendation_id, workflow_id, pattern_id, recommended_strategies,
                     recommended_agents, applied, workflow_success, actual_duration,
                     user_notes, recorded_at)
                    VALUES (?, ?, NULL, '[]', '[]', ?, ?, ?, ?, ?)
                """, (recommendation_id, workflow_id, was_applied, workflow_success,
                      actual_duration, user_notes, time.time()))

            conn.commit()
            conn.close()

            logger.info(f"Recorded recommendation outcome (success={workflow_success}, applied={was_applied})")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to record recommendation outcome: {e}")
            return False

    @retry_on_locked(max_attempts=3)
    def store_recommendation(self, recommendation: WorkflowRecommendation) -> bool:
        """
        Store recommendation for later outcome tracking.

        Args:
            recommendation: WorkflowRecommendation to store

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            conn = self._get_connection()

            # Convert lists to JSON strings
            import json
            strategies_json = json.dumps(recommendation.recommended_strategies)
            agents_json = json.dumps(recommendation.recommended_agents)
            issues_json = json.dumps(recommendation.potential_issues)

            conn.execute("""
                INSERT INTO pattern_recommendations
                (recommendation_id, workflow_id, pattern_id, recommended_strategies,
                 recommended_agents, applied, workflow_success, recorded_at)
                VALUES (?, NULL, NULL, ?, ?, 0, 0, ?)
            """, (recommendation.recommendation_id, strategies_json, agents_json, time.time()))

            conn.commit()
            conn.close()

            logger.debug(f"Stored recommendation {recommendation.recommendation_id[:8]}...")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to store recommendation: {e}")
            return False

    def get_recommendation_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get statistics on recommendation performance.

        Args:
            days: Number of days to include in statistics

        Returns:
            Dictionary with recommendation statistics
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cutoff_time = time.time() - (days * 24 * 60 * 60)

            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_recommendations,
                    SUM(CASE WHEN applied = 1 THEN 1 ELSE 0 END) as applied_count,
                    SUM(CASE WHEN workflow_success = 1 THEN 1 ELSE 0 END) as success_count,
                    AVG(CASE WHEN workflow_success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
                FROM pattern_recommendations
                WHERE recorded_at > ?
            """, (cutoff_time,))

            stats = dict(cursor.fetchone())
            stats['days'] = days
            stats['period_start'] = cutoff_time

            conn.close()

            # Log stats (handle None for empty tables)
            if stats['total_recommendations'] > 0 and stats['success_rate'] is not None:
                logger.debug(f"Recommendation stats: {stats['applied_count']}/{stats['total_recommendations']} applied, "
                            f"{stats['success_rate']:.1%} success rate")
            else:
                logger.debug(f"Recommendation stats: No data yet (0 recommendations)")

            return stats

        except sqlite3.Error as e:
            logger.error(f"Failed to get recommendation statistics: {e}")
            return {}
