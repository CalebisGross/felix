"""
ThresholdLearner - Optimizes confidence and spawning thresholds per task type.

Felix uses multiple thresholds for decision-making:
- confidence_threshold (0.8): Synthesis quality gate
- team_expansion_threshold (0.7): Trigger additional agents
- volatility_threshold (0.15): Spawn critic for unstable confidence
- web_search_threshold (0.7): Trigger external research

This module learns optimal threshold values per task type by tracking
threshold performance vs workflow success rates.
"""

import sqlite3
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import statistics

from .db_utils import get_connection_with_wal, retry_on_locked

logger = logging.getLogger(__name__)


# Standard thresholds in Felix
STANDARD_THRESHOLDS = {
    'confidence_threshold': 0.8,
    'team_expansion_threshold': 0.7,
    'volatility_threshold': 0.15,
    'web_search_threshold': 0.7,
    'max_agents': 10
}


@dataclass
class ThresholdRecord:
    """Record of learned threshold for a task type."""
    threshold_id: int
    task_type: str
    threshold_name: str
    learned_value: float
    confidence: float
    sample_size: int
    success_rate: float
    last_updated: float


class ThresholdLearner:
    """
    Learns optimal thresholds per task type through performance tracking.

    Analyzes the relationship between threshold values and workflow success
    to recommend task-specific threshold adjustments.
    """

    def __init__(self,
                 db_path: Path = None,
                 min_samples: int = 15,
                 learning_rate: float = 0.1):
        """
        Initialize ThresholdLearner.

        Args:
            db_path: Path to felix_task_memory.db (default: auto-detect)
            min_samples: Minimum samples before learning thresholds (default: 15)
            learning_rate: How quickly to adjust thresholds (default: 0.1)
        """
        if db_path is None:
            db_path = Path("felix_task_memory.db")
        self.db_path = db_path

        self.min_samples = min_samples
        self.learning_rate = learning_rate

        # Cache for learned thresholds
        self._threshold_cache = {}
        self._cache_timestamp = {}
        self._cache_ttl = 300  # 5 minutes

        logger.info(f"ThresholdLearner initialized (min_samples={min_samples}, "
                   f"learning_rate={learning_rate})")

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with WAL mode enabled."""
        return get_connection_with_wal(self.db_path, timeout=30.0)

    @retry_on_locked(max_attempts=3)
    def record_threshold_performance(self,
                                      task_type: str,
                                      threshold_name: str,
                                      threshold_value: float,
                                      workflow_success: bool,
                                      workflow_id: Optional[str] = None) -> bool:
        """
        Record performance of a threshold value for a task type.

        Args:
            task_type: Type of task (research, coding, analysis, etc.)
            threshold_name: Name of threshold (confidence_threshold, etc.)
            threshold_value: Value that was used (0.0-1.0)
            workflow_success: Whether workflow succeeded
            workflow_id: Optional workflow ID for tracking

        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            conn = self._get_connection()

            success_float = 1.0 if workflow_success else 0.0

            # Check if learned threshold record exists
            cursor = conn.execute("""
                SELECT threshold_id, learned_value, confidence, sample_size, success_rate
                FROM learned_thresholds
                WHERE task_type = ? AND threshold_name = ?
            """, (task_type, threshold_name))

            row = cursor.fetchone()

            if row:
                # Update existing record
                threshold_id, learned_value, confidence, sample_size, success_rate = row

                new_sample_size = sample_size + 1
                new_success_rate = (success_rate * sample_size + success_float) / new_sample_size

                # Adjust learned value based on performance
                # If success rate is improving and we're using higher thresholds, increase threshold
                # If success rate is declining, decrease threshold
                if new_success_rate > success_rate and threshold_value > learned_value:
                    # Higher threshold led to better success - nudge upward
                    adjustment = self.learning_rate * (threshold_value - learned_value)
                    new_learned_value = learned_value + adjustment
                elif new_success_rate < success_rate and threshold_value < learned_value:
                    # Lower threshold led to worse success - nudge downward
                    adjustment = self.learning_rate * (threshold_value - learned_value)
                    new_learned_value = learned_value + adjustment
                else:
                    # No clear signal - small adjustment toward observed value
                    adjustment = (self.learning_rate / 2) * (threshold_value - learned_value)
                    new_learned_value = learned_value + adjustment

                # Clamp to reasonable range
                new_learned_value = max(0.1, min(0.95, new_learned_value))

                # Update confidence based on sample size
                new_confidence = min(0.95, sample_size / (sample_size + 10))

                conn.execute("""
                    UPDATE learned_thresholds
                    SET learned_value = ?,
                        confidence = ?,
                        sample_size = ?,
                        success_rate = ?,
                        last_updated = ?
                    WHERE threshold_id = ?
                """, (new_learned_value, new_confidence, new_sample_size,
                      new_success_rate, time.time(), threshold_id))

                logger.debug(f"Updated threshold {threshold_name} for {task_type}: "
                            f"{learned_value:.3f} → {new_learned_value:.3f} "
                            f"(success_rate={new_success_rate:.1%}, samples={new_sample_size})")

            else:
                # Create new record
                initial_confidence = 0.1  # Low confidence with just 1 sample
                conn.execute("""
                    INSERT INTO learned_thresholds
                    (task_type, threshold_name, learned_value, confidence,
                     sample_size, success_rate, last_updated)
                    VALUES (?, ?, ?, ?, 1, ?, ?)
                """, (task_type, threshold_name, threshold_value,
                      initial_confidence, success_float, time.time()))

                logger.info(f"Created threshold record for {task_type}/{threshold_name}")

            conn.commit()
            conn.close()

            # Invalidate cache
            cache_key = f"{task_type}:{threshold_name}"
            if cache_key in self._threshold_cache:
                del self._threshold_cache[cache_key]

            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to record threshold performance: {e}")
            return False

    def get_learned_threshold(self,
                              task_type: str,
                              threshold_name: str,
                              use_cache: bool = True) -> Optional[float]:
        """
        Get learned threshold value for task type.

        Returns None if insufficient data or no learned threshold exists.
        Use get_threshold_with_fallback() to get standard value as fallback.

        Args:
            task_type: Type of task
            threshold_name: Name of threshold
            use_cache: Whether to use cached values (default: True)

        Returns:
            Learned threshold value or None
        """
        cache_key = f"{task_type}:{threshold_name}"

        # Check cache
        if use_cache and cache_key in self._threshold_cache:
            cache_age = time.time() - self._cache_timestamp.get(cache_key, 0)
            if cache_age < self._cache_ttl:
                return self._threshold_cache[cache_key]

        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT learned_value, confidence, sample_size
                FROM learned_thresholds
                WHERE task_type = ? AND threshold_name = ?
            """, (task_type, threshold_name))

            row = cursor.fetchone()
            conn.close()

            if row and row['sample_size'] >= self.min_samples and row['confidence'] >= 0.5:
                learned_value = row['learned_value']

                # Update cache
                self._threshold_cache[cache_key] = learned_value
                self._cache_timestamp[cache_key] = time.time()

                logger.debug(f"Learned threshold for {task_type}/{threshold_name}: {learned_value:.3f} "
                            f"(confidence={row['confidence']:.1%}, samples={row['sample_size']})")
                return learned_value
            else:
                if row:
                    logger.debug(f"Insufficient data for {task_type}/{threshold_name} "
                                f"({row['sample_size']}/{self.min_samples} samples)")
                return None

        except sqlite3.Error as e:
            logger.error(f"Failed to get learned threshold: {e}")
            return None

    def get_threshold_with_fallback(self,
                                     task_type: str,
                                     threshold_name: str) -> float:
        """
        Get learned threshold with fallback to standard value.

        This is the main method to use in production - always returns a value.

        Args:
            task_type: Type of task
            threshold_name: Name of threshold

        Returns:
            Learned threshold if available, otherwise standard threshold
        """
        learned = self.get_learned_threshold(task_type, threshold_name)

        if learned is not None:
            return learned

        # Fallback to standard threshold
        standard = STANDARD_THRESHOLDS.get(threshold_name)
        if standard is None:
            logger.warning(f"Unknown threshold name: {threshold_name} - using 0.75")
            return 0.75

        return standard

    def get_threshold_statistics(self,
                                  threshold_name: Optional[str] = None,
                                  days: int = 30) -> Dict[str, Any]:
        """
        Get statistics on learned thresholds.

        Args:
            threshold_name: Optional threshold to filter by (default: all)
            days: Number of days to include (default: 30)

        Returns:
            Dictionary with threshold statistics
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cutoff_time = time.time() - (days * 24 * 60 * 60)

            if threshold_name:
                query = """
                    SELECT
                        task_type,
                        threshold_name,
                        learned_value,
                        confidence,
                        sample_size,
                        success_rate
                    FROM learned_thresholds
                    WHERE threshold_name = ? AND last_updated > ?
                    ORDER BY sample_size DESC
                """
                params = (threshold_name, cutoff_time)
            else:
                query = """
                    SELECT
                        task_type,
                        threshold_name,
                        learned_value,
                        confidence,
                        sample_size,
                        success_rate
                    FROM learned_thresholds
                    WHERE last_updated > ?
                    ORDER BY threshold_name, sample_size DESC
                """
                params = (cutoff_time,)

            cursor = conn.execute(query, params)
            records = [dict(row) for row in cursor.fetchall()]

            conn.close()

            # Calculate aggregate statistics
            if records:
                total_samples = sum(r['sample_size'] for r in records)
                avg_success_rate = statistics.mean(r['success_rate'] for r in records)
                avg_confidence = statistics.mean(r['confidence'] for r in records)
            else:
                total_samples = 0
                avg_success_rate = 0.0
                avg_confidence = 0.0

            return {
                'total_records': len(records),
                'total_samples': total_samples,
                'avg_success_rate': avg_success_rate,
                'avg_confidence': avg_confidence,
                'records': records,
                'days': days
            }

        except sqlite3.Error as e:
            logger.error(f"Failed to get threshold statistics: {e}")
            return {}

    def optimize_threshold(self,
                           task_type: str,
                           threshold_name: str,
                           target_success_rate: float = 0.85) -> Optional[float]:
        """
        Suggest optimal threshold value for a task type.

        Analyzes historical performance to recommend a threshold value
        that would achieve the target success rate.

        Args:
            task_type: Type of task
            threshold_name: Name of threshold
            target_success_rate: Desired success rate (default: 0.85)

        Returns:
            Recommended threshold value or None if insufficient data
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT learned_value, success_rate, sample_size
                FROM learned_thresholds
                WHERE task_type = ? AND threshold_name = ?
            """, (task_type, threshold_name))

            row = cursor.fetchone()
            conn.close()

            if not row or row['sample_size'] < self.min_samples:
                logger.info(f"Insufficient data to optimize {threshold_name} for {task_type}")
                return None

            current_value = row['learned_value']
            current_success = row['success_rate']

            if abs(current_success - target_success_rate) < 0.05:
                # Already near target
                logger.info(f"Current threshold {current_value:.3f} is near target "
                           f"({current_success:.1%} vs {target_success_rate:.1%})")
                return current_value

            # Simple linear adjustment
            # If success rate is too low, decrease threshold (make it easier)
            # If success rate is too high, increase threshold (make it harder)
            success_gap = target_success_rate - current_success
            adjustment = success_gap * 0.2  # Scale adjustment

            optimized_value = current_value + adjustment
            optimized_value = max(0.1, min(0.95, optimized_value))

            logger.info(f"Optimized {threshold_name} for {task_type}: "
                       f"{current_value:.3f} → {optimized_value:.3f} "
                       f"(success: {current_success:.1%} → target: {target_success_rate:.1%})")

            return optimized_value

        except sqlite3.Error as e:
            logger.error(f"Failed to optimize threshold: {e}")
            return None

    def get_all_threshold_records(self) -> List[ThresholdRecord]:
        """
        Get all learned threshold records.

        Returns:
            List of ThresholdRecord objects
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT * FROM learned_thresholds
                ORDER BY last_updated DESC
            """)

            records = []
            for row in cursor.fetchall():
                record = ThresholdRecord(
                    threshold_id=row['threshold_id'],
                    task_type=row['task_type'],
                    threshold_name=row['threshold_name'],
                    learned_value=row['learned_value'],
                    confidence=row['confidence'],
                    sample_size=row['sample_size'],
                    success_rate=row['success_rate'],
                    last_updated=row['last_updated']
                )
                records.append(record)

            conn.close()
            return records

        except sqlite3.Error as e:
            logger.error(f"Failed to get threshold records: {e}")
            return []

    def compare_with_standard(self, task_type: str) -> Dict[str, Any]:
        """
        Compare learned thresholds with standard thresholds.

        Args:
            task_type: Type of task

        Returns:
            Dictionary comparing learned vs standard values
        """
        comparisons = {}

        for threshold_name, standard_value in STANDARD_THRESHOLDS.items():
            learned_value = self.get_learned_threshold(task_type, threshold_name)

            if learned_value is not None:
                difference = learned_value - standard_value
                percent_change = (difference / standard_value) * 100

                comparisons[threshold_name] = {
                    'standard': standard_value,
                    'learned': learned_value,
                    'difference': difference,
                    'percent_change': percent_change,
                    'recommendation': 'increase' if difference > 0 else 'decrease' if difference < 0 else 'keep'
                }

        return comparisons

    @retry_on_locked(max_attempts=3)
    def reset_thresholds(self, task_type: Optional[str] = None) -> int:
        """
        Reset learned thresholds (useful for retraining).

        Args:
            task_type: Optional task type to reset (default: reset all)

        Returns:
            Number of records deleted
        """
        try:
            conn = self._get_connection()

            if task_type:
                cursor = conn.execute("""
                    DELETE FROM learned_thresholds
                    WHERE task_type = ?
                """, (task_type,))
            else:
                cursor = conn.execute("DELETE FROM learned_thresholds")

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            # Clear cache
            self._threshold_cache.clear()
            self._cache_timestamp.clear()

            logger.info(f"Reset {deleted_count} threshold records" +
                       (f" for {task_type}" if task_type else ""))

            return deleted_count

        except sqlite3.Error as e:
            logger.error(f"Failed to reset thresholds: {e}")
            return 0
