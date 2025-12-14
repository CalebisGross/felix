"""
ConfidenceCalibrator - Learns and corrects systematic agent confidence biases.

Agents may be systematically overconfident or underconfident. This module
tracks predicted confidence vs actual success rates to calculate calibration
factors that adjust raw confidence scores for more accurate predictions.

Key Concepts:
- Calibration Factor: Multiplier to adjust raw confidence (0.5-2.0 range)
- Overconfident agents: predicted > actual → factor < 1.0
- Underconfident agents: predicted < actual → factor > 1.0
- Well-calibrated agents: predicted ≈ actual → factor ≈ 1.0
"""

import sqlite3
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

from .db_utils import get_connection_with_wal, retry_on_locked
from src.memory.task_memory import TaskComplexity

logger = logging.getLogger(__name__)


@dataclass
class CalibrationRecord:
    """Record of agent calibration data."""
    calibration_id: int
    agent_type: str
    task_complexity: str
    calibration_factor: float
    avg_predicted_confidence: float
    avg_actual_success: float
    calibration_error: float
    sample_size: int
    last_updated: float


class ConfidenceCalibrator:
    """
    Learns systematic biases in agent confidence predictions.

    Tracks the relationship between predicted confidence and actual
    workflow success to adjust future confidence predictions.
    """

    def __init__(self,
                 db_path: Path = None,
                 min_samples: int = 10,
                 max_calibration_factor: float = 2.0,
                 min_calibration_factor: float = 0.5):
        """
        Initialize ConfidenceCalibrator.

        Args:
            db_path: Path to felix_task_memory.db (default: auto-detect)
            min_samples: Minimum samples before calculating calibration (default: 10)
            max_calibration_factor: Maximum calibration multiplier (default: 2.0)
            min_calibration_factor: Minimum calibration multiplier (default: 0.5)
        """
        if db_path is None:
            db_path = Path("felix_task_memory.db")
        self.db_path = db_path

        self.min_samples = min_samples
        self.max_calibration_factor = max_calibration_factor
        self.min_calibration_factor = min_calibration_factor

        # Cache for calibration factors (reduces DB queries)
        self._calibration_cache = {}
        self._cache_timestamp = {}
        self._cache_ttl = 300  # 5 minutes

        # Ensure database table exists
        self._ensure_database()

        logger.info(f"ConfidenceCalibrator initialized (min_samples={min_samples})")

    def _ensure_database(self):
        """Ensure database table exists."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS confidence_calibration (
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calibration_agent
                ON confidence_calibration(agent_type, task_complexity)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_calibration_error
                ON confidence_calibration(calibration_error ASC)
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with WAL mode enabled."""
        return get_connection_with_wal(self.db_path, timeout=30.0)

    @retry_on_locked(max_attempts=3)
    def record_agent_prediction(self,
                                 agent_type: str,
                                 task_complexity: TaskComplexity,
                                 predicted_confidence: float,
                                 actual_success: bool) -> bool:
        """
        Record an agent's confidence prediction and actual outcome.

        This incrementally updates calibration statistics for the given
        agent_type + task_complexity combination.

        Args:
            agent_type: Type of agent (research, analysis, critic, synthesis)
            task_complexity: Complexity level (TaskComplexity enum)
            predicted_confidence: Agent's predicted confidence (0.0-1.0)
            actual_success: Whether workflow actually succeeded (True/False)

        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            conn = self._get_connection()

            # Convert boolean to float for averaging
            actual_success_float = 1.0 if actual_success else 0.0

            # Check if calibration record exists
            cursor = conn.execute("""
                SELECT calibration_id, avg_predicted_confidence, avg_actual_success, sample_size
                FROM confidence_calibration
                WHERE agent_type = ? AND task_complexity = ?
            """, (agent_type, task_complexity.value))

            row = cursor.fetchone()

            if row:
                # Incremental update
                calibration_id, avg_predicted, avg_actual, sample_size = row

                # Calculate new averages
                new_sample_size = sample_size + 1
                new_avg_predicted = (avg_predicted * sample_size + predicted_confidence) / new_sample_size
                new_avg_actual = (avg_actual * sample_size + actual_success_float) / new_sample_size

                # Calculate calibration error (absolute difference)
                calibration_error = abs(new_avg_predicted - new_avg_actual)

                # Calculate calibration factor
                # If avg_actual > avg_predicted: agent is underconfident → factor > 1.0
                # If avg_actual < avg_predicted: agent is overconfident → factor < 1.0
                if new_avg_predicted > 0.01:  # Avoid division by zero
                    calibration_factor = new_avg_actual / new_avg_predicted
                    # Clamp to reasonable range
                    calibration_factor = max(self.min_calibration_factor,
                                             min(self.max_calibration_factor, calibration_factor))
                else:
                    calibration_factor = 1.0

                # Update record
                conn.execute("""
                    UPDATE confidence_calibration
                    SET avg_predicted_confidence = ?,
                        avg_actual_success = ?,
                        calibration_factor = ?,
                        calibration_error = ?,
                        sample_size = ?,
                        last_updated = ?
                    WHERE calibration_id = ?
                """, (new_avg_predicted, new_avg_actual, calibration_factor,
                      calibration_error, new_sample_size, time.time(), calibration_id))

                logger.debug(f"Updated calibration for {agent_type}/{task_complexity.value}: "
                            f"factor={calibration_factor:.3f}, samples={new_sample_size}")

            else:
                # Create new record
                calibration_error = abs(predicted_confidence - actual_success_float)

                if predicted_confidence > 0.01:
                    calibration_factor = actual_success_float / predicted_confidence
                    calibration_factor = max(self.min_calibration_factor,
                                             min(self.max_calibration_factor, calibration_factor))
                else:
                    calibration_factor = 1.0

                conn.execute("""
                    INSERT INTO confidence_calibration
                    (agent_type, task_complexity, avg_predicted_confidence,
                     avg_actual_success, calibration_factor, calibration_error,
                     sample_size, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """, (agent_type, task_complexity.value, predicted_confidence,
                      actual_success_float, calibration_factor, calibration_error, time.time()))

                logger.info(f"Created calibration record for {agent_type}/{task_complexity.value}")

            conn.commit()
            conn.close()

            # Invalidate cache for this agent/complexity
            cache_key = f"{agent_type}:{task_complexity.value}"
            if cache_key in self._calibration_cache:
                del self._calibration_cache[cache_key]

            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to record agent prediction: {e}")
            return False

    def get_calibration_factor(self,
                                agent_type: str,
                                task_complexity: TaskComplexity,
                                use_cache: bool = True) -> float:
        """
        Get calibration factor for agent type + task complexity.

        Returns 1.0 if insufficient data or no calibration record exists.

        Args:
            agent_type: Type of agent
            task_complexity: Complexity level (TaskComplexity enum)
            use_cache: Whether to use cached values (default: True)

        Returns:
            Calibration factor (typically 0.5-2.0, default 1.0)
        """
        cache_key = f"{agent_type}:{task_complexity.value}"

        # Check cache
        if use_cache and cache_key in self._calibration_cache:
            cache_age = time.time() - self._cache_timestamp.get(cache_key, 0)
            if cache_age < self._cache_ttl:
                return self._calibration_cache[cache_key]

        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT calibration_factor, sample_size
                FROM confidence_calibration
                WHERE agent_type = ? AND task_complexity = ?
            """, (agent_type, task_complexity.value))

            row = cursor.fetchone()
            conn.close()

            if row and row['sample_size'] >= self.min_samples:
                factor = row['calibration_factor']

                # Update cache
                self._calibration_cache[cache_key] = factor
                self._cache_timestamp[cache_key] = time.time()

                logger.debug(f"Calibration factor for {agent_type}/{task_complexity}: {factor:.3f}")
                return factor
            else:
                # Insufficient data - return neutral factor
                if row:
                    logger.debug(f"Insufficient samples for {agent_type}/{task_complexity} "
                                f"({row['sample_size']}/{self.min_samples}) - using factor 1.0")
                return 1.0

        except sqlite3.Error as e:
            logger.error(f"Failed to get calibration factor: {e}")
            return 1.0

    def calibrate_confidence(self,
                             agent_type: str,
                             task_complexity: TaskComplexity,
                             raw_confidence: float) -> float:
        """
        Apply calibration to raw confidence score.

        Args:
            agent_type: Type of agent
            task_complexity: Complexity level (TaskComplexity enum)
            raw_confidence: Uncalibrated confidence (0.0-1.0)

        Returns:
            Calibrated confidence (0.0-1.0)
        """
        if not (0.0 <= raw_confidence <= 1.0):
            logger.warning(f"Invalid raw_confidence: {raw_confidence} - clamping to [0,1]")
            raw_confidence = max(0.0, min(1.0, raw_confidence))

        factor = self.get_calibration_factor(agent_type, task_complexity)

        calibrated = raw_confidence * factor

        # Clamp to valid range
        calibrated = max(0.0, min(1.0, calibrated))

        if abs(factor - 1.0) > 0.05:  # Only log if factor is significant
            logger.debug(f"Calibrated {agent_type} confidence: {raw_confidence:.3f} → {calibrated:.3f} "
                        f"(factor={factor:.3f})")

        return calibrated

    def get_calibration_statistics(self,
                                    agent_type: Optional[str] = None,
                                    days: int = 30) -> Dict[str, Any]:
        """
        Get calibration statistics for analysis.

        Args:
            agent_type: Optional agent type to filter by (default: all types)
            days: Number of days to include (default: 30)

        Returns:
            Dictionary with calibration statistics
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cutoff_time = time.time() - (days * 24 * 60 * 60)

            if agent_type:
                query = """
                    SELECT
                        agent_type,
                        task_complexity,
                        calibration_factor,
                        avg_predicted_confidence,
                        avg_actual_success,
                        calibration_error,
                        sample_size
                    FROM confidence_calibration
                    WHERE agent_type = ? AND last_updated > ?
                    ORDER BY sample_size DESC
                """
                params = (agent_type, cutoff_time)
            else:
                query = """
                    SELECT
                        agent_type,
                        task_complexity,
                        calibration_factor,
                        avg_predicted_confidence,
                        avg_actual_success,
                        calibration_error,
                        sample_size
                    FROM confidence_calibration
                    WHERE last_updated > ?
                    ORDER BY sample_size DESC
                """
                params = (cutoff_time,)

            cursor = conn.execute(query, params)
            records = [dict(row) for row in cursor.fetchall()]

            conn.close()

            # Calculate aggregate statistics
            if records:
                total_samples = sum(r['sample_size'] for r in records)
                avg_calibration_error = statistics.mean(r['calibration_error'] for r in records)

                # Identify most overconfident and underconfident agents
                most_overconfident = min(records, key=lambda r: r['calibration_factor'])
                most_underconfident = max(records, key=lambda r: r['calibration_factor'])
            else:
                total_samples = 0
                avg_calibration_error = 0.0
                most_overconfident = None
                most_underconfident = None

            return {
                'total_records': len(records),
                'total_samples': total_samples,
                'avg_calibration_error': avg_calibration_error,
                'most_overconfident': most_overconfident,
                'most_underconfident': most_underconfident,
                'records': records,
                'days': days
            }

        except sqlite3.Error as e:
            logger.error(f"Failed to get calibration statistics: {e}")
            return {}

    def get_all_calibration_records(self) -> list[CalibrationRecord]:
        """
        Get all calibration records.

        Returns:
            List of CalibrationRecord objects
        """
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT * FROM confidence_calibration
                ORDER BY last_updated DESC
            """)

            records = []
            for row in cursor.fetchall():
                record = CalibrationRecord(
                    calibration_id=row['calibration_id'],
                    agent_type=row['agent_type'],
                    task_complexity=row['task_complexity'],
                    calibration_factor=row['calibration_factor'],
                    avg_predicted_confidence=row['avg_predicted_confidence'],
                    avg_actual_success=row['avg_actual_success'],
                    calibration_error=row['calibration_error'],
                    sample_size=row['sample_size'],
                    last_updated=row['last_updated']
                )
                records.append(record)

            conn.close()
            return records

        except sqlite3.Error as e:
            logger.error(f"Failed to get calibration records: {e}")
            return []

    @retry_on_locked(max_attempts=3)
    def reset_calibration(self, agent_type: Optional[str] = None) -> int:
        """
        Reset calibration data (useful for retraining).

        Args:
            agent_type: Optional agent type to reset (default: reset all)

        Returns:
            Number of records deleted
        """
        try:
            conn = self._get_connection()

            if agent_type:
                cursor = conn.execute("""
                    DELETE FROM confidence_calibration
                    WHERE agent_type = ?
                """, (agent_type,))
            else:
                cursor = conn.execute("DELETE FROM confidence_calibration")

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            # Clear cache
            self._calibration_cache.clear()
            self._cache_timestamp.clear()

            logger.info(f"Reset {deleted_count} calibration records" +
                       (f" for {agent_type}" if agent_type else ""))

            return deleted_count

        except sqlite3.Error as e:
            logger.error(f"Failed to reset calibration: {e}")
            return 0
