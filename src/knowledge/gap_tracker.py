"""
Knowledge Gap Tracker for the Felix Framework.

Tracks knowledge gaps discovered during workflows and correlates them
with workflow outcomes. This enables gap-directed learning by identifying
high-priority gaps that consistently hurt synthesis quality.

Key Features:
- Record gaps discovered during coverage analysis
- Correlate gaps with workflow outcomes (confidence, success)
- Identify priority gaps for acquisition
- Track gap resolution
"""

import time
import uuid
import logging
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Gap:
    """Represents a knowledge gap."""
    gap_id: str
    domain: str
    concept: Optional[str]
    first_seen: float
    last_seen: float
    occurrence_count: int
    impact_severity_avg: float
    resolved: bool
    resolution_method: Optional[str] = None
    resolution_timestamp: Optional[float] = None


class GapTracker:
    """
    Tracks and manages knowledge gaps.

    Enables epistemic self-awareness by monitoring which gaps hurt
    workflow quality and prioritizing them for acquisition.
    """

    def __init__(self, db_path: str = "felix_knowledge.db"):
        """
        Initialize Gap Tracker.

        Args:
            db_path: Path to knowledge database
        """
        self.db_path = db_path
        self._ensure_tables_exist()
        logger.info("GapTracker initialized")

    def _ensure_tables_exist(self):
        """Create tables if they don't exist (run migration)."""
        try:
            from src.migration.add_knowledge_gaps_table import migrate_up
            migrate_up(self.db_path)
        except Exception as e:
            logger.warning(f"Could not run gap tables migration: {e}")

    def record_gap(self, domain: str, concept: Optional[str] = None,
                  workflow_id: Optional[str] = None,
                  severity: float = 0.5) -> str:
        """
        Record a knowledge gap discovered during coverage analysis.

        Args:
            domain: Domain where gap was identified
            concept: Specific concept lacking (optional)
            workflow_id: Workflow that discovered the gap
            severity: Impact severity (0.0-1.0)

        Returns:
            Gap ID (new or existing)
        """
        current_time = time.time()

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if gap already exists
                cursor = conn.execute("""
                    SELECT gap_id, occurrence_count, impact_severity_avg
                    FROM knowledge_gaps
                    WHERE domain = ? AND (concept = ? OR (concept IS NULL AND ? IS NULL))
                    AND resolved = FALSE
                """, (domain, concept, concept))

                row = cursor.fetchone()

                if row:
                    # Update existing gap
                    gap_id = row[0]
                    occurrence_count = row[1] + 1
                    # Running average of severity
                    old_severity = row[2]
                    new_severity = (old_severity * (occurrence_count - 1) + severity) / occurrence_count

                    conn.execute("""
                        UPDATE knowledge_gaps
                        SET last_seen = ?,
                            occurrence_count = ?,
                            impact_severity_avg = ?
                        WHERE gap_id = ?
                    """, (current_time, occurrence_count, new_severity, gap_id))

                    logger.debug(f"Updated existing gap {gap_id}: occurrences={occurrence_count}, severity={new_severity:.2f}")
                else:
                    # Create new gap
                    gap_id = f"gap_{uuid.uuid4().hex[:12]}"

                    conn.execute("""
                        INSERT INTO knowledge_gaps
                        (gap_id, domain, concept, first_seen, last_seen,
                         occurrence_count, impact_severity_avg, resolved)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (gap_id, domain, concept, current_time, current_time,
                          1, severity, False))

                    logger.info(f"Recorded new gap {gap_id}: domain={domain}, concept={concept}")

                # Record workflow correlation if workflow_id provided
                if workflow_id:
                    conn.execute("""
                        INSERT INTO gap_workflow_correlation
                        (gap_id, workflow_id, detected_at, impact_severity)
                        VALUES (?, ?, ?, ?)
                    """, (gap_id, workflow_id, current_time, severity))

                conn.commit()
                return gap_id

        except sqlite3.Error as e:
            logger.error(f"Failed to record gap: {e}")
            return ""

    def correlate_with_outcome(self, workflow_id: str,
                               completion_status: str,
                               confidence: float) -> None:
        """
        Update gap correlations with workflow outcome.

        After a workflow completes, update the gap-workflow correlations
        with the outcome to better understand gap impact.

        Args:
            workflow_id: Workflow that completed
            completion_status: 'success', 'partial', or 'failed'
            confidence: Final workflow confidence
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update workflow correlations with confidence
                conn.execute("""
                    UPDATE gap_workflow_correlation
                    SET workflow_confidence = ?
                    WHERE workflow_id = ?
                """, (confidence, workflow_id))

                # Adjust gap severity based on outcome
                # Low confidence = gap had high impact
                impact_adjustment = 1.0 - confidence  # 0.2 confidence = 0.8 impact

                cursor = conn.execute("""
                    SELECT DISTINCT gap_id FROM gap_workflow_correlation
                    WHERE workflow_id = ?
                """, (workflow_id,))

                gap_ids = [row[0] for row in cursor.fetchall()]

                for gap_id in gap_ids:
                    # Update severity using exponential moving average
                    cursor = conn.execute("""
                        SELECT impact_severity_avg FROM knowledge_gaps
                        WHERE gap_id = ?
                    """, (gap_id,))

                    row = cursor.fetchone()
                    if row:
                        old_severity = row[0]
                        alpha = 0.3  # Learning rate
                        new_severity = (1 - alpha) * old_severity + alpha * impact_adjustment
                        new_severity = max(0.0, min(1.0, new_severity))

                        conn.execute("""
                            UPDATE knowledge_gaps
                            SET impact_severity_avg = ?
                            WHERE gap_id = ?
                        """, (new_severity, gap_id))

                        logger.debug(f"Gap {gap_id} severity adjusted: {old_severity:.2f} → {new_severity:.2f} "
                                   f"(workflow confidence={confidence:.2f})")

                conn.commit()

        except sqlite3.Error as e:
            logger.error(f"Failed to correlate gap with outcome: {e}")

    def get_priority_gaps(self, min_severity: float = 0.5,
                         min_occurrences: int = 2,
                         limit: int = 10) -> List[Gap]:
        """
        Get high-priority gaps for acquisition.

        Returns gaps that:
        - Have high impact severity
        - Occur frequently
        - Are not yet resolved

        Args:
            min_severity: Minimum average severity to include
            min_occurrences: Minimum occurrence count
            limit: Maximum number of gaps to return

        Returns:
            List of Gap objects sorted by priority (severity * occurrences)
        """
        gaps = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT gap_id, domain, concept, first_seen, last_seen,
                           occurrence_count, impact_severity_avg, resolved,
                           resolution_method, resolution_timestamp
                    FROM knowledge_gaps
                    WHERE resolved = FALSE
                    AND impact_severity_avg >= ?
                    AND occurrence_count >= ?
                    ORDER BY (impact_severity_avg * occurrence_count) DESC
                    LIMIT ?
                """, (min_severity, min_occurrences, limit))

                for row in cursor.fetchall():
                    gaps.append(Gap(
                        gap_id=row[0],
                        domain=row[1],
                        concept=row[2],
                        first_seen=row[3],
                        last_seen=row[4],
                        occurrence_count=row[5],
                        impact_severity_avg=row[6],
                        resolved=bool(row[7]),
                        resolution_method=row[8],
                        resolution_timestamp=row[9]
                    ))

        except sqlite3.Error as e:
            logger.error(f"Failed to get priority gaps: {e}")

        return gaps

    def mark_gap_resolved(self, gap_id: str, method: str) -> bool:
        """
        Mark a gap as resolved.

        Args:
            gap_id: Gap to mark resolved
            method: Resolution method (web_search, manual, acquired, etc.)

        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE knowledge_gaps
                    SET resolved = TRUE,
                        resolution_method = ?,
                        resolution_timestamp = ?
                    WHERE gap_id = ?
                """, (method, time.time(), gap_id))
                conn.commit()

                logger.info(f"Gap {gap_id} marked as resolved via {method}")
                return True

        except sqlite3.Error as e:
            logger.error(f"Failed to mark gap resolved: {e}")
            return False

    def get_gaps_for_display(self, limit: int = 100,
                            include_resolved: bool = False) -> List[Dict[str, Any]]:
        """
        Get gaps formatted for GUI display.

        Args:
            limit: Maximum number of gaps
            include_resolved: Whether to include resolved gaps

        Returns:
            List of gap dictionaries with display-friendly format
        """
        gaps = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                resolved_filter = "" if include_resolved else "AND resolved = FALSE"

                cursor = conn.execute(f"""
                    SELECT gap_id, domain, concept, first_seen, last_seen,
                           occurrence_count, impact_severity_avg, resolved,
                           resolution_method, resolution_timestamp
                    FROM knowledge_gaps
                    WHERE 1=1 {resolved_filter}
                    ORDER BY last_seen DESC
                    LIMIT ?
                """, (limit,))

                for row in cursor.fetchall():
                    gaps.append({
                        'gap_id': row[0],
                        'domain': row[1],
                        'concept': row[2] or "(general)",
                        'first_seen': row[3],
                        'last_seen': row[4],
                        'occurrence_count': row[5],
                        'severity': row[6],
                        'resolved': bool(row[7]),
                        'resolution_method': row[8],
                        'status': 'Resolved' if row[7] else 'Active'
                    })

        except sqlite3.Error as e:
            logger.error(f"Failed to get gaps for display: {e}")

        return gaps

    def get_gap_workflow_history(self, gap_id: str) -> List[Dict[str, Any]]:
        """
        Get workflow history for a specific gap.

        Args:
            gap_id: Gap to get history for

        Returns:
            List of workflow correlations
        """
        history = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT workflow_id, detected_at, workflow_confidence, impact_severity
                    FROM gap_workflow_correlation
                    WHERE gap_id = ?
                    ORDER BY detected_at DESC
                """, (gap_id,))

                for row in cursor.fetchall():
                    history.append({
                        'workflow_id': row[0],
                        'detected_at': row[1],
                        'workflow_confidence': row[2],
                        'impact_severity': row[3]
                    })

        except sqlite3.Error as e:
            logger.error(f"Failed to get gap workflow history: {e}")

        return history
