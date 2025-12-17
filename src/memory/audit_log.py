"""
Audit Logging System for Knowledge Brain

Provides comprehensive audit trail for all CRUD operations on knowledge entries.
Tracks who did what, when, and captures before/after state for accountability
and debugging.

Features:
- Decorator-based automatic logging
- Transaction-level grouping
- Before/after state capture
- Query and export capabilities
- Automatic cleanup of old logs
"""

import sqlite3
import json
import logging
import functools
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    Audit logging system for knowledge base operations.

    Records all CRUD operations with full context and state tracking.
    Designed for integration with KnowledgeStore via decorator pattern.
    """

    def __init__(self, db_path: str = "felix_knowledge.db"):
        """
        Initialize audit logger.

        Args:
            db_path: Path to knowledge database containing audit_log table
        """
        self.db_path = db_path
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """
        Verify audit log table exists. If not, run migration.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_audit_log'
            """)

            if not cursor.fetchone():
                logger.warning(
                    "Audit log table not found. Run migration: "
                    "python3 src/migration/add_audit_log_table.py"
                )

            conn.close()

        except Exception as e:
            logger.error(f"Failed to verify audit table: {e}")

    def log_operation(
        self,
        operation: str,
        knowledge_id: Optional[str] = None,
        user_agent: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        details: Optional[str] = None,
        transaction_id: Optional[str] = None
    ) -> bool:
        """
        Log a knowledge base operation to audit trail.

        Args:
            operation: Type of operation (INSERT, UPDATE, DELETE, MERGE, CLEANUP)
            knowledge_id: ID of knowledge entry affected
            user_agent: Who performed operation (agent name, "GUI User", etc.)
            old_values: Previous state (for UPDATE/DELETE)
            new_values: New state (for INSERT/UPDATE)
            details: Human-readable description
            transaction_id: Group related operations (e.g., merge operations)

        Returns:
            True if logged successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Serialize state dictionaries
            old_json = json.dumps(old_values) if old_values else None
            new_json = json.dumps(new_values) if new_values else None

            cursor.execute("""
                INSERT INTO knowledge_audit_log (
                    timestamp, operation, knowledge_id, user_agent,
                    old_values_json, new_values_json, details, transaction_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().timestamp(),
                operation,
                knowledge_id,
                user_agent,
                old_json,
                new_json,
                details,
                transaction_id
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
            if 'conn' in locals():
                conn.close()
            return False

    def get_audit_history(
        self,
        knowledge_id: Optional[str] = None,
        operation: Optional[str] = None,
        user_agent: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 500,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query audit history with optional filters.

        Args:
            knowledge_id: Filter by knowledge entry ID
            operation: Filter by operation type
            user_agent: Filter by user/agent
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            limit: Maximum records to return
            offset: Number of records to skip (for pagination)

        Returns:
            List of audit log entries as dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build query with filters
            query = "SELECT * FROM knowledge_audit_log WHERE 1=1"
            params = []

            if knowledge_id:
                query += " AND knowledge_id = ?"
                params.append(knowledge_id)

            if operation:
                query += " AND operation = ?"
                params.append(operation)

            if user_agent:
                query += " AND user_agent = ?"
                params.append(user_agent)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.timestamp())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.timestamp())

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to dictionaries
            entries = []
            for row in rows:
                entry = {
                    'audit_id': row['audit_id'],
                    'timestamp': datetime.fromtimestamp(row['timestamp']),
                    'operation': row['operation'],
                    'knowledge_id': row['knowledge_id'],
                    'user_agent': row['user_agent'],
                    'old_values': json.loads(row['old_values_json']) if row['old_values_json'] else None,
                    'new_values': json.loads(row['new_values_json']) if row['new_values_json'] else None,
                    'details': row['details'],
                    'transaction_id': row['transaction_id']
                }
                entries.append(entry)

            conn.close()
            return entries

        except Exception as e:
            logger.error(f"Failed to query audit history: {e}")
            if 'conn' in locals():
                conn.close()
            return []

    def get_entry_history(self, knowledge_id: str) -> List[Dict[str, Any]]:
        """
        Get complete audit history for a specific knowledge entry.

        Args:
            knowledge_id: Knowledge entry ID

        Returns:
            List of audit entries for this knowledge_id, newest first
        """
        return self.get_audit_history(knowledge_id=knowledge_id, limit=1000)

    def get_recent_changes(self, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent changes in last N hours.

        Args:
            hours: Number of hours to look back
            limit: Maximum records to return

        Returns:
            List of recent audit entries
        """
        start_date = datetime.now() - timedelta(hours=hours)
        return self.get_audit_history(start_date=start_date, limit=limit)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Returns:
            Dictionary with statistics:
            - total_entries: Total audit log entries
            - by_operation: Breakdown by operation type
            - by_user_agent: Breakdown by user/agent
            - date_range: Oldest and newest timestamps
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Total entries
            cursor.execute("SELECT COUNT(*) FROM knowledge_audit_log")
            total = cursor.fetchone()[0]

            # By operation
            cursor.execute("""
                SELECT operation, COUNT(*) as count
                FROM knowledge_audit_log
                GROUP BY operation
                ORDER BY count DESC
            """)
            by_operation = {row[0]: row[1] for row in cursor.fetchall()}

            # By user/agent
            cursor.execute("""
                SELECT user_agent, COUNT(*) as count
                FROM knowledge_audit_log
                WHERE user_agent IS NOT NULL
                GROUP BY user_agent
                ORDER BY count DESC
                LIMIT 10
            """)
            by_user_agent = {row[0]: row[1] for row in cursor.fetchall()}

            # Date range
            cursor.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM knowledge_audit_log
            """)
            min_ts, max_ts = cursor.fetchone()
            date_range = None
            if min_ts and max_ts:
                date_range = {
                    'oldest': datetime.fromtimestamp(min_ts),
                    'newest': datetime.fromtimestamp(max_ts)
                }

            conn.close()

            return {
                'total_entries': total,
                'by_operation': by_operation,
                'by_user_agent': by_user_agent,
                'date_range': date_range
            }

        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            if 'conn' in locals():
                conn.close()
            return {
                'total_entries': 0,
                'by_operation': {},
                'by_user_agent': {},
                'date_range': None
            }

    def cleanup_old_logs(self, max_age_days: int = 90) -> int:
        """
        Remove audit logs older than specified age.

        Args:
            max_age_days: Delete logs older than this many days

        Returns:
            Number of entries deleted
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            cutoff_timestamp = cutoff_date.timestamp()

            cursor.execute("""
                DELETE FROM knowledge_audit_log
                WHERE timestamp < ?
            """, (cutoff_timestamp,))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Deleted {deleted} audit log entries older than {max_age_days} days")
            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old audit logs: {e}")
            if 'conn' in locals():
                conn.close()
            return 0

    def export_to_csv(self, output_path: str, **filter_kwargs) -> bool:
        """
        Export audit log to CSV file.

        Args:
            output_path: Path to output CSV file
            **filter_kwargs: Filters to pass to get_audit_history()

        Returns:
            True if export successful, False otherwise
        """
        try:
            import csv

            entries = self.get_audit_history(**filter_kwargs, limit=100000)

            if not entries:
                logger.warning("No audit entries to export")
                return False

            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'audit_id', 'timestamp', 'operation', 'knowledge_id',
                    'user_agent', 'details', 'transaction_id'
                ])
                writer.writeheader()

                for entry in entries:
                    # Simplify for CSV (exclude JSON fields)
                    row = {
                        'audit_id': entry['audit_id'],
                        'timestamp': entry['timestamp'].isoformat(),
                        'operation': entry['operation'],
                        'knowledge_id': entry['knowledge_id'],
                        'user_agent': entry['user_agent'],
                        'details': entry['details'],
                        'transaction_id': entry['transaction_id']
                    }
                    writer.writerow(row)

            logger.info(f"Exported {len(entries)} audit entries to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export audit log: {e}")
            return False

    # ==================================================================
    # Synthesis Audit Operations (extended for synthesis transparency)
    # ==================================================================

    def log_synthesis_operation(
        self,
        workflow_id: Optional[str] = None,
        task_description: str = "",
        task_complexity: str = "COMPLEX",
        agent_count: int = 0,
        synthesis_confidence: float = 0.0,
        validation_score: Optional[float] = None,
        validation_flags: Optional[List[str]] = None,
        used_fallback: bool = False,
        degraded: bool = False,
        degraded_reasons: Optional[List[str]] = None,
        user_approved: bool = True,
        regeneration_requested: bool = False,
        regeneration_strategy: Optional[str] = None
    ) -> bool:
        """
        Log a synthesis operation to the synthesis_audit table.

        This provides a simpler interface than _log_synthesis_audit() in
        SynthesisEngine, for cases where you don't have all the detailed
        context (prompts, raw outputs) but want to log a synthesis event.

        Args:
            workflow_id: ID of the workflow
            task_description: Original task
            task_complexity: SIMPLE_FACTUAL, MEDIUM, or COMPLEX
            agent_count: Number of agents involved
            synthesis_confidence: Final confidence score
            validation_score: Validation score (0.0-1.0)
            validation_flags: List of validation issues
            used_fallback: Whether fallback synthesis was used
            degraded: Whether synthesis was degraded
            degraded_reasons: List of degradation reasons
            user_approved: Whether user approved the synthesis
            regeneration_requested: Whether regeneration was requested
            regeneration_strategy: Strategy requested for regeneration

        Returns:
            True if logged successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if synthesis_audit table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='synthesis_audit'
            """)
            if not cursor.fetchone():
                logger.debug("synthesis_audit table not found, skipping synthesis audit log")
                conn.close()
                return False

            cursor.execute("""
                INSERT INTO synthesis_audit (
                    workflow_id, timestamp, task_description, task_complexity,
                    agent_count, synthesis_confidence,
                    validation_called, validation_score, validation_flags_json,
                    used_fallback, degraded, degraded_reasons_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_id,
                datetime.now().timestamp(),
                task_description,
                task_complexity,
                agent_count,
                synthesis_confidence,
                1 if validation_score is not None else 0,
                validation_score,
                json.dumps(validation_flags) if validation_flags else None,
                1 if used_fallback else 0,
                1 if degraded else 0,
                json.dumps(degraded_reasons) if degraded_reasons else None
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to log synthesis operation: {e}")
            if 'conn' in locals():
                conn.close()
            return False

    def get_synthesis_history(
        self,
        workflow_id: Optional[str] = None,
        task_complexity: Optional[str] = None,
        degraded_only: bool = False,
        low_confidence_only: bool = False,
        confidence_threshold: float = 0.5,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query synthesis audit history with optional filters.

        Args:
            workflow_id: Filter by workflow ID
            task_complexity: Filter by complexity (SIMPLE_FACTUAL, MEDIUM, COMPLEX)
            degraded_only: Only return degraded syntheses
            low_confidence_only: Only return low-confidence syntheses
            confidence_threshold: Threshold for low_confidence_only filter
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            limit: Maximum records to return
            offset: Number of records to skip (for pagination)

        Returns:
            List of synthesis audit entries as dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Check if synthesis_audit table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='synthesis_audit'
            """)
            if not cursor.fetchone():
                logger.warning("synthesis_audit table not found")
                conn.close()
                return []

            # Build query with filters
            query = "SELECT * FROM synthesis_audit WHERE 1=1"
            params = []

            if workflow_id:
                query += " AND workflow_id = ?"
                params.append(workflow_id)

            if task_complexity:
                query += " AND task_complexity = ?"
                params.append(task_complexity)

            if degraded_only:
                query += " AND degraded = 1"

            if low_confidence_only:
                query += " AND synthesis_confidence < ?"
                params.append(confidence_threshold)

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.timestamp())

            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.timestamp())

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert to dictionaries
            entries = []
            for row in rows:
                entry = {
                    'audit_id': row['audit_id'],
                    'workflow_id': row['workflow_id'],
                    'timestamp': datetime.fromtimestamp(row['timestamp']),
                    'task_description': row['task_description'],
                    'task_complexity': row['task_complexity'],
                    'agent_count': row['agent_count'],
                    'synthesis_confidence': row['synthesis_confidence'],
                    'validation_called': bool(row['validation_called']),
                    'validation_score': row['validation_score'],
                    'validation_flags': json.loads(row['validation_flags_json']) if row['validation_flags_json'] else None,
                    'used_fallback': bool(row['used_fallback']),
                    'degraded': bool(row['degraded']),
                    'degraded_reasons': json.loads(row['degraded_reasons_json']) if row['degraded_reasons_json'] else None,
                    'tokens_used': row['tokens_used'],
                    'synthesis_time': row['synthesis_time'],
                }
                entries.append(entry)

            conn.close()
            return entries

        except Exception as e:
            logger.error(f"Failed to query synthesis history: {e}")
            if 'conn' in locals():
                conn.close()
            return []

    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """
        Get synthesis audit statistics.

        Returns:
            Dictionary with statistics:
            - total_syntheses: Total number of syntheses
            - avg_confidence: Average synthesis confidence
            - degraded_count: Number of degraded syntheses
            - fallback_count: Number of fallback syntheses
            - by_complexity: Breakdown by task complexity
            - confidence_distribution: Distribution of confidence scores
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if synthesis_audit table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='synthesis_audit'
            """)
            if not cursor.fetchone():
                logger.warning("synthesis_audit table not found")
                conn.close()
                return {
                    'total_syntheses': 0,
                    'avg_confidence': 0.0,
                    'degraded_count': 0,
                    'fallback_count': 0,
                    'by_complexity': {},
                    'confidence_distribution': {}
                }

            # Total and averages
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(synthesis_confidence) as avg_conf,
                    SUM(CASE WHEN degraded = 1 THEN 1 ELSE 0 END) as degraded,
                    SUM(CASE WHEN used_fallback = 1 THEN 1 ELSE 0 END) as fallback
                FROM synthesis_audit
                WHERE task_description != 'SYSTEM_MIGRATION'
            """)
            row = cursor.fetchone()
            total = row[0] or 0
            avg_conf = row[1] or 0.0
            degraded = row[2] or 0
            fallback = row[3] or 0

            # By complexity
            cursor.execute("""
                SELECT task_complexity, COUNT(*) as count
                FROM synthesis_audit
                WHERE task_description != 'SYSTEM_MIGRATION'
                GROUP BY task_complexity
            """)
            by_complexity = {r[0]: r[1] for r in cursor.fetchall()}

            # Confidence distribution (buckets)
            cursor.execute("""
                SELECT
                    CASE
                        WHEN synthesis_confidence < 0.3 THEN 'very_low'
                        WHEN synthesis_confidence < 0.5 THEN 'low'
                        WHEN synthesis_confidence < 0.7 THEN 'moderate'
                        WHEN synthesis_confidence < 0.9 THEN 'good'
                        ELSE 'excellent'
                    END as bucket,
                    COUNT(*) as count
                FROM synthesis_audit
                WHERE task_description != 'SYSTEM_MIGRATION'
                GROUP BY bucket
            """)
            confidence_dist = {r[0]: r[1] for r in cursor.fetchall()}

            conn.close()

            return {
                'total_syntheses': total,
                'avg_confidence': round(avg_conf, 3),
                'degraded_count': degraded,
                'degraded_rate': round(degraded / total, 3) if total > 0 else 0.0,
                'fallback_count': fallback,
                'fallback_rate': round(fallback / total, 3) if total > 0 else 0.0,
                'by_complexity': by_complexity,
                'confidence_distribution': confidence_dist
            }

        except Exception as e:
            logger.error(f"Failed to get synthesis statistics: {e}")
            if 'conn' in locals():
                conn.close()
            return {
                'total_syntheses': 0,
                'avg_confidence': 0.0,
                'degraded_count': 0,
                'fallback_count': 0,
                'by_complexity': {},
                'confidence_distribution': {}
            }

    def get_recent_low_confidence_syntheses(
        self,
        hours: int = 24,
        threshold: float = 0.5,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get recent low-confidence syntheses for review.

        Useful for identifying patterns in synthesis quality issues.

        Args:
            hours: Number of hours to look back
            threshold: Confidence threshold (return syntheses below this)
            limit: Maximum records to return

        Returns:
            List of low-confidence synthesis entries
        """
        start_date = datetime.now() - timedelta(hours=hours)
        return self.get_synthesis_history(
            low_confidence_only=True,
            confidence_threshold=threshold,
            start_date=start_date,
            limit=limit
        )


def audit_logged(operation: str, user_agent: str = "KnowledgeStore"):
    """
    Decorator for automatic audit logging of CRUD operations.

    Usage:
        @audit_logged("INSERT", "ResearchAgent")
        def store_knowledge(self, entry):
            # ... implementation ...
            return knowledge_id

    Args:
        operation: Operation type (INSERT, UPDATE, DELETE, MERGE, CLEANUP)
        user_agent: Who is performing operation (default: "KnowledgeStore")

    Returns:
        Decorator function that wraps method with audit logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate transaction ID for grouping related operations
            transaction_id = str(uuid.uuid4())

            # Capture old state for UPDATE/DELETE operations
            old_values = None
            if operation in ("UPDATE", "DELETE"):
                # Try to get knowledge_id from args/kwargs
                knowledge_id = None
                if args and isinstance(args[0], str):
                    knowledge_id = args[0]
                elif 'knowledge_id' in kwargs:
                    knowledge_id = kwargs['knowledge_id']

                # Fetch old state if we have knowledge_id
                if knowledge_id and hasattr(self, 'get_entry_by_id'):
                    try:
                        old_entry = self.get_entry_by_id(knowledge_id)
                        if old_entry:
                            old_values = {
                                'knowledge_id': old_entry.knowledge_id,
                                'knowledge_type': old_entry.knowledge_type.value,
                                'content': old_entry.content,
                                'confidence_level': old_entry.confidence_level.value,
                                'domain': old_entry.domain
                            }
                    except Exception as e:
                        logger.debug(f"Could not capture old state: {e}")

            # Execute the wrapped function
            result = func(self, *args, **kwargs)

            # Log the operation (async to not block)
            try:
                audit_logger = AuditLogger(getattr(self, 'db_path', 'felix_knowledge.db'))

                # Extract knowledge_id from result or args
                knowledge_id = None
                if isinstance(result, str):
                    knowledge_id = result
                elif args and isinstance(args[0], str):
                    knowledge_id = args[0]
                elif 'knowledge_id' in kwargs:
                    knowledge_id = kwargs['knowledge_id']

                # Build details string
                func_name = func.__name__
                details = f"{func_name}() called"
                if args:
                    details += f" with {len(args)} args"
                if kwargs:
                    details += f" and {len(kwargs)} kwargs"

                # Capture new state for INSERT/UPDATE operations
                new_values = None
                if operation in ("INSERT", "UPDATE") and knowledge_id:
                    if hasattr(self, 'get_entry_by_id'):
                        try:
                            new_entry = self.get_entry_by_id(knowledge_id)
                            if new_entry:
                                new_values = {
                                    'knowledge_id': new_entry.knowledge_id,
                                    'knowledge_type': new_entry.knowledge_type.value,
                                    'content': new_entry.content,
                                    'confidence_level': new_entry.confidence_level.value,
                                    'domain': new_entry.domain
                                }
                        except Exception as e:
                            logger.debug(f"Could not capture new state: {e}")

                # Log to audit trail
                audit_logger.log_operation(
                    operation=operation,
                    knowledge_id=knowledge_id,
                    user_agent=user_agent,
                    old_values=old_values,
                    new_values=new_values,
                    details=details,
                    transaction_id=transaction_id
                )

            except Exception as e:
                logger.warning(f"Failed to log audit entry: {e}")
                # Don't fail the operation if logging fails

            return result

        return wrapper
    return decorator


# Convenience singleton instance
_default_audit_logger = None


def get_audit_logger(db_path: str = "felix_knowledge.db") -> AuditLogger:
    """
    Get singleton audit logger instance.

    Args:
        db_path: Path to knowledge database

    Returns:
        AuditLogger instance
    """
    global _default_audit_logger
    if _default_audit_logger is None:
        _default_audit_logger = AuditLogger(db_path)
    return _default_audit_logger
