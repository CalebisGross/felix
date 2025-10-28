"""
Command History Database Wrapper.

Provides interface to felix_system_actions.db for:
- Recording command executions
- Querying command history
- Learning from patterns
- Pattern management
"""

import sqlite3
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from .system_executor import CommandResult
from .trust_manager import TrustLevel

logger = logging.getLogger(__name__)


class CommandHistory:
    """
    Wrapper for command history database operations.

    Interfaces with felix_system_actions.db to track:
    - Command executions with full context
    - Success/failure patterns
    - Command patterns (learned sequences)
    - Approval history
    """

    def __init__(self, db_path: str = "felix_system_actions.db"):
        """
        Initialize command history.

        Args:
            db_path: Path to system actions database
        """
        self.db_path = Path(db_path)

        if not self.db_path.exists():
            logger.warning(f"System actions database not found: {db_path}")
            logger.info("Run migrations to create database")

        logger.info(f"CommandHistory initialized: {db_path}")

    def record_execution(self,
                        command: str,
                        command_hash: str,
                        result: CommandResult,
                        agent_id: str,
                        agent_type: Optional[str] = None,
                        workflow_id: Optional[int] = None,
                        trust_level: Optional[TrustLevel] = None,
                        approved_by: Optional[str] = None,
                        context: str = "",
                        status: str = "completed") -> int:
        """
        Record a command execution in the database.

        Args:
            command: Command that was executed
            command_hash: Hash of command for deduplication
            result: CommandResult from execution
            agent_id: ID of agent that requested execution
            agent_type: Type of agent
            workflow_id: Associated workflow ID
            trust_level: Trust level of command
            approved_by: Who approved (if applicable)
            context: Context/reason for execution
            status: Execution status ('pending', 'running', 'completed', 'failed')

        Returns:
            execution_id of inserted record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO command_executions (
                    workflow_id, agent_id, agent_type, command, command_hash,
                    trust_level, approved_by, approval_timestamp,
                    executed, execution_timestamp, exit_code, duration,
                    stdout_preview, stderr_preview, output_size,
                    context, cwd, env_snapshot, venv_active,
                    success, error_category, timestamp, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_id,
                agent_id,
                agent_type,
                command,
                command_hash,
                trust_level.value if trust_level else "unknown",
                approved_by,
                result.timestamp if approved_by else None,
                True,  # executed
                result.timestamp,
                result.exit_code,
                result.duration,
                result.stdout[:1000] if result.stdout else "",  # Preview
                result.stderr[:1000] if result.stderr else "",
                result.output_size,
                context,
                result.cwd,
                None,  # env_snapshot (can add later)
                result.venv_active,
                result.success,
                result.error_category.value if result.error_category else None,
                time.time(),
                status
            ))

            execution_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Recorded execution: {execution_id} (status={status})")

            return execution_id

    def create_execution_placeholder(self,
                                    command: str,
                                    command_hash: str,
                                    agent_id: str,
                                    agent_type: Optional[str] = None,
                                    workflow_id: Optional[int] = None,
                                    trust_level: Optional[TrustLevel] = None,
                                    approved_by: Optional[str] = None,
                                    context: str = "") -> int:
        """
        Create a placeholder execution record with status='running'.

        Used to get an execution_id before command starts, enabling real-time
        status tracking in the Terminal tab.

        Args:
            command: Command to execute
            command_hash: Hash of command
            agent_id: Requesting agent ID
            agent_type: Type of agent
            workflow_id: Associated workflow ID
            trust_level: Trust level of command
            approved_by: Who approved (if applicable)
            context: Command context

        Returns:
            execution_id of placeholder record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO command_executions (
                    workflow_id, agent_id, agent_type, command, command_hash,
                    trust_level, approved_by, context, executed,
                    execution_timestamp, timestamp, status,
                    exit_code, duration, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                workflow_id,
                agent_id,
                agent_type,
                command,
                command_hash,
                trust_level.value if trust_level else "unknown",
                approved_by,
                context,
                False,  # Not yet executed
                time.time(),
                time.time(),
                'running',
                None,  # Will be updated
                None,  # Will be updated
                None   # Will be updated
            ))

            execution_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Created execution placeholder: {execution_id}")

            return execution_id

    def update_execution_result(self,
                               execution_id: int,
                               result: CommandResult) -> bool:
        """
        Update an execution record with final result.

        Used to complete a placeholder record after command finishes.

        Args:
            execution_id: Execution ID to update
            result: Final CommandResult

        Returns:
            True if updated successfully, False otherwise
        """
        status = 'completed' if result.success else 'failed'

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE command_executions
                SET executed = ?,
                    exit_code = ?,
                    duration = ?,
                    stdout_preview = ?,
                    stderr_preview = ?,
                    output_size = ?,
                    cwd = ?,
                    venv_active = ?,
                    success = ?,
                    error_category = ?,
                    status = ?
                WHERE execution_id = ?
            """, (
                True,
                result.exit_code,
                result.duration,
                result.stdout[:1000] if result.stdout else "",
                result.stderr[:1000] if result.stderr else "",
                result.output_size,
                result.cwd,
                result.venv_active,
                result.success,
                result.error_category.value if result.error_category else None,
                status,
                execution_id
            ))

            conn.commit()

            logger.info(f"Updated execution {execution_id}: {status}")

            return True

    def get_command_stats(self, command: str) -> Dict[str, Any]:
        """
        Get statistics for a specific command.

        Args:
            command: Command to get stats for

        Returns:
            Dictionary with execution statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get overall stats
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_executions,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    AVG(duration) as avg_duration,
                    MIN(duration) as min_duration,
                    MAX(duration) as max_duration,
                    MAX(timestamp) as last_execution
                FROM command_executions
                WHERE command = ?
            """, (command,))

            row = cursor.fetchone()

            if row and row['total_executions'] > 0:
                success_rate = (row['successful'] / row['total_executions']) * 100

                return {
                    'command': command,
                    'total_executions': row['total_executions'],
                    'successful': row['successful'],
                    'failed': row['total_executions'] - row['successful'],
                    'success_rate': success_rate,
                    'avg_duration': row['avg_duration'],
                    'min_duration': row['min_duration'],
                    'max_duration': row['max_duration'],
                    'last_execution': row['last_execution']
                }

            return {
                'command': command,
                'total_executions': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0
            }

    def get_recent_executions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent command executions.

        Args:
            limit: Maximum number of records

        Returns:
            List of execution dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT * FROM command_executions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))

            return [dict(row) for row in cursor.fetchall()]

    def get_success_patterns(self, min_executions: int = 3) -> List[Dict[str, Any]]:
        """
        Get commands with high success rates.

        Args:
            min_executions: Minimum execution count

        Returns:
            List of successful command patterns
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT
                    command,
                    COUNT(*) as uses,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    AVG(duration) as avg_duration
                FROM command_executions
                WHERE executed = 1
                GROUP BY command
                HAVING COUNT(*) >= ?
                ORDER BY success_rate DESC, uses DESC
            """, (min_executions,))

            return [dict(row) for row in cursor.fetchall()]

    def get_failure_patterns(self, min_failures: int = 2) -> List[Dict[str, Any]]:
        """
        Get common failure patterns.

        Args:
            min_failures: Minimum failure count

        Returns:
            List of failure patterns
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT
                    command,
                    error_category,
                    COUNT(*) as failure_count,
                    GROUP_CONCAT(DISTINCT stderr_preview, '; ') as error_messages
                FROM command_executions
                WHERE success = 0
                GROUP BY command, error_category
                HAVING COUNT(*) >= ?
                ORDER BY failure_count DESC
            """, (min_failures,))

            return [dict(row) for row in cursor.fetchall()]

    def search_commands(self, search_query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Full-text search across command history.

        Args:
            search_query: Search query
            limit: Maximum results

        Returns:
            List of matching command executions
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT ce.*
                FROM command_executions ce
                INNER JOIN command_fts cf ON ce.execution_id = cf.execution_id
                WHERE cf.command MATCH ?
                ORDER BY ce.timestamp DESC
                LIMIT ?
            """, (search_query, limit))

            return [dict(row) for row in cursor.fetchall()]

    def get_venv_violations(self) -> List[Dict[str, Any]]:
        """
        Get commands that needed venv but ran without it.

        Returns:
            List of venv violation records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT command, COUNT(*) as count
                FROM command_executions
                WHERE (command LIKE 'pip %' OR command LIKE 'python %')
                AND venv_active = 0
                GROUP BY command
                ORDER BY count DESC
            """)

            return [dict(row) for row in cursor.fetchall()]

    def create_pattern(self,
                      pattern_name: str,
                      command_sequence: List[str],
                      task_category: str,
                      preconditions: Optional[Dict[str, Any]] = None) -> int:
        """
        Create a command pattern (learned sequence).

        Args:
            pattern_name: Name of pattern
            command_sequence: List of commands in sequence
            task_category: Category/type of task
            preconditions: Required preconditions (venv, cwd, etc.)

        Returns:
            pattern_id of created pattern
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO command_patterns (
                    pattern_name, command_sequence, task_category,
                    success_rate, usage_count, avg_duration,
                    preconditions, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern_name,
                json.dumps(command_sequence),
                task_category,
                0.0,  # Initial success rate
                0,    # Initial usage count
                0.0,  # Initial avg duration
                json.dumps(preconditions) if preconditions else None,
                time.time(),
                time.time()
            ))

            pattern_id = cursor.lastrowid
            conn.commit()

            logger.info(f"Created pattern: {pattern_id} - {pattern_name}")

            return pattern_id

    def record_pattern_usage(self,
                            pattern_id: int,
                            agent_id: str,
                            workflow_id: Optional[int],
                            success: bool,
                            duration: float) -> int:
        """
        Record usage of a pattern.

        Args:
            pattern_id: Pattern that was used
            agent_id: Agent that used it
            workflow_id: Associated workflow
            success: Whether pattern execution succeeded
            duration: Total duration

        Returns:
            usage_id of recorded usage
        """
        with sqlite3.connect(self.db_path) as conn:
            # Record usage
            cursor = conn.execute("""
                INSERT INTO command_pattern_usage (
                    pattern_id, workflow_id, agent_id,
                    success, duration, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (pattern_id, workflow_id, agent_id, success, duration, time.time()))

            usage_id = cursor.lastrowid

            # Update pattern statistics
            conn.execute("""
                UPDATE command_patterns
                SET usage_count = usage_count + 1,
                    updated_at = ?
                WHERE pattern_id = ?
            """, (time.time(), pattern_id))

            # Recalculate success rate
            stats = conn.execute("""
                SELECT
                    AVG(CASE WHEN success = 1 THEN 100.0 ELSE 0.0 END) as success_rate,
                    AVG(duration) as avg_duration
                FROM command_pattern_usage
                WHERE pattern_id = ?
            """, (pattern_id,)).fetchone()

            conn.execute("""
                UPDATE command_patterns
                SET success_rate = ?,
                    avg_duration = ?
                WHERE pattern_id = ?
            """, (stats[0], stats[1], pattern_id))

            conn.commit()

            logger.info(f"Recorded pattern usage: {usage_id}")

            return usage_id

    def get_patterns_by_category(self, task_category: str) -> List[Dict[str, Any]]:
        """
        Get patterns for a specific task category.

        Args:
            task_category: Category to filter by

        Returns:
            List of pattern dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT * FROM command_patterns
                WHERE task_category = ?
                ORDER BY success_rate DESC, usage_count DESC
            """, (task_category,))

            patterns = []
            for row in cursor.fetchall():
                pattern = dict(row)
                pattern['command_sequence'] = json.loads(pattern['command_sequence'])
                if pattern['preconditions']:
                    pattern['preconditions'] = json.loads(pattern['preconditions'])
                patterns.append(pattern)

            return patterns

    def get_active_commands(self) -> List[Dict[str, Any]]:
        """
        Get commands currently executing (status='running').

        Returns:
            List of active command dictionaries with execution details
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT
                    execution_id,
                    workflow_id,
                    agent_id,
                    agent_type,
                    command,
                    status,
                    execution_timestamp,
                    duration,
                    context,
                    trust_level
                FROM command_executions
                WHERE status = 'running'
                ORDER BY execution_timestamp DESC
            """)

            return [dict(row) for row in cursor.fetchall()]

    def get_filtered_history(self,
                            search_query: Optional[str] = None,
                            status: Optional[str] = None,  # 'success', 'failed', 'all'
                            agent_id: Optional[str] = None,
                            workflow_id: Optional[int] = None,
                            date_from: Optional[float] = None,
                            date_to: Optional[float] = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get command history with comprehensive filtering.

        Args:
            search_query: Search text (searches command, context, output)
            status: Filter by status ('success', 'failed', 'all')
            agent_id: Filter by specific agent
            workflow_id: Filter by workflow
            date_from: Start timestamp (Unix time)
            date_to: End timestamp (Unix time)
            limit: Maximum number of results

        Returns:
            List of command execution dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build query dynamically based on filters
            where_clauses = []
            params = []

            # Status filter
            if status and status != 'all':
                if status == 'success':
                    where_clauses.append("success = 1")
                elif status == 'failed':
                    where_clauses.append("success = 0")

            # Agent filter
            if agent_id:
                where_clauses.append("agent_id = ?")
                params.append(agent_id)

            # Workflow filter
            if workflow_id is not None:
                where_clauses.append("workflow_id = ?")
                params.append(workflow_id)

            # Date range filter
            if date_from is not None:
                where_clauses.append("timestamp >= ?")
                params.append(date_from)

            if date_to is not None:
                where_clauses.append("timestamp <= ?")
                params.append(date_to)

            # Search query (use FTS if provided)
            if search_query:
                # Use FTS for full-text search
                where_clauses.append("""
                    execution_id IN (
                        SELECT execution_id FROM command_fts
                        WHERE command_fts MATCH ?
                    )
                """)
                params.append(search_query)

            # Build final query
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

            query = f"""
                SELECT
                    execution_id,
                    workflow_id,
                    agent_id,
                    agent_type,
                    command,
                    status,
                    trust_level,
                    approved_by,
                    execution_timestamp,
                    exit_code,
                    duration,
                    success,
                    error_category,
                    context,
                    stdout_preview,
                    stderr_preview,
                    timestamp
                FROM command_executions
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """

            params.append(limit)
            cursor = conn.execute(query, params)

            return [dict(row) for row in cursor.fetchall()]

    def get_command_details(self, execution_id: int) -> Optional[Dict[str, Any]]:
        """
        Get full details for a specific command execution.

        Args:
            execution_id: Execution ID to retrieve

        Returns:
            Dictionary with all execution details, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            cursor = conn.execute("""
                SELECT * FROM command_executions
                WHERE execution_id = ?
            """, (execution_id,))

            row = cursor.fetchone()
            return dict(row) if row else None
