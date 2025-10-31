"""
Workflow History Database Module

This module manages persistent storage of workflow execution history.
Each workflow result is saved with comprehensive metadata for tracking
and analysis.
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WorkflowOutput:
    """Represents a saved workflow output."""
    workflow_id: int
    task_input: str
    status: str
    created_at: str
    completed_at: str
    final_synthesis: str
    confidence: float
    agents_count: int
    tokens_used: int
    max_tokens: int
    processing_time: float
    temperature: float
    metadata: Dict[str, Any]
    parent_workflow_id: Optional[int] = None
    conversation_thread_id: Optional[str] = None


class WorkflowHistory:
    """
    Manages workflow history database.

    Provides methods to save, retrieve, search, and delete workflow outputs.
    """

    def __init__(self, db_path: str = "felix_workflow_history.db"):
        """
        Initialize workflow history database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema if it doesn't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create workflow_outputs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_outputs (
                    workflow_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_input TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    final_synthesis TEXT,
                    confidence REAL DEFAULT 0.0,
                    agents_count INTEGER DEFAULT 0,
                    tokens_used INTEGER DEFAULT 0,
                    max_tokens INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0.0,
                    temperature REAL DEFAULT 0.0,
                    metadata TEXT,
                    parent_workflow_id INTEGER,
                    conversation_thread_id TEXT,
                    FOREIGN KEY (parent_workflow_id) REFERENCES workflow_outputs(workflow_id)
                )
            """)

            # Create index on created_at for faster date-based queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON workflow_outputs(created_at DESC)
            """)

            # Create index on status for filtering
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON workflow_outputs(status)
            """)

            conn.commit()

            # Migrate existing database if needed (adds columns if missing)
            self._migrate_schema(cursor)
            conn.commit()

            # Create indexes for conversation threading (after migration ensures columns exist)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_thread
                ON workflow_outputs(conversation_thread_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent_workflow
                ON workflow_outputs(parent_workflow_id)
            """)

            conn.commit()

            conn.close()
            logger.info(f"Workflow history database initialized at {self.db_path}")

        except sqlite3.Error as e:
            logger.error(f"Failed to initialize workflow history database: {e}")
            raise

    def _migrate_schema(self, cursor):
        """
        Migrate existing database schema to add conversation threading columns.

        Args:
            cursor: SQLite cursor
        """
        try:
            # Check if parent_workflow_id column exists
            cursor.execute("PRAGMA table_info(workflow_outputs)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'parent_workflow_id' not in columns:
                logger.info("Migrating database: adding parent_workflow_id column")
                cursor.execute("""
                    ALTER TABLE workflow_outputs
                    ADD COLUMN parent_workflow_id INTEGER
                """)

            if 'conversation_thread_id' not in columns:
                logger.info("Migrating database: adding conversation_thread_id column")
                cursor.execute("""
                    ALTER TABLE workflow_outputs
                    ADD COLUMN conversation_thread_id TEXT
                """)

        except sqlite3.Error as e:
            logger.warning(f"Schema migration warning: {e}")

    def save_workflow_output(self, result: Dict[str, Any],
                            parent_workflow_id: Optional[int] = None,
                            conversation_thread_id: Optional[str] = None) -> Optional[int]:
        """
        Save a workflow output to the database.

        Args:
            result: Workflow result dictionary from run_felix_workflow
            parent_workflow_id: Optional ID of parent workflow for conversation threading
            conversation_thread_id: Optional thread ID for grouping related workflows

        Returns:
            workflow_id of saved entry, or None if failed
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract fields from result
            status = result.get("status", "unknown")
            task_input = result.get("task_input", "")

            # Get timestamps
            created_at = result.get("start_time", datetime.now().isoformat())
            completed_at = result.get("end_time", datetime.now().isoformat())

            # Calculate processing time
            try:
                start = datetime.fromisoformat(created_at)
                end = datetime.fromisoformat(completed_at)
                processing_time = (end - start).total_seconds()
            except:
                processing_time = result.get("processing_time", 0.0)

            # Extract synthesis data
            centralpost_synthesis = result.get("centralpost_synthesis", {})
            final_synthesis = centralpost_synthesis.get("synthesis_content", "")
            confidence = centralpost_synthesis.get("confidence", 0.0)
            agents_count = centralpost_synthesis.get("agents_synthesized", len(result.get("agents_spawned", [])))
            tokens_used = centralpost_synthesis.get("tokens_used", 0)
            max_tokens = centralpost_synthesis.get("max_tokens", 0)
            temperature = centralpost_synthesis.get("temperature", 0.0)

            # Store entire result as metadata for comprehensive history
            metadata_json = json.dumps(result)

            # Generate thread ID if parent provided but no thread ID
            if parent_workflow_id and not conversation_thread_id:
                # Inherit thread ID from parent or create new one
                cursor.execute("""
                    SELECT conversation_thread_id FROM workflow_outputs
                    WHERE workflow_id = ?
                """, (parent_workflow_id,))
                parent_row = cursor.fetchone()
                if parent_row and parent_row[0]:
                    conversation_thread_id = parent_row[0]
                else:
                    # Parent has no thread, create new one using parent's ID
                    import uuid
                    conversation_thread_id = str(uuid.uuid4())

            # Insert into database
            cursor.execute("""
                INSERT INTO workflow_outputs
                (task_input, status, created_at, completed_at, final_synthesis,
                 confidence, agents_count, tokens_used, max_tokens,
                 processing_time, temperature, metadata,
                 parent_workflow_id, conversation_thread_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_input, status, created_at, completed_at, final_synthesis,
                confidence, agents_count, tokens_used, max_tokens,
                processing_time, temperature, metadata_json,
                parent_workflow_id, conversation_thread_id
            ))

            workflow_id = cursor.lastrowid
            conn.commit()
            conn.close()

            if parent_workflow_id:
                logger.info(f"Saved workflow output with ID: {workflow_id} (continuing from workflow {parent_workflow_id})")
            else:
                logger.info(f"Saved workflow output with ID: {workflow_id}")
            return workflow_id

        except sqlite3.Error as e:
            logger.error(f"Failed to save workflow output: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error saving workflow output: {e}", exc_info=True)
            return None

    def get_workflow_outputs(self,
                            status_filter: Optional[str] = None,
                            limit: int = 100,
                            offset: int = 0) -> List[WorkflowOutput]:
        """
        Retrieve workflow outputs with optional filtering.

        Args:
            status_filter: Filter by status ("completed", "failed", or None for all)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of WorkflowOutput objects
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build query
            query = """
                SELECT workflow_id, task_input, status, created_at, completed_at,
                       final_synthesis, confidence, agents_count, tokens_used,
                       max_tokens, processing_time, temperature, metadata,
                       parent_workflow_id, conversation_thread_id
                FROM workflow_outputs
            """
            params = []

            if status_filter:
                query += " WHERE status = ?"
                params.append(status_filter)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()

            # Convert to WorkflowOutput objects
            outputs = []
            for row in rows:
                metadata = json.loads(row[12]) if row[12] else {}
                output = WorkflowOutput(
                    workflow_id=row[0],
                    task_input=row[1],
                    status=row[2],
                    created_at=row[3],
                    completed_at=row[4],
                    final_synthesis=row[5] or "",
                    confidence=row[6],
                    agents_count=row[7],
                    tokens_used=row[8],
                    max_tokens=row[9],
                    processing_time=row[10],
                    temperature=row[11],
                    metadata=metadata,
                    parent_workflow_id=row[13],
                    conversation_thread_id=row[14]
                )
                outputs.append(output)

            return outputs

        except sqlite3.Error as e:
            logger.error(f"Failed to retrieve workflow outputs: {e}", exc_info=True)
            return []

    def get_workflow_by_id(self, workflow_id: int) -> Optional[WorkflowOutput]:
        """
        Get a specific workflow by ID.

        Args:
            workflow_id: Workflow ID to retrieve

        Returns:
            WorkflowOutput object or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT workflow_id, task_input, status, created_at, completed_at,
                       final_synthesis, confidence, agents_count, tokens_used,
                       max_tokens, processing_time, temperature, metadata,
                       parent_workflow_id, conversation_thread_id
                FROM workflow_outputs
                WHERE workflow_id = ?
            """, (workflow_id,))

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            metadata = json.loads(row[12]) if row[12] else {}
            return WorkflowOutput(
                workflow_id=row[0],
                task_input=row[1],
                status=row[2],
                created_at=row[3],
                completed_at=row[4],
                final_synthesis=row[5] or "",
                confidence=row[6],
                agents_count=row[7],
                tokens_used=row[8],
                max_tokens=row[9],
                processing_time=row[10],
                temperature=row[11],
                metadata=metadata,
                parent_workflow_id=row[13],
                conversation_thread_id=row[14]
            )

        except sqlite3.Error as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}", exc_info=True)
            return None

    def search_workflows(self, keyword: str, limit: int = 100, offset: int = 0) -> List[WorkflowOutput]:
        """
        Search workflows by keyword in task_input or final_synthesis.

        Args:
            keyword: Search keyword
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of matching WorkflowOutput objects
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            search_pattern = f"%{keyword}%"
            cursor.execute("""
                SELECT workflow_id, task_input, status, created_at, completed_at,
                       final_synthesis, confidence, agents_count, tokens_used,
                       max_tokens, processing_time, temperature, metadata,
                       parent_workflow_id, conversation_thread_id
                FROM workflow_outputs
                WHERE task_input LIKE ? OR final_synthesis LIKE ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (search_pattern, search_pattern, limit, offset))

            rows = cursor.fetchall()
            conn.close()

            # Convert to WorkflowOutput objects
            outputs = []
            for row in rows:
                metadata = json.loads(row[12]) if row[12] else {}
                output = WorkflowOutput(
                    workflow_id=row[0],
                    task_input=row[1],
                    status=row[2],
                    created_at=row[3],
                    completed_at=row[4],
                    final_synthesis=row[5] or "",
                    confidence=row[6],
                    agents_count=row[7],
                    tokens_used=row[8],
                    max_tokens=row[9],
                    processing_time=row[10],
                    temperature=row[11],
                    metadata=metadata,
                    parent_workflow_id=row[13],
                    conversation_thread_id=row[14]
                )
                outputs.append(output)

            return outputs

        except sqlite3.Error as e:
            logger.error(f"Failed to search workflows: {e}", exc_info=True)
            return []

    def get_conversation_thread(self, workflow_id: int) -> List[WorkflowOutput]:
        """
        Get all workflows in a conversation thread.

        Args:
            workflow_id: ID of any workflow in the thread

        Returns:
            List of WorkflowOutput objects in chronological order
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # First, get the thread ID and parent ID for this workflow
            cursor.execute("""
                SELECT conversation_thread_id, parent_workflow_id FROM workflow_outputs
                WHERE workflow_id = ?
            """, (workflow_id,))
            row = cursor.fetchone()

            if not row:
                # Workflow not found
                conn.close()
                return []

            thread_id, parent_id = row[0], row[1]

            # Always use parent/child traversal to ensure we get complete thread
            # even if thread_id assignment was incomplete

            # Find root of conversation by walking up parent_id chain
            root_id = workflow_id
            current_parent = parent_id
            visited = set([workflow_id])

            while current_parent and current_parent not in visited:
                visited.add(current_parent)
                cursor.execute("""
                    SELECT parent_workflow_id FROM workflow_outputs
                    WHERE workflow_id = ?
                """, (current_parent,))
                next_row = cursor.fetchone()
                root_id = current_parent
                if next_row and next_row[0]:
                    current_parent = next_row[0]
                else:
                    break

            # Now collect all descendants from root
            workflow_ids = [root_id]
            to_process = [root_id]

            while to_process:
                current_id = to_process.pop(0)
                cursor.execute("""
                    SELECT workflow_id FROM workflow_outputs
                    WHERE parent_workflow_id = ?
                """, (current_id,))
                children = cursor.fetchall()
                for child in children:
                    child_id = child[0]
                    if child_id not in workflow_ids:
                        workflow_ids.append(child_id)
                        to_process.append(child_id)

            # Fetch all workflows in conversation
            placeholders = ','.join('?' * len(workflow_ids))
            cursor.execute(f"""
                SELECT workflow_id, task_input, status, created_at, completed_at,
                       final_synthesis, confidence, agents_count, tokens_used,
                       max_tokens, processing_time, temperature, metadata,
                       parent_workflow_id, conversation_thread_id
                FROM workflow_outputs
                WHERE workflow_id IN ({placeholders})
                ORDER BY created_at ASC
            """, workflow_ids)

            rows = cursor.fetchall()
            conn.close()

            # Convert to WorkflowOutput objects
            outputs = []
            for row in rows:
                metadata = json.loads(row[12]) if row[12] else {}
                output = WorkflowOutput(
                    workflow_id=row[0],
                    task_input=row[1],
                    status=row[2],
                    created_at=row[3],
                    completed_at=row[4],
                    final_synthesis=row[5] or "",
                    confidence=row[6],
                    agents_count=row[7],
                    tokens_used=row[8],
                    max_tokens=row[9],
                    processing_time=row[10],
                    temperature=row[11],
                    metadata=metadata,
                    parent_workflow_id=row[13],
                    conversation_thread_id=row[14]
                )
                outputs.append(output)

            return outputs

        except sqlite3.Error as e:
            logger.error(f"Failed to get conversation thread: {e}", exc_info=True)
            return []

    def get_parent_workflow(self, workflow_id: int) -> Optional[WorkflowOutput]:
        """
        Get the parent workflow for a given workflow.

        Args:
            workflow_id: ID of the child workflow

        Returns:
            Parent WorkflowOutput or None if no parent
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT parent_workflow_id FROM workflow_outputs
                WHERE workflow_id = ?
            """, (workflow_id,))
            row = cursor.fetchone()
            conn.close()

            if not row or not row[0]:
                return None

            return self.get_workflow_by_id(row[0])

        except sqlite3.Error as e:
            logger.error(f"Failed to get parent workflow: {e}", exc_info=True)
            return None

    def delete_workflow(self, workflow_id: int) -> bool:
        """
        Delete a workflow from the database.

        Args:
            workflow_id: ID of workflow to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM workflow_outputs WHERE workflow_id = ?", (workflow_id,))
            deleted = cursor.rowcount > 0

            conn.commit()
            conn.close()

            if deleted:
                logger.info(f"Deleted workflow {workflow_id}")
            else:
                logger.warning(f"Workflow {workflow_id} not found for deletion")

            return deleted

        except sqlite3.Error as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}", exc_info=True)
            return False

    def get_workflow_count(self, status_filter: Optional[str] = None) -> int:
        """
        Get total count of workflows.

        Args:
            status_filter: Filter by status or None for all

        Returns:
            Count of workflows
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if status_filter:
                cursor.execute("SELECT COUNT(*) FROM workflow_outputs WHERE status = ?", (status_filter,))
            else:
                cursor.execute("SELECT COUNT(*) FROM workflow_outputs")

            count = cursor.fetchone()[0]
            conn.close()

            return count

        except sqlite3.Error as e:
            logger.error(f"Failed to get workflow count: {e}", exc_info=True)
            return 0
