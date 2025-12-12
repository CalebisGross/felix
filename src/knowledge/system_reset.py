"""
System Reset Manager for Felix.

Handles full system backup and reset operations for all Felix databases.
"""

import os
import shutil
import glob
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SystemResetManager:
    """Handles full system backup and reset operations."""

    # All Felix database files relative to project root
    DATABASE_FILES = [
        "felix_knowledge.db",
        "felix_workflow_history.db",
        "felix_memory.db",
        "felix_task_memory.db",
        "felix_system_actions.db",
        "felix_cli_sessions.db",
        "felix_agent_performance.db",
        "prompts/felix_prompts.db",
    ]

    # Index files to delete on reset (glob patterns)
    INDEX_PATTERNS = [
        ".felix_index.json",
        "**/.felix_index.json",
    ]

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize SystemResetManager.

        Args:
            project_root: Path to Felix project root. Defaults to current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.backup_base_dir = self.project_root / "backups" / "system"

    def get_database_stats(self) -> List[Dict[str, Any]]:
        """
        Get status and size information for all databases.

        Returns:
            List of dicts with database info: name, path, exists, size_bytes, size_display
        """
        stats = []
        for db_file in self.DATABASE_FILES:
            db_path = self.project_root / db_file
            exists = db_path.exists()
            size_bytes = db_path.stat().st_size if exists else 0

            # Format size for display
            if size_bytes >= 1024 * 1024:
                size_display = f"{size_bytes / (1024 * 1024):.1f} MB"
            elif size_bytes >= 1024:
                size_display = f"{size_bytes / 1024:.1f} KB"
            else:
                size_display = f"{size_bytes} B"

            stats.append({
                "name": db_file,
                "path": str(db_path),
                "exists": exists,
                "size_bytes": size_bytes,
                "size_display": size_display,
            })

        return stats

    def get_total_size(self) -> int:
        """
        Get total size of all existing databases in bytes.

        Returns:
            Total size in bytes
        """
        total = 0
        for db_file in self.DATABASE_FILES:
            db_path = self.project_root / db_file
            if db_path.exists():
                total += db_path.stat().st_size
        return total

    def get_last_backup_time(self) -> Optional[datetime]:
        """
        Get timestamp of most recent backup.

        Returns:
            datetime of last backup, or None if no backups exist
        """
        if not self.backup_base_dir.exists():
            return None

        # List backup directories (format: YYYY-MM-DD_HHMMSS)
        backup_dirs = sorted(self.backup_base_dir.iterdir(), reverse=True)
        for backup_dir in backup_dirs:
            if backup_dir.is_dir():
                try:
                    # Parse directory name as timestamp
                    timestamp_str = backup_dir.name
                    return datetime.strptime(timestamp_str, "%Y-%m-%d_%H%M%S")
                except ValueError:
                    continue
        return None

    def backup_all_databases(self) -> Dict[str, Any]:
        """
        Backup all databases to a timestamped directory.

        Returns:
            Dict with: success, backup_path, backed_up (list), skipped (list), error (if failed)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup_dir = self.backup_base_dir / timestamp

        try:
            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)

            backed_up = []
            skipped = []

            for db_file in self.DATABASE_FILES:
                db_path = self.project_root / db_file

                if not db_path.exists():
                    skipped.append(db_file)
                    continue

                # Create backup filename (replace / with _ for subdirectory files)
                backup_name = db_file.replace("/", "_")
                backup_path = backup_dir / backup_name

                # Copy the database file
                shutil.copy2(db_path, backup_path)
                backed_up.append(db_file)
                logger.info(f"Backed up {db_file} to {backup_path}")

            logger.info(f"System backup complete: {len(backed_up)} databases backed up to {backup_dir}")

            return {
                "success": True,
                "backup_path": str(backup_dir),
                "backed_up": backed_up,
                "skipped": skipped,
            }

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "backed_up": [],
                "skipped": [],
            }

    def find_index_files(self) -> List[str]:
        """
        Find all .felix_index.json files.

        Returns:
            List of paths to index files
        """
        index_files = []
        for pattern in self.INDEX_PATTERNS:
            matches = glob.glob(str(self.project_root / pattern), recursive=True)
            index_files.extend(matches)
        return list(set(index_files))  # Remove duplicates

    def wipe_all_databases(self, delete_indexes: bool = True) -> Dict[str, Any]:
        """
        Delete all database files and optionally index files.

        Args:
            delete_indexes: Whether to also delete .felix_index.json files

        Returns:
            Dict with: success, deleted (list), not_found (list), index_files_deleted (list), error (if failed)
        """
        deleted = []
        not_found = []
        index_files_deleted = []

        try:
            # Delete database files
            for db_file in self.DATABASE_FILES:
                db_path = self.project_root / db_file

                if not db_path.exists():
                    not_found.append(db_file)
                    continue

                db_path.unlink()
                deleted.append(db_file)
                logger.info(f"Deleted {db_file}")

            # Delete index files if requested
            if delete_indexes:
                index_files = self.find_index_files()
                for index_file in index_files:
                    try:
                        os.remove(index_file)
                        index_files_deleted.append(index_file)
                        logger.info(f"Deleted index file: {index_file}")
                    except Exception as e:
                        logger.warning(f"Could not delete index file {index_file}: {e}")

            logger.info(f"System wipe complete: {len(deleted)} databases deleted, {len(index_files_deleted)} index files deleted")

            return {
                "success": True,
                "deleted": deleted,
                "not_found": not_found,
                "index_files_deleted": index_files_deleted,
            }

        except Exception as e:
            logger.error(f"Wipe failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "deleted": deleted,
                "not_found": not_found,
                "index_files_deleted": index_files_deleted,
            }

    def backup_and_reset(self, delete_indexes: bool = True) -> Dict[str, Any]:
        """
        Convenience method to backup then wipe all databases.

        Args:
            delete_indexes: Whether to also delete .felix_index.json files

        Returns:
            Dict with backup and wipe results
        """
        # First backup
        backup_result = self.backup_all_databases()
        if not backup_result["success"]:
            return {
                "success": False,
                "phase": "backup",
                "error": backup_result.get("error", "Backup failed"),
                "backup_result": backup_result,
            }

        # Then wipe
        wipe_result = self.wipe_all_databases(delete_indexes=delete_indexes)
        if not wipe_result["success"]:
            return {
                "success": False,
                "phase": "wipe",
                "error": wipe_result.get("error", "Wipe failed"),
                "backup_result": backup_result,
                "wipe_result": wipe_result,
            }

        return {
            "success": True,
            "backup_result": backup_result,
            "wipe_result": wipe_result,
        }

    def initialize_all_databases(self) -> Dict[str, Any]:
        """
        Initialize all databases with proper schemas after a reset.

        Returns:
            Dict with: success, initialized (list), errors (list)
        """
        initialized = []
        errors = []

        # 1. Initialize KnowledgeStore (auto-initializes schema)
        try:
            from src.memory.knowledge_store import KnowledgeStore
            db_path = str(self.project_root / "felix_knowledge.db")
            store = KnowledgeStore(db_path)

            # Run Knowledge Brain migration to add document_sources, etc.
            try:
                from src.migration.add_knowledge_brain import AddKnowledgeBrain
                conn = sqlite3.connect(db_path)
                kb_mig = AddKnowledgeBrain()
                kb_mig.up(conn)
                conn.commit()
                conn.close()
                logger.info("Applied Knowledge Brain migration")
            except Exception as mig_e:
                logger.warning(f"Knowledge Brain migration: {mig_e}")

            # Run audit log migration
            try:
                from src.migration.add_audit_log_table import migrate_up as audit_migrate_up
                audit_migrate_up(db_path)
                logger.info("Applied audit log migration")
            except Exception as mig_e:
                logger.warning(f"Audit log migration: {mig_e}")

            initialized.append("felix_knowledge.db")
            logger.info("Initialized felix_knowledge.db")
        except Exception as e:
            errors.append(f"felix_knowledge.db: {str(e)}")
            logger.error(f"Failed to initialize felix_knowledge.db: {e}")

        # 2. Initialize TaskMemory (auto-initializes schema)
        try:
            from src.memory.task_memory import TaskMemory
            db_path = str(self.project_root / "felix_task_memory.db")
            memory = TaskMemory(db_path)
            initialized.append("felix_task_memory.db")
            logger.info("Initialized felix_task_memory.db")
        except Exception as e:
            errors.append(f"felix_task_memory.db: {str(e)}")
            logger.error(f"Failed to initialize felix_task_memory.db: {e}")

        # 3. Initialize WorkflowHistory (auto-initializes schema)
        try:
            from src.memory.workflow_history import WorkflowHistory
            db_path = str(self.project_root / "felix_workflow_history.db")
            history = WorkflowHistory(db_path)
            initialized.append("felix_workflow_history.db")
            logger.info("Initialized felix_workflow_history.db")
        except Exception as e:
            errors.append(f"felix_workflow_history.db: {str(e)}")
            logger.error(f"Failed to initialize felix_workflow_history.db: {e}")

        # 4. Initialize CLI SessionManager (auto-initializes schema)
        try:
            from src.cli_chat.session_manager import SessionManager
            db_path = str(self.project_root / "felix_cli_sessions.db")
            session_mgr = SessionManager(db_path)
            initialized.append("felix_cli_sessions.db")
            logger.info("Initialized felix_cli_sessions.db")
        except Exception as e:
            errors.append(f"felix_cli_sessions.db: {str(e)}")
            logger.error(f"Failed to initialize felix_cli_sessions.db: {e}")

        # 5. Initialize SystemActions database (requires migration)
        try:
            db_path = str(self.project_root / "felix_system_actions.db")
            self._initialize_system_actions_db(db_path)
            initialized.append("felix_system_actions.db")
            logger.info("Initialized felix_system_actions.db")
        except Exception as e:
            errors.append(f"felix_system_actions.db: {str(e)}")
            logger.error(f"Failed to initialize felix_system_actions.db: {e}")

        # 6. Initialize felix_memory.db (simple schema)
        try:
            db_path = str(self.project_root / "felix_memory.db")
            self._initialize_memory_db(db_path)
            initialized.append("felix_memory.db")
            logger.info("Initialized felix_memory.db")
        except Exception as e:
            errors.append(f"felix_memory.db: {str(e)}")
            logger.error(f"Failed to initialize felix_memory.db: {e}")

        # 7. Initialize agent_performance database
        try:
            db_path = str(self.project_root / "felix_agent_performance.db")
            self._initialize_agent_performance_db(db_path)
            initialized.append("felix_agent_performance.db")
            logger.info("Initialized felix_agent_performance.db")
        except Exception as e:
            errors.append(f"felix_agent_performance.db: {str(e)}")
            logger.error(f"Failed to initialize felix_agent_performance.db: {e}")

        # 8. Initialize prompts database
        try:
            prompts_dir = self.project_root / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(prompts_dir / "felix_prompts.db")
            self._initialize_prompts_db(db_path)
            initialized.append("prompts/felix_prompts.db")
            logger.info("Initialized prompts/felix_prompts.db")
        except Exception as e:
            errors.append(f"prompts/felix_prompts.db: {str(e)}")
            logger.error(f"Failed to initialize prompts/felix_prompts.db: {e}")

        success = len(errors) == 0
        logger.info(f"Database initialization complete: {len(initialized)} initialized, {len(errors)} errors")

        return {
            "success": success,
            "initialized": initialized,
            "errors": errors,
        }

    def _initialize_system_actions_db(self, db_path: str):
        """Initialize felix_system_actions.db with its schema."""
        try:
            from src.migration.create_system_actions import get_migrations
            conn = sqlite3.connect(db_path)
            for migration in get_migrations():
                migration.up(conn)
            conn.close()
        except ImportError:
            # Fallback: create minimal schema if migration not available
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS command_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    output TEXT,
                    exit_code INTEGER,
                    timestamp REAL NOT NULL,
                    duration_ms INTEGER,
                    cwd TEXT,
                    trust_level TEXT DEFAULT 'UNTRUSTED'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    reason TEXT,
                    trust_level TEXT,
                    created_at REAL NOT NULL,
                    status TEXT DEFAULT 'PENDING'
                )
            """)
            conn.commit()
            conn.close()

    def _initialize_memory_db(self, db_path: str):
        """Initialize felix_memory.db with minimal schema."""
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                created_at REAL NOT NULL,
                completed_at REAL,
                status TEXT DEFAULT 'pending'
            )
        """)
        conn.commit()
        conn.close()

    def _initialize_agent_performance_db(self, db_path: str):
        """Initialize felix_agent_performance.db with its schema."""
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                task_id TEXT,
                tokens_used INTEGER,
                response_time_ms INTEGER,
                confidence REAL,
                synthesis_integration_rate REAL,
                timestamp REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS confidence_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                predicted_confidence REAL,
                actual_confidence REAL,
                calibration_gap REAL,
                timestamp REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _initialize_prompts_db(self, db_path: str):
        """Initialize prompts/felix_prompts.db with its schema."""
        conn = sqlite3.connect(db_path)

        # Custom prompts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS custom_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_key TEXT NOT NULL,
                template_text TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                version INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)

        # Prompt performance tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_key TEXT NOT NULL,
                version INTEGER NOT NULL,
                agent_id TEXT,
                confidence REAL,
                tokens_used INTEGER,
                processing_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Prompt version history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prompt_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_key TEXT NOT NULL,
                version INTEGER NOT NULL,
                template_text TEXT NOT NULL,
                action TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)

        # Create indices
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prompt_key ON custom_prompts(prompt_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_active ON custom_prompts(is_active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_performance ON prompt_performance(prompt_key, version)")

        conn.commit()
        conn.close()
