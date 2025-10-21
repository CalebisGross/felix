"""
System monitoring module for Felix Framework.

Provides read-only monitoring of the running Felix system and shared databases.
"""

import sqlite3
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging
import socket
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Felix components without modification
try:
    from src.gui.felix_system import FelixSystem, FelixConfig
    from src.memory.knowledge_store import KnowledgeStore
    from src.memory.task_memory import TaskMemory
except ImportError as e:
    logging.warning(f"Could not import Felix components: {e}")
    FelixSystem = None
    FelixConfig = None
    KnowledgeStore = None
    TaskMemory = None

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitor running Felix system and databases in read-only mode.
    This class provides safe, non-interfering access to Felix data.
    """

    def __init__(self):
        self.felix_system: Optional[FelixSystem] = None
        self.is_connected = False
        self.read_only = True

        # Connect to shared databases (read-only by default)
        self.knowledge_store = None
        self.task_memory = None
        self._init_db_connections()

    def _init_db_connections(self):
        """Initialize read-only database connections."""
        try:
            # Connect to shared databases
            if KnowledgeStore and Path("felix_knowledge.db").exists():
                self.knowledge_store = KnowledgeStore("felix_knowledge.db")
                logger.info("Connected to felix_knowledge.db (read-only)")

            if TaskMemory and Path("felix_memory.db").exists():
                self.task_memory = TaskMemory("felix_memory.db")
                logger.info("Connected to felix_memory.db (read-only)")

        except Exception as e:
            logger.warning(f"Could not connect to databases: {e}")

    def check_felix_running(self) -> bool:
        """
        Check if Felix system is currently running.
        Uses lock file or port check to detect running instance.
        """
        # Check for lock file (created by tkinter GUI)
        lock_file = Path(".felix_running.lock")
        if lock_file.exists():
            try:
                with open(lock_file, 'r') as f:
                    lock_data = json.load(f)
                    return lock_data.get("running", False)
            except:
                pass

        # Alternative: Check if LM Studio port is in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 1234))
        sock.close()
        return result == 0

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics from shared databases."""
        metrics = {
            "felix_running": self.check_felix_running(),
            "agents": 0,
            "messages": 0,
            "knowledge_entries": 0,
            "task_patterns": 0,
            "confidence_avg": 0.0,
            "new_entries": 0  # Added for delta display
        }

        # Read metrics from databases
        if self.knowledge_store:
            try:
                # For simplicity, we'll get all entries
                # In production, you'd want to optimize this
                conn = sqlite3.connect("felix_knowledge.db")
                cursor = conn.cursor()

                # Count total entries
                cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
                count = cursor.fetchone()
                if count:
                    metrics["knowledge_entries"] = count[0]

                # Get average confidence
                cursor.execute("SELECT AVG(confidence) FROM knowledge_entries WHERE confidence IS NOT NULL")
                avg_conf = cursor.fetchone()
                if avg_conf and avg_conf[0]:
                    metrics["confidence_avg"] = avg_conf[0]

                # Count recent entries (last hour)
                cursor.execute("""
                    SELECT COUNT(*) FROM knowledge_entries
                    WHERE timestamp > datetime('now', '-1 hour')
                """)
                new_count = cursor.fetchone()
                if new_count:
                    metrics["new_entries"] = new_count[0]

                conn.close()

            except Exception as e:
                logger.debug(f"Could not read knowledge entries: {e}")

        if self.task_memory:
            try:
                # Count task patterns
                conn = sqlite3.connect("felix_memory.db")
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM task_patterns")
                count = cursor.fetchone()
                if count:
                    metrics["task_patterns"] = count[0]

                conn.close()

            except Exception as e:
                logger.debug(f"Could not read task patterns: {e}")

        return metrics

    def get_agent_data(self) -> List[Dict[str, Any]]:
        """Read agent data from shared databases."""
        agents = []

        try:
            if Path("felix_knowledge.db").exists():
                conn = sqlite3.connect("felix_knowledge.db")
                cursor = conn.cursor()

                # Get recent agent activities - check column names
                # First check if these columns exist
                cursor.execute("PRAGMA table_info(knowledge_entries)")
                columns = [col[1] for col in cursor.fetchall()]

                # Build query based on available columns
                if 'entry_id' in columns and 'domain' in columns and 'confidence' in columns:
                    cursor.execute("""
                        SELECT entry_id, domain, confidence, timestamp, content
                        FROM knowledge_entries
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """)

                for row in cursor.fetchall():
                    agents.append({
                        "agent_id": row[0],
                        "domain": row[1],
                        "confidence": row[2],
                        "timestamp": row[3],
                        "content": row[4][:200] if row[4] else ""  # Truncate for display
                    })

                conn.close()

        except Exception as e:
            logger.debug(f"Could not read agent data: {e}")

        return agents

    def get_workflow_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Read recent workflow results from databases."""
        results = []

        try:
            if Path("felix_task_memory.db").exists():
                conn = sqlite3.connect("felix_task_memory.db")
                cursor = conn.cursor()

                # Check if the table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='workflows'
                """)

                if cursor.fetchone():
                    # Get recent workflow results
                    cursor.execute(f"""
                        SELECT task, timestamp, success, agent_count, final_synthesis
                        FROM workflows
                        ORDER BY timestamp DESC
                        LIMIT {limit}
                    """)

                    for row in cursor.fetchall():
                        results.append({
                            "task": row[0],
                            "timestamp": row[1],
                            "success": row[2],
                            "agent_count": row[3],
                            "final_synthesis": row[4] if row[4] else ""
                        })

                conn.close()

        except Exception as e:
            logger.debug(f"Could not read workflow results: {e}")

        # Fallback to task memory if no workflow table
        if not results and Path("felix_memory.db").exists():
            try:
                conn = sqlite3.connect("felix_memory.db")
                cursor = conn.cursor()

                cursor.execute(f"""
                    SELECT pattern, timestamp
                    FROM task_patterns
                    ORDER BY timestamp DESC
                    LIMIT {limit}
                """)

                for row in cursor.fetchall():
                    results.append({
                        "task": row[0],
                        "timestamp": row[1],
                        "success": True,  # Assume success if in database
                        "agent_count": 0,
                        "final_synthesis": ""
                    })

                conn.close()

            except Exception as e:
                logger.debug(f"Could not read task memory: {e}")

        return results

    def create_isolated_instance(self, config: Dict[str, Any]) -> Optional[FelixSystem]:
        """
        Create an isolated Felix instance for benchmarking.
        This instance uses temporary databases to avoid interference.
        """
        if not FelixSystem or not FelixConfig:
            logger.error("Felix components not available for isolated instance")
            return None

        try:
            # Create config with temporary databases
            felix_config = FelixConfig(
                lm_host=config.get('lm_host', '127.0.0.1'),
                lm_port=config.get('lm_port', 1234),
                memory_db_path="temp_memory.db",  # Temporary DB
                knowledge_db_path="temp_knowledge.db",  # Temporary DB
                **config
            )

            # Create isolated instance
            isolated_system = FelixSystem(felix_config)
            logger.info("Created isolated Felix instance for benchmarking")
            return isolated_system

        except Exception as e:
            logger.error(f"Could not create isolated instance: {e}")
            return None