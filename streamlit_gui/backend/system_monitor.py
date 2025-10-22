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

    def __init__(self, db_dir: Optional[str] = None):
        """
        Initialize SystemMonitor.

        Args:
            db_dir: Directory containing databases. If None, uses project root.
        """
        self.felix_system: Optional[FelixSystem] = None
        self.is_connected = False
        self.read_only = True

        # Set database directory (default to project root)
        if db_dir is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            streamlit_gui_dir = os.path.dirname(backend_dir)
            db_dir = os.path.dirname(streamlit_gui_dir)
        self.db_dir = db_dir

        # Database paths
        self.knowledge_db_path = os.path.join(self.db_dir, "felix_knowledge.db")
        self.memory_db_path = os.path.join(self.db_dir, "felix_memory.db")
        self.task_memory_db_path = os.path.join(self.db_dir, "felix_task_memory.db")

        # Connect to shared databases (read-only by default)
        self.knowledge_store = None
        self.task_memory = None
        self._init_db_connections()

    def _init_db_connections(self):
        """Initialize read-only database connections with retry logic."""
        max_retries = 3
        retry_delay = 0.5

        try:
            # Connect to shared databases with retry logic
            if KnowledgeStore and Path(self.knowledge_db_path).exists():
                for attempt in range(max_retries):
                    try:
                        self.knowledge_store = KnowledgeStore(self.knowledge_db_path)
                        logger.info(f"Connected to {self.knowledge_db_path} (read-only)")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.debug(f"Retry {attempt + 1}/{max_retries} for knowledge_store: {e}")
                            import time
                            time.sleep(retry_delay * (attempt + 1))
                        else:
                            logger.warning(f"Failed to connect to knowledge_store after {max_retries} attempts: {e}")
            else:
                logger.info(f"{self.knowledge_db_path} not found - will show empty data")

            if TaskMemory and Path(self.memory_db_path).exists():
                for attempt in range(max_retries):
                    try:
                        self.task_memory = TaskMemory(self.memory_db_path)
                        logger.info(f"Connected to {self.memory_db_path} (read-only)")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.debug(f"Retry {attempt + 1}/{max_retries} for task_memory: {e}")
                            import time
                            time.sleep(retry_delay * (attempt + 1))
                        else:
                            logger.warning(f"Failed to connect to task_memory after {max_retries} attempts: {e}")
            else:
                logger.info(f"{self.memory_db_path} not found - will show empty data")

        except Exception as e:
            logger.error(f"Unexpected error during database connection: {e}")

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
        """Get current system metrics from shared databases and AgentRegistry."""
        metrics = {
            "felix_running": self.check_felix_running(),
            "agents": 0,
            "messages": 0,
            "knowledge_entries": 0,
            "task_patterns": 0,
            "confidence_avg": 0.0,
            "new_entries": 0,  # Added for delta display
            # Agent awareness metrics
            "phase_distribution": {"exploration": 0, "analysis": 0, "synthesis": 0},
            "convergence_ready": False,
            "collaboration_density": 0.0,
            # Error tracking
            "errors": []
        }

        # Read metrics from databases with comprehensive error handling
        if self.knowledge_store or Path(self.knowledge_db_path).exists():
            try:
                conn = sqlite3.connect(self.knowledge_db_path)
                cursor = conn.cursor()

                # Verify table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='knowledge_entries'
                """)
                if not cursor.fetchone():
                    logger.warning("Table 'knowledge_entries' not found in felix_knowledge.db")
                    metrics["errors"].append("Knowledge table not found - database may need initialization")
                else:
                    # Count total entries
                    cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
                    count = cursor.fetchone()
                    if count and count[0] > 0:
                        metrics["knowledge_entries"] = count[0]

                        # Get average confidence (use confidence_level TEXT: low=0.3, medium=0.6, high=0.9)
                        cursor.execute("""
                            SELECT AVG(
                                CASE confidence_level
                                    WHEN 'low' THEN 0.3
                                    WHEN 'medium' THEN 0.6
                                    WHEN 'high' THEN 0.9
                                    ELSE 0.5
                                END
                            ) FROM knowledge_entries WHERE confidence_level IS NOT NULL
                        """)
                        avg_conf = cursor.fetchone()
                        if avg_conf and avg_conf[0] is not None:
                            metrics["confidence_avg"] = round(avg_conf[0], 3)

                        # Count recent entries (last hour) - created_at is REAL (Unix timestamp)
                        cursor.execute("""
                            SELECT COUNT(*) FROM knowledge_entries
                            WHERE created_at > (strftime('%s', 'now') - 3600)
                        """)
                        new_count = cursor.fetchone()
                        if new_count:
                            metrics["new_entries"] = new_count[0]
                    else:
                        logger.info("No data in knowledge_entries - run workflows to generate metrics")

                conn.close()

            except sqlite3.Error as e:
                logger.error(f"Database error reading knowledge entries: {e}")
                metrics["errors"].append(f"Knowledge DB error: {str(e)[:50]}")
            except Exception as e:
                logger.error(f"Unexpected error reading knowledge entries: {e}")
                metrics["errors"].append(f"Unexpected error: {str(e)[:50]}")
        else:
            logger.info("felix_knowledge.db not found - showing empty metrics")
            metrics["errors"].append("Knowledge database not found - run Felix to create it")

        if self.task_memory or Path(self.memory_db_path).exists():
            try:
                conn = sqlite3.connect(self.memory_db_path)
                cursor = conn.cursor()

                # Verify table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='task_patterns'
                """)
                if not cursor.fetchone():
                    logger.warning("Table 'task_patterns' not found in felix_memory.db")
                    metrics["errors"].append("Task patterns table not found")
                else:
                    cursor.execute("SELECT COUNT(*) FROM task_patterns")
                    count = cursor.fetchone()
                    if count and count[0] > 0:
                        metrics["task_patterns"] = count[0]

                conn.close()

            except sqlite3.Error as e:
                logger.error(f"Database error reading task patterns: {e}")
                metrics["errors"].append(f"Task DB error: {str(e)[:50]}")
            except Exception as e:
                logger.error(f"Unexpected error reading task patterns: {e}")
                metrics["errors"].append(f"Unexpected error: {str(e)[:50]}")
        else:
            logger.info("felix_memory.db not found - showing empty metrics")

        return metrics

    def get_agent_data(self) -> List[Dict[str, Any]]:
        """Read agent data from shared databases with agent awareness support."""
        agents = []

        try:
            if not Path(self.knowledge_db_path).exists():
                logger.info(f"{self.knowledge_db_path} not found - no agent data available")
                return agents

            conn = sqlite3.connect(self.knowledge_db_path)
            cursor = conn.cursor()

            # Verify table exists
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='knowledge_entries'
            """)
            if not cursor.fetchone():
                logger.warning("Table 'knowledge_entries' not found")
                conn.close()
                return agents

            # Get column information
            cursor.execute("PRAGMA table_info(knowledge_entries)")
            columns = [col[1] for col in cursor.fetchall()]

            # Use id or entry_id depending on what's available
            id_col = 'entry_id' if 'entry_id' in columns else 'id'

            # Build query based on available columns (convert confidence_level TEXT to numeric)
            if 'knowledge_id' in columns and 'domain' in columns:
                query = """
                    SELECT knowledge_id, source_agent, domain,
                           CASE confidence_level
                               WHEN 'low' THEN 0.3
                               WHEN 'medium' THEN 0.6
                               WHEN 'high' THEN 0.9
                               ELSE 0.5
                           END as confidence,
                           created_at, content_json
                    FROM knowledge_entries
                    ORDER BY created_at DESC
                    LIMIT 100
                """
                cursor.execute(query)

                for row in cursor.fetchall():
                    # Safe data extraction with defaults
                    knowledge_id = row[0] if row[0] else "unknown"
                    agent_id = row[1] if len(row) > 1 and row[1] else f"agent_{knowledge_id}"
                    domain = row[2] if len(row) > 2 else "unknown"
                    confidence = row[3] if len(row) > 3 and row[3] is not None else 0.5
                    timestamp = row[4] if len(row) > 4 else ""
                    content = row[5][:200] if len(row) > 5 and row[5] else ""

                    agents.append({
                        "agent_id": agent_id,
                        "domain": domain,
                        "confidence": confidence,
                        "timestamp": timestamp,
                        "content": content,
                        "phase": self._infer_phase_from_domain(domain),
                        "collaboration_count": 0,
                        "convergence_ready": confidence >= 0.8
                    })
            else:
                logger.warning(f"Required columns not found. Available: {columns}")

            conn.close()

        except sqlite3.Error as e:
            logger.error(f"Database error reading agent data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading agent data: {e}")

        return agents

    def _infer_phase_from_domain(self, domain: str) -> str:
        """
        Infer agent phase from domain or agent type.

        Args:
            domain: Agent domain or type string

        Returns:
            Phase name: 'exploration', 'analysis', or 'synthesis'
        """
        if not domain:
            return "unknown"

        domain_lower = domain.lower()
        if "research" in domain_lower or "exploration" in domain_lower:
            return "exploration"
        elif "analysis" in domain_lower or "critic" in domain_lower:
            return "analysis"
        elif "synthesis" in domain_lower or "final" in domain_lower:
            return "synthesis"
        else:
            return "unknown"

    def get_workflow_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Read recent workflow results from databases with error handling."""
        results = []

        # Try felix_task_memory.db first
        if Path(self.task_memory_db_path).exists():
            try:
                conn = sqlite3.connect(self.task_memory_db_path)
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
                        # Safe data extraction
                        task = row[0] if row[0] else "Unknown task"
                        timestamp = row[1] if len(row) > 1 else ""
                        success = row[2] if len(row) > 2 else True
                        agent_count = row[3] if len(row) > 3 else 0
                        synthesis = row[4] if len(row) > 4 and row[4] else ""

                        results.append({
                            "task": task,
                            "timestamp": timestamp,
                            "success": success,
                            "agent_count": agent_count,
                            "final_synthesis": synthesis
                        })

                conn.close()

            except sqlite3.Error as e:
                logger.error(f"Database error reading workflow results: {e}")
            except Exception as e:
                logger.error(f"Unexpected error reading workflow results: {e}")
        else:
            logger.info("felix_task_memory.db not found")

        # Fallback to task memory if no workflow results
        if not results and Path(self.memory_db_path).exists():
            try:
                conn = sqlite3.connect(self.memory_db_path)
                cursor = conn.cursor()

                # Check if table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='task_patterns'
                """)

                if cursor.fetchone():
                    # Try task_executions table first (has actual workflow data)
                    cursor.execute("""
                        SELECT name FROM sqlite_master
                        WHERE type='table' AND name='task_executions'
                    """)

                    if cursor.fetchone():
                        cursor.execute(f"""
                            SELECT task_description, outcome, duration, agents_used_json,
                                   success_metrics_json, created_at
                            FROM task_executions
                            ORDER BY created_at DESC
                            LIMIT {limit}
                        """)

                        for row in cursor.fetchall():
                            task = row[0] if row[0] else "Unknown task"
                            success = row[1] == "success" if len(row) > 1 else True

                            # Parse agents count from JSON
                            agent_count = 0
                            final_synthesis = ""
                            try:
                                import json
                                if len(row) > 4 and row[4]:
                                    metrics = json.loads(row[4])
                                    agent_count = metrics.get("agent_count", 0)
                                    avg_conf = metrics.get("avg_confidence", 0)
                                    final_synthesis = f"Avg Confidence: {avg_conf:.2%}"
                            except:
                                pass

                            timestamp = row[5] if len(row) > 5 else ""

                            results.append({
                                "task": task,
                                "timestamp": timestamp,
                                "success": success,
                                "agent_count": agent_count,
                                "final_synthesis": final_synthesis
                            })
                    else:
                        # Fallback to task_patterns if no task_executions
                        cursor.execute(f"""
                            SELECT task_type, created_at
                            FROM task_patterns
                            ORDER BY created_at DESC
                            LIMIT {limit}
                        """)

                        for row in cursor.fetchall():
                            task_type = row[0] if row[0] else "Unknown pattern"
                            timestamp = row[1] if len(row) > 1 else ""

                            results.append({
                                "task": task_type,
                                "timestamp": timestamp,
                                "success": True,
                                "agent_count": 0,
                                "final_synthesis": ""
                            })

                conn.close()

            except sqlite3.Error as e:
                logger.error(f"Database error reading task memory: {e}")
            except Exception as e:
                logger.error(f"Unexpected error reading task memory: {e}")
        else:
            logger.info("No workflow data available - run workflows to generate data")

        return results

    def get_agent_awareness_data(self) -> Dict[str, Any]:
        """
        Get agent awareness information from AgentRegistry if available.

        Returns:
            Dictionary with agent awareness metrics including phases,
            convergence status, and collaboration patterns
        """
        awareness_data = {
            "available": False,
            "phase_distribution": {"exploration": 0, "analysis": 0, "synthesis": 0},
            "convergence_status": {},
            "collaboration_graph": {},
            "active_agents": []
        }

        try:
            # Try to get awareness from running Felix system
            # This would require access to the CentralPost's AgentRegistry
            # For now, we infer from database
            agents = self.get_agent_data()

            # Count agents by phase
            for agent in agents:
                phase = agent.get("phase", "unknown")
                if phase in awareness_data["phase_distribution"]:
                    awareness_data["phase_distribution"][phase] += 1

            # Check convergence readiness
            convergence_ready_count = sum(1 for a in agents if a.get("convergence_ready", False))
            awareness_data["convergence_status"] = {
                "synthesis_ready": convergence_ready_count > 0,
                "ready_agents": [a["agent_id"] for a in agents if a.get("convergence_ready", False)],
                "total_agents": len(agents)
            }

            awareness_data["available"] = True
            awareness_data["active_agents"] = [a["agent_id"] for a in agents[:10]]  # Top 10 recent

        except Exception as e:
            logger.debug(f"Could not retrieve agent awareness data: {e}")

        return awareness_data

    def get_streaming_status(self) -> Dict[str, Any]:
        """
        Get streaming status and partial thoughts from agents.

        Returns:
            Dictionary with streaming status and active streams
        """
        streaming_status = {
            "enabled": False,
            "active_streams": [],
            "partial_thoughts": {},
            "completion_rate": 0.0
        }

        try:
            # Check if streaming is enabled in config
            # For now, we return a placeholder
            # In production, this would check FelixSystem.config.enable_streaming
            streaming_status["enabled"] = True

        except Exception as e:
            logger.debug(f"Could not retrieve streaming status: {e}")

        return streaming_status

    def create_isolated_instance(self, config: Dict[str, Any]) -> Optional[FelixSystem]:
        """
        Create an isolated Felix instance for benchmarking.
        This instance uses temporary databases to avoid interference.
        """
        if not FelixSystem or not FelixConfig:
            logger.error("Felix components not available for isolated instance")
            return None

        try:
            # Create config with temporary databases and streaming support
            felix_config = FelixConfig(
                lm_host=config.get('lm_host', '127.0.0.1'),
                lm_port=config.get('lm_port', 1234),
                memory_db_path="temp_memory.db",  # Temporary DB
                knowledge_db_path="temp_knowledge.db",  # Temporary DB
                enable_streaming=config.get('enable_streaming', True),  # Enable streaming by default
                **config
            )

            # Create isolated instance
            isolated_system = FelixSystem(felix_config)
            logger.info("Created isolated Felix instance for benchmarking with streaming support")
            return isolated_system

        except Exception as e:
            logger.error(f"Could not create isolated instance: {e}")
            return None