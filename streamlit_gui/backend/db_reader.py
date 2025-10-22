"""
Database reader module for Felix Framework.

Provides safe read-only access to Felix databases with pandas DataFrames
for easy visualization in Streamlit.
"""

import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)


class DatabaseReader:
    """
    Safe read-only access to Felix databases.
    Provides pandas DataFrames for easy visualization.
    """

    def __init__(self, db_dir: Optional[str] = None):
        """
        Initialize DatabaseReader.

        Args:
            db_dir: Directory containing databases. If None, uses project root.
        """
        # Default to project root (one level up from streamlit_gui/backend)
        if db_dir is None:
            import os
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            streamlit_gui_dir = os.path.dirname(backend_dir)
            db_dir = os.path.dirname(streamlit_gui_dir)

        self.db_paths = {
            "knowledge": os.path.join(db_dir, "felix_knowledge.db"),
            "memory": os.path.join(db_dir, "felix_memory.db"),
            "task_memory": os.path.join(db_dir, "felix_task_memory.db")
        }

    def _read_query(self, db_name: str, query: str) -> Optional[pd.DataFrame]:
        """Execute read-only query and return DataFrame with comprehensive error handling."""
        db_path = self.db_paths.get(db_name)

        # Check if database exists
        if not db_path:
            logger.warning(f"No path configured for database: {db_name}")
            return None

        if not Path(db_path).exists():
            logger.info(f"Database not found: {db_path}")
            return None

        try:
            # Try read-only connection first
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df

        except sqlite3.OperationalError as e:
            logger.debug(f"Read-only mode not supported for {db_name}, trying regular mode: {e}")
            # Fallback to regular connection if read-only fails
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df
            except sqlite3.DatabaseError as e2:
                logger.error(f"Database error on {db_name}: {e2}")
                return None
            except Exception as e2:
                logger.error(f"Unexpected error querying {db_name}: {e2}")
                return None

        except pd.errors.DatabaseError as e:
            logger.error(f"Pandas database error on {db_name}: {e}")
            return None

        except Exception as e:
            logger.error(f"Unexpected error reading {db_name}: {e}")
            return None

    def get_knowledge_entries(self, limit: int = 100) -> pd.DataFrame:
        """Get knowledge entries as DataFrame."""
        query = f"""
            SELECT
                knowledge_id as id,
                source_agent as agent_id,
                domain,
                CASE confidence_level
                    WHEN 'low' THEN 0.3
                    WHEN 'medium' THEN 0.6
                    WHEN 'high' THEN 0.9
                    ELSE 0.5
                END as confidence,
                created_at as timestamp,
                substr(content_json, 1, 200) as content_preview
            FROM knowledge_entries
            ORDER BY created_at DESC
            LIMIT {limit}
        """
        df = self._read_query("knowledge", query)
        if df is None:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "id", "agent_id", "domain", "confidence",
                "timestamp", "content_preview"
            ])
        return df

    def get_task_patterns(self, limit: int = 100) -> pd.DataFrame:
        """Get task patterns as DataFrame."""
        query = f"""
            SELECT
                pattern_id as id,
                task_type as task,
                complexity as pattern,
                usage_count as frequency,
                created_at as timestamp
            FROM task_patterns
            ORDER BY created_at DESC
            LIMIT {limit}
        """
        df = self._read_query("memory", query)
        if df is None:
            return pd.DataFrame(columns=[
                "id", "task", "pattern", "frequency", "timestamp"
            ])
        return df

    def get_agent_metrics(self) -> pd.DataFrame:
        """Aggregate agent performance metrics."""
        query = """
            SELECT
                source_agent as agent_id,
                COUNT(*) as output_count,
                AVG(CASE confidence_level
                    WHEN 'low' THEN 0.3
                    WHEN 'medium' THEN 0.6
                    WHEN 'high' THEN 0.9
                    ELSE 0.5
                END) as avg_confidence,
                MIN(CASE confidence_level
                    WHEN 'low' THEN 0.3
                    WHEN 'medium' THEN 0.6
                    WHEN 'high' THEN 0.9
                    ELSE 0.5
                END) as min_confidence,
                MAX(CASE confidence_level
                    WHEN 'low' THEN 0.3
                    WHEN 'medium' THEN 0.6
                    WHEN 'high' THEN 0.9
                    ELSE 0.5
                END) as max_confidence,
                MIN(created_at) as first_seen,
                MAX(created_at) as last_seen
            FROM knowledge_entries
            WHERE source_agent IS NOT NULL
            GROUP BY source_agent
            ORDER BY output_count DESC
        """
        df = self._read_query("knowledge", query)
        if df is None:
            return pd.DataFrame(columns=[
                "agent_id", "output_count", "avg_confidence",
                "min_confidence", "max_confidence", "first_seen", "last_seen"
            ])
        return df

    def get_time_series_metrics(self, hours: int = 24) -> pd.DataFrame:
        """Get time-series metrics for visualization."""
        query = f"""
            SELECT
                datetime(created_at, 'unixepoch') as time,
                COUNT(*) as entries,
                AVG(CASE confidence_level
                    WHEN 'low' THEN 0.3
                    WHEN 'medium' THEN 0.6
                    WHEN 'high' THEN 0.9
                    ELSE 0.5
                END) as avg_confidence
            FROM knowledge_entries
            WHERE created_at > (strftime('%s', 'now') - {hours * 3600})
            GROUP BY strftime('%Y-%m-%d %H', datetime(created_at, 'unixepoch'))
            ORDER BY time
        """
        df = self._read_query("knowledge", query)
        if df is None:
            return pd.DataFrame(columns=["time", "entries", "avg_confidence"])
        return df

    def get_workflow_history(self, limit: int = 50) -> pd.DataFrame:
        """Get workflow execution history."""
        # First try the task_memory database
        query = f"""
            SELECT
                task_type as task,
                created_at as timestamp,
                complexity as pattern,
                success_rate as success
            FROM task_patterns
            ORDER BY created_at DESC
            LIMIT {limit}
        """
        df = self._read_query("memory", query)
        if df is None:
            return pd.DataFrame(columns=["task", "timestamp", "pattern", "success"])
        return df

    def get_database_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all databases."""
        stats = {}

        for db_name, db_path in self.db_paths.items():
            if not Path(db_path).exists():
                stats[db_name] = {
                    "exists": False,
                    "size_mb": 0,
                    "table_count": 0,
                    "total_rows": 0
                }
                continue

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Get database size
                size_mb = Path(db_path).stat().st_size / (1024 * 1024)

                # Count tables
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM sqlite_master
                    WHERE type='table'
                """)
                table_count = cursor.fetchone()[0]

                # Count total rows across all tables
                total_rows = 0
                cursor.execute("""
                    SELECT name
                    FROM sqlite_master
                    WHERE type='table'
                """)
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    total_rows += row_count

                stats[db_name] = {
                    "exists": True,
                    "size_mb": size_mb,
                    "table_count": table_count,
                    "total_rows": total_rows
                }

                conn.close()

            except Exception as e:
                logger.debug(f"Could not get stats for {db_name}: {e}")
                stats[db_name] = {
                    "exists": True,
                    "size_mb": Path(db_path).stat().st_size / (1024 * 1024),
                    "table_count": 0,
                    "total_rows": 0
                }

        return stats

    def get_domain_distribution(self) -> pd.DataFrame:
        """Get distribution of knowledge entries by domain."""
        query = """
            SELECT
                domain,
                COUNT(*) as count,
                AVG(CASE confidence_level
                    WHEN 'low' THEN 0.3
                    WHEN 'medium' THEN 0.6
                    WHEN 'high' THEN 0.9
                    ELSE 0.5
                END) as avg_confidence
            FROM knowledge_entries
            WHERE domain IS NOT NULL
            GROUP BY domain
            ORDER BY count DESC
        """
        df = self._read_query("knowledge", query)
        if df is None:
            return pd.DataFrame(columns=["domain", "count", "avg_confidence"])
        return df

    def get_recent_activity(self, minutes: int = 60) -> pd.DataFrame:
        """Get recent activity across all databases."""
        query = f"""
            SELECT
                'knowledge' as source,
                created_at as timestamp,
                source_agent as agent_id,
                CASE confidence_level
                    WHEN 'low' THEN 0.3
                    WHEN 'medium' THEN 0.6
                    WHEN 'high' THEN 0.9
                    ELSE 0.5
                END as confidence
            FROM knowledge_entries
            WHERE created_at > (strftime('%s', 'now') - {minutes * 60})
            ORDER BY created_at DESC
            LIMIT 100
        """
        df_knowledge = self._read_query("knowledge", query)

        query = f"""
            SELECT
                'task' as source,
                created_at as timestamp,
                task_type as agent_id,
                success_rate as confidence
            FROM task_patterns
            WHERE created_at > (strftime('%s', 'now') - {minutes * 60})
            ORDER BY created_at DESC
            LIMIT 100
        """
        df_tasks = self._read_query("memory", query)

        # Combine dataframes
        dfs = []
        if df_knowledge is not None:
            dfs.append(df_knowledge)
        if df_tasks is not None:
            dfs.append(df_tasks)

        if dfs:
            return pd.concat(dfs, ignore_index=True).sort_values('timestamp', ascending=False)
        else:
            return pd.DataFrame(columns=["source", "timestamp", "agent_id", "confidence"])