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

logger = logging.getLogger(__name__)


class DatabaseReader:
    """
    Safe read-only access to Felix databases.
    Provides pandas DataFrames for easy visualization.
    """

    def __init__(self):
        self.db_paths = {
            "knowledge": "felix_knowledge.db",
            "memory": "felix_memory.db",
            "task_memory": "felix_task_memory.db"
        }

    def _read_query(self, db_name: str, query: str) -> Optional[pd.DataFrame]:
        """Execute read-only query and return DataFrame."""
        db_path = self.db_paths.get(db_name)
        if not db_path or not Path(db_path).exists():
            return None

        try:
            # Use read-only connection
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df

        except Exception as e:
            logger.debug(f"Query failed on {db_name}: {e}")
            # Fallback to regular connection if read-only fails
            try:
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()
                return df
            except Exception as e2:
                logger.debug(f"Fallback query also failed: {e2}")
                return None

    def get_knowledge_entries(self, limit: int = 100) -> pd.DataFrame:
        """Get knowledge entries as DataFrame."""
        query = f"""
            SELECT
                id,
                agent_id,
                domain,
                confidence,
                timestamp,
                substr(content, 1, 200) as content_preview
            FROM knowledge
            ORDER BY timestamp DESC
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
                id,
                task,
                pattern,
                frequency,
                timestamp
            FROM tasks
            ORDER BY timestamp DESC
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
                agent_id,
                COUNT(*) as output_count,
                AVG(confidence) as avg_confidence,
                MIN(confidence) as min_confidence,
                MAX(confidence) as max_confidence,
                MIN(timestamp) as first_seen,
                MAX(timestamp) as last_seen
            FROM knowledge
            WHERE agent_id IS NOT NULL
            GROUP BY agent_id
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
                datetime(timestamp, 'unixepoch') as time,
                COUNT(*) as entries,
                AVG(confidence) as avg_confidence
            FROM knowledge
            WHERE timestamp > (strftime('%s', 'now') - {hours * 3600})
            GROUP BY strftime('%Y-%m-%d %H', datetime(timestamp, 'unixepoch'))
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
                task,
                timestamp,
                pattern,
                1 as success
            FROM tasks
            ORDER BY timestamp DESC
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
                AVG(confidence) as avg_confidence
            FROM knowledge
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
                timestamp,
                agent_id,
                confidence
            FROM knowledge
            WHERE timestamp > (strftime('%s', 'now') - {minutes * 60})
            ORDER BY timestamp DESC
            LIMIT 100
        """
        df_knowledge = self._read_query("knowledge", query)

        query = f"""
            SELECT
                'task' as source,
                timestamp,
                task as agent_id,
                1.0 as confidence
            FROM tasks
            WHERE timestamp > (strftime('%s', 'now') - {minutes * 60})
            ORDER BY timestamp DESC
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