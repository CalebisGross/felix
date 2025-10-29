"""Database reader module for Felix Framework."""

import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
import os
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DatabaseReader:
    """Safe read-only access to Felix databases with pandas DataFrames."""

    def __init__(self, db_dir: Optional[str] = None):
        """
        Initialize DatabaseReader.

        Args:
            db_dir: Directory containing databases. If None, uses project root.
        """
        # Default to project root (one level up from streamlit_gui/backend)
        if db_dir is None:
            backend_dir = Path(__file__).parent
            streamlit_gui_dir = backend_dir.parent
            db_dir = str(streamlit_gui_dir.parent)

        db_path = Path(db_dir)
        self.db_paths = {
            "knowledge": str(db_path / "felix_knowledge.db"),
            "memory": str(db_path / "felix_memory.db"),
            "task_memory": str(db_path / "felix_task_memory.db"),
            "workflow_history": str(db_path / "felix_workflow_history.db"),
            "live_agents": str(db_path / "felix_live_agents.db"),
            "system_actions": str(db_path / "felix_system_actions.db")
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

    def get_workflow_history(
        self,
        limit: int = 100,
        status_filter: Optional[str] = None,
        days_back: int = 7,
        search_query: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get workflow execution history.

        Args:
            limit: Maximum number of workflows to return
            status_filter: Filter by status ("completed", "failed", None for all)
            days_back: Number of days back to search
            search_query: Search term for task_input

        Returns:
            DataFrame with workflow history
        """
        # Build WHERE clause
        where_clauses = []

        if status_filter:
            where_clauses.append(f"status = '{status_filter}'")

        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            where_clauses.append(f"created_at >= '{cutoff_date.isoformat()}'")

        if search_query:
            where_clauses.append(f"task_input LIKE '%{search_query}%'")

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        query = f"""
            SELECT
                workflow_id,
                task_input,
                status,
                created_at,
                completed_at,
                confidence,
                agents_count,
                tokens_used,
                max_tokens,
                processing_time,
                temperature,
                final_synthesis
            FROM workflow_outputs
            {where_sql}
            ORDER BY created_at DESC
            LIMIT {limit}
        """

        df = self._read_query("workflow_history", query)
        if df is None:
            return pd.DataFrame(columns=[
                "workflow_id", "task_input", "status", "created_at",
                "completed_at", "confidence", "agents_count", "tokens_used",
                "max_tokens", "processing_time", "temperature", "final_synthesis"
            ])

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

    def get_workflow_by_id(self, workflow_id: int) -> Optional[Dict[str, Any]]:
        """
        Get complete workflow details by ID.

        Args:
            workflow_id: Workflow ID to retrieve

        Returns:
            Dictionary with workflow data or None
        """
        query = f"""
            SELECT *
            FROM workflow_outputs
            WHERE workflow_id = {workflow_id}
        """

        df = self._read_query("workflow_history", query)
        if df is None or df.empty:
            return None

        return df.iloc[0].to_dict()

    def get_workflow_stats(self) -> Dict[str, Any]:
        """
        Get aggregate workflow statistics.

        Returns:
            Dictionary with summary statistics
        """
        query = """
            SELECT
                COUNT(*) as total_count,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                AVG(confidence) as avg_confidence,
                AVG(processing_time) as avg_processing_time,
                AVG(agents_count) as avg_agents,
                AVG(tokens_used) as avg_tokens
            FROM workflow_outputs
        """

        df = self._read_query("workflow_history", query)
        if df is None or df.empty:
            return {
                'total_count': 0,
                'completed_count': 0,
                'failed_count': 0,
                'avg_confidence': 0.0,
                'avg_processing_time': 0.0,
                'avg_agents': 0,
                'avg_tokens': 0
            }

        return df.iloc[0].to_dict()

    def get_web_search_activity(self, limit: int = 50) -> pd.DataFrame:
        """
        Extract web search activity from knowledge entries.

        Args:
            limit: Maximum number of entries to analyze

        Returns:
            DataFrame with columns: agent_id, timestamp, query, sources, results_count
        """
        query = f"""
            SELECT
                source_agent,
                created_at,
                content_json
            FROM knowledge_entries
            WHERE domain = 'web_search'
            ORDER BY created_at DESC
            LIMIT {limit}
        """

        df = self._read_query("knowledge", query)
        if df is None or df.empty:
            return pd.DataFrame(columns=[
                "agent_id", "timestamp", "query", "sources", "results_count"
            ])

        # Parse content_json to extract search data
        search_data = []
        for _, row in df.iterrows():
            try:
                content = json.loads(row['content_json'])

                # Extract search queries
                search_queries = content.get('search_queries', [])
                information_sources = content.get('information_sources', [])
                web_search_results = content.get('web_search_results', [])

                # Handle case where search_queries might be empty
                if search_queries:
                    for query in search_queries:
                        search_data.append({
                            'agent_id': row['source_agent'],
                            'timestamp': row['created_at'],
                            'query': query,
                            'sources': ', '.join(information_sources) if information_sources else '',
                            'results_count': len(web_search_results)
                        })
                else:
                    # No specific queries, create single entry
                    search_data.append({
                        'agent_id': row['source_agent'],
                        'timestamp': row['created_at'],
                        'query': '',
                        'sources': ', '.join(information_sources) if information_sources else '',
                        'results_count': len(web_search_results)
                    })

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"Could not parse web search content_json: {e}")
                # Skip malformed entries
                continue

        if not search_data:
            return pd.DataFrame(columns=[
                "agent_id", "timestamp", "query", "sources", "results_count"
            ])

        return pd.DataFrame(search_data)

    def get_web_search_stats(self) -> Dict[str, Any]:
        """
        Get web search usage statistics.

        Returns:
            Dictionary with: total_searches, unique_queries, avg_results_per_search,
                            total_sources, searches_last_24h
        """
        df = self.get_web_search_activity(limit=1000)

        if df.empty:
            return {
                'total_searches': 0,
                'unique_queries': 0,
                'avg_results_per_search': 0.0,
                'total_sources': 0,
                'searches_last_24h': 0
            }

        # Calculate statistics
        total_searches = len(df)
        unique_queries = df['query'].nunique() if not df['query'].empty else 0
        avg_results = df['results_count'].mean() if 'results_count' in df.columns else 0.0

        # Count total unique sources
        all_sources = set()
        for sources_str in df['sources']:
            if sources_str:
                sources_list = [s.strip() for s in sources_str.split(',')]
                all_sources.update(sources_list)
        total_sources = len(all_sources)

        # Filter last 24 hours
        try:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            cutoff_time = datetime.now() - timedelta(hours=24)
            searches_last_24h = len(df[df['timestamp_dt'] >= cutoff_time])
        except Exception as e:
            logger.debug(f"Could not filter by last 24h: {e}")
            searches_last_24h = 0

        return {
            'total_searches': total_searches,
            'unique_queries': unique_queries,
            'avg_results_per_search': round(avg_results, 2),
            'total_sources': total_sources,
            'searches_last_24h': searches_last_24h
        }

    def get_live_agents(self, max_age_seconds: float = 5.0) -> List[Dict[str, Any]]:
        """
        Get currently active agents from live tracking database.

        Args:
            max_age_seconds: Maximum age of agent data to include (default: 5 seconds)

        Returns:
            List of agent dictionaries with position and status data
        """
        import time

        db_path = self.db_paths.get("live_agents")

        # Check if database exists
        if not db_path or not Path(db_path).exists():
            logger.debug(f"Live agents database not found: {db_path}")
            return []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cutoff_time = time.time() - max_age_seconds

            cursor.execute("""
                SELECT agent_id, agent_type, phase, progress,
                       x_position, y_position, z_position,
                       confidence, last_update, status
                FROM live_agents
                WHERE last_update > ?
                  AND status = 'active'
                ORDER BY last_update DESC
            """, (cutoff_time,))

            agents = []
            for row in cursor.fetchall():
                agents.append({
                    'id': row[0],
                    'type': row[1],
                    'phase': row[2],
                    'progress': row[3],
                    'x': row[4],
                    'y': row[5],
                    'z': row[6],
                    'confidence': row[7],
                    'last_update': row[8],
                    'status': row[9]
                })

            conn.close()
            logger.debug(f"Retrieved {len(agents)} live agents from database")
            return agents

        except Exception as e:
            logger.error(f"Error retrieving live agents: {e}")
            return []