"""Web Search Monitor component for Streamlit GUI."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any


class WebSearchMonitor:
    """Web search activity monitoring component with visualizations."""

    def __init__(self, db_reader):
        """Initialize monitor with database reader."""
        self.db_reader = db_reader

    def render(self):
        """Render the complete web search monitor."""
        st.header("🔍 Web Search Activity")
        st.markdown("Monitor Research agent web search queries and results")

        # Get stats
        stats = self.db_reader.get_web_search_stats()

        if stats['total_searches'] == 0:
            st.info("""
            No web search activity detected yet.

            Web search features:
            - Requires DuckDuckGo (ddgs package) or SearxNG
            - Research agents perform searches at helix depth 0.0-0.3
            - Enable in Felix GUI Settings > Web Search Configuration
            """)
            return

        # Summary Metrics
        self._render_summary_metrics(stats)

        st.divider()

        # Search Activity Table
        self._render_search_activity()

        st.divider()

        # Charts
        self._render_search_charts()

    def _render_summary_metrics(self, stats: Dict[str, Any]):
        """Render web search summary metrics."""
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Searches", stats['total_searches'])

        with col2:
            st.metric("Unique Queries", stats['unique_queries'])

        with col3:
            st.metric("Avg Results/Search", f"{stats['avg_results_per_search']:.1f}")

        with col4:
            st.metric("Searches (24h)", stats['searches_last_24h'])

    def _render_search_activity(self):
        """Render recent search activity table."""
        st.subheader("Recent Search Queries")

        limit = st.slider("Show last N searches", 10, 100, 50)

        df = self.db_reader.get_web_search_activity(limit=limit)

        if df.empty:
            st.info("No search activity found")
            return

        # Format for display
        display_df = df.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['sources_preview'] = display_df['sources'].apply(
            lambda x: ', '.join(x[:2]) + ('...' if len(x) > 2 else '') if isinstance(x, list) else ''
        )

        # Show table
        st.dataframe(
            display_df[['agent_id', 'timestamp', 'query', 'results_count', 'sources_preview']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "agent_id": "Agent",
                "timestamp": "Time",
                "query": st.column_config.TextColumn("Search Query", width="large"),
                "results_count": "Results",
                "sources_preview": st.column_config.TextColumn("Sources", width="medium")
            }
        )

    def _render_search_charts(self):
        """Render web search analytics charts."""
        st.subheader("Search Analytics")

        df = self.db_reader.get_web_search_activity(limit=200)

        if df.empty:
            return

        tab1, tab2, tab3 = st.tabs([
            "Search Frequency",
            "Results Distribution",
            "Top Queries"
        ])

        with tab1:
            # Searches over time
            df_time = df.copy()
            df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
            df_time['date'] = df_time['timestamp'].dt.date

            daily_counts = df_time.groupby('date').size().reset_index(name='count')

            fig = px.bar(
                daily_counts,
                x='date',
                y='count',
                title="Searches Per Day",
                labels={'date': 'Date', 'count': 'Number of Searches'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Results count distribution
            fig = px.histogram(
                df,
                x='results_count',
                title="Search Results Distribution",
                nbins=20,
                labels={'results_count': 'Results per Search', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            # Top queries
            top_queries = df['query'].value_counts().head(10)

            fig = px.bar(
                x=top_queries.values,
                y=top_queries.index,
                orientation='h',
                title="Top 10 Search Queries",
                labels={'x': 'Frequency', 'y': 'Query'}
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
