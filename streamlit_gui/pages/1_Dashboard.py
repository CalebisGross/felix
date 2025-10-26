"""Dashboard page for Felix Framework monitoring."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)
import pandas as pd
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit_gui.backend.system_monitor import SystemMonitor
from streamlit_gui.backend.db_reader import DatabaseReader

st.set_page_config(
    page_title="Felix Dashboard",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def get_monitor():
    """Get cached SystemMonitor instance."""
    return SystemMonitor()

@st.cache_resource
def get_db_reader():
    """Get cached DatabaseReader instance."""
    return DatabaseReader()

def main():
    st.title("🏠 Felix System Dashboard")
    st.markdown("Real-time monitoring of Felix Framework")

    # Real data indicator
    st.success("✅ **Real Data**: This dashboard displays live metrics from Felix databases. All data is pulled from actual system execution.")

    monitor = get_monitor()
    db_reader = get_db_reader()

    # System Status Header
    col1, col2, col3, col4 = st.columns(4)

    metrics = monitor.get_system_metrics()

    with col1:
        status = "🟢 Running" if metrics["felix_running"] else "🔴 Stopped"
        st.metric("System Status", status, help="Felix system running status (checks LM Studio port 1234)")

    with col2:
        st.metric("Knowledge Entries", metrics["knowledge_entries"],
                 delta=f"+{metrics.get('new_entries', 0)}" if metrics.get('new_entries', 0) > 0 else None,
                 help="Total entries in knowledge database from all agents")

    with col3:
        st.metric("Task Patterns", metrics["task_patterns"],
                 help="Number of task patterns identified in memory database")

    with col4:
        confidence_pct = metrics['confidence_avg'] * 100 if metrics['confidence_avg'] > 0 else 0
        st.metric("Avg Confidence", f"{confidence_pct:.1f}%",
                 help="Average confidence score across all agent knowledge entries")

    st.divider()

    # Web Search Activity Metrics
    st.subheader("🔍 Web Search Activity")
    web_stats = db_reader.get_web_search_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Searches", web_stats.get("total_searches", 0),
                 help="Total number of web searches performed")

    with col2:
        st.metric("Last 24h", web_stats.get("searches_24h", 0),
                 help="Searches performed in the last 24 hours")

    with col3:
        st.metric("Unique Queries", web_stats.get("unique_queries", 0),
                 help="Number of unique search queries")

    with col4:
        st.metric("Total Sources", web_stats.get("total_sources", 0),
                 help="Total number of sources retrieved from searches")

    st.divider()

    # Real-time Metrics
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Agent Activity",
        "📈 Performance Trends",
        "🔄 Recent Workflows",
        "💾 Database Status",
        "📜 Workflow History",
        "🔍 Web Search"
    ])

    with tab1:
        st.subheader("Agent Activity Monitor")

        # Get agent data
        agent_df = db_reader.get_agent_metrics()

        if not agent_df.empty:
            # Agent performance scatter plot
            fig = px.scatter(
                agent_df,
                x="output_count",
                y="avg_confidence",
                size="output_count",
                color="avg_confidence",
                hover_data=["agent_id"],
                title="Agent Performance Overview",
                labels={
                    "output_count": "Number of Outputs",
                    "avg_confidence": "Average Confidence"
                },
                color_continuous_scale="viridis"
            )
            fig.update_layout(height=400)
            logger.info(f"DEBUG: About to call st.plotly_chart with fig type: {type(fig)}")
            st.plotly_chart(fig, width='stretch')

            # Agent details table
            st.subheader("Agent Details")

            # Format the dataframe for display
            display_df = agent_df[["agent_id", "output_count", "avg_confidence", "last_seen"]].copy()

            # Format confidence as percentage
            display_df["avg_confidence"] = display_df["avg_confidence"].apply(
                lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A"
            )

            # Format timestamp
            if "last_seen" in display_df.columns:
                display_df["last_seen"] = pd.to_datetime(display_df["last_seen"], unit='s', errors='coerce')
                display_df["last_seen"] = display_df["last_seen"].dt.strftime('%Y-%m-%d %H:%M:%S')

            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True
            )
        else:
            st.info("No agent activity recorded yet. Start Felix system to begin monitoring.")

    with tab2:
        st.subheader("Performance Trends")

        # Time series metrics
        ts_df = db_reader.get_time_series_metrics(hours=24)

        if not ts_df.empty:
            # Confidence trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(ts_df["time"]),
                y=ts_df["avg_confidence"],
                mode='lines+markers',
                name='Avg Confidence',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title="Confidence Trend (24 Hours)",
                xaxis_title="Time",
                yaxis_title="Average Confidence",
                hovermode='x',
                height=350
            )
            st.plotly_chart(fig, width='stretch')

            # Activity volume
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=pd.to_datetime(ts_df["time"]),
                y=ts_df["entries"],
                name='Entries',
                marker_color='lightblue'
            ))
            fig2.update_layout(
                title="Activity Volume",
                xaxis_title="Time",
                yaxis_title="Number of Entries",
                height=350
            )
            st.plotly_chart(fig2, width='stretch')
        else:
            st.info("No trend data available. Metrics will appear as system runs.")

    with tab3:
        st.subheader("Recent Workflow Results")

        workflows = monitor.get_workflow_results(limit=10)

        if workflows:
            for workflow in workflows:
                # Format timestamp
                try:
                    timestamp = pd.to_datetime(workflow['timestamp'], unit='s')
                    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_str = str(workflow['timestamp'])

                task_preview = workflow['task'][:50] + "..." if len(workflow['task']) > 50 else workflow['task']

                with st.expander(f"🔄 {task_preview} - {time_str}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        agent_count = workflow.get('agent_count', 0)
                        st.metric("Agents", agent_count if agent_count > 0 else "N/A")
                    with col2:
                        status = "✅ Success" if workflow.get('success', True) else "❌ Failed"
                        st.metric("Status", status)
                    with col3:
                        has_synthesis = bool(workflow.get('final_synthesis'))
                        st.metric("Synthesis", "Yes" if has_synthesis else "No")

                    if workflow.get('final_synthesis'):
                        st.text_area("Final Synthesis", workflow['final_synthesis'], height=100, disabled=True)
        else:
            st.info("No workflow results available. Run workflows from tkinter GUI to see results here.")

    with tab4:
        st.subheader("Database Status")

        # Database sizes and info
        db_stats = db_reader.get_database_stats()

        db_info = []
        for db_name, stats in db_stats.items():
            db_info.append({
                "Database": db_name.title(),
                "Status": "🟢 Connected" if stats["exists"] else "🔴 Not Found",
                "Size (MB)": f"{stats['size_mb']:.2f}" if stats["exists"] else "N/A",
                "Tables": stats.get("table_count", 0),
                "Total Rows": stats.get("total_rows", 0)
            })

        df_info = pd.DataFrame(db_info)
        st.dataframe(df_info, width='stretch', hide_index=True)

        # Domain distribution if available
        domain_df = db_reader.get_domain_distribution()
        if not domain_df.empty:
            st.subheader("Knowledge Distribution by Domain")

            fig = px.pie(
                domain_df,
                values='count',
                names='domain',
                title='Domain Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=350)
            st.plotly_chart(fig, width='stretch')

        # Connection info
        st.info(
            "ℹ️ **Database Access Mode**: Read-Only\n\n"
            "The Streamlit GUI monitors shared databases without modifying them. "
            "All write operations are performed by the tkinter GUI and Felix system."
        )

    with tab5:
        from streamlit_gui.components.workflow_history_viewer import WorkflowHistoryViewer

        history_viewer = WorkflowHistoryViewer(db_reader)
        history_viewer.render()

    with tab6:
        from streamlit_gui.components.web_search_monitor import WebSearchMonitor

        search_monitor = WebSearchMonitor(db_reader)
        search_monitor.render()

    # Auto-refresh option
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    with col2:
        if auto_refresh:
            st.write("Dashboard will refresh every 5 seconds")
            import time
            time.sleep(5)
            st.rerun()

if __name__ == "__main__":
    main()