"""
System Monitor Page for Felix Framework (READ-ONLY)

Monitors system command execution WITHOUT control capabilities.
For approvals and control, use the tkinter GUI.

Features:
- View pending approvals (read-only)
- Monitor live terminal output
- Browse command history
- View system statistics

IMPORTANT: This is a monitoring interface only. All command approvals
and system control operations must be performed via the tkinter GUI.
"""

import streamlit as st
import pandas as pd
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.communication.central_post import CentralPost
from src.execution.command_history import CommandHistory

# Import view-only components
from streamlit_gui.components.live_terminal import render_live_terminal
from streamlit_gui.components.execution_viewer import render_execution_viewer

# Page configuration
st.set_page_config(
    page_title="System Monitor",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ System Command Monitor")
st.markdown("**Read-only** monitoring of system command execution")

# Notice about view-only mode
st.info("ℹ️ **Monitoring Mode**: This interface is read-only. To approve commands or control system operations, use the tkinter GUI.")

# Initialize connections
@st.cache_resource
def get_central_post():
    """Get or create CentralPost instance."""
    try:
        return CentralPost()
    except Exception as e:
        st.error(f"Failed to initialize CentralPost: {e}")
        return None

@st.cache_resource
def get_command_history():
    """Get CommandHistory instance."""
    try:
        db_path = Path("felix_system_actions.db")
        if not db_path.exists():
            st.warning("System actions database not found. Run a workflow to initialize.")
            return None
        return CommandHistory(str(db_path))
    except Exception as e:
        st.error(f"Failed to initialize CommandHistory: {e}")
        return None

# Initialize components
central_post = get_central_post()
command_history = get_command_history()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "⏳ Pending Approvals",
    "⚡ Live Terminal",
    "📜 Command History",
    "📊 System Statistics"
])

# Tab 1: View Pending Approvals (READ-ONLY)
with tab1:
    st.subheader("Pending Command Approvals (View Only)")

    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=True, key="approval_auto_refresh")
    with col2:
        refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 2, key="approval_refresh_interval")
    with col3:
        if st.button("🔄 Refresh Now", key="approval_refresh_now"):
            st.rerun()

    # Approval queue container (VIEW ONLY)
    if central_post:
        try:
            pending = central_post.get_pending_actions()

            if not pending:
                st.success("✅ No pending approvals")
            else:
                st.warning(f"⚠️ {len(pending)} command(s) awaiting approval in tkinter GUI")

                # Display each pending approval as INFO ONLY
                for idx, action in enumerate(pending):
                    with st.expander(f"📋 Command #{idx+1}: `{action.get('command', 'Unknown')}`", expanded=False):
                        # Command details
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown("**Command:**")
                            st.code(action.get('command', 'N/A'), language='bash')
                            st.markdown(f"**Agent:** {action.get('agent_id', 'Unknown')}")
                            st.markdown(f"**Context:** {action.get('context', 'No context provided')}")

                            # Working directory if specified
                            if action.get('cwd'):
                                st.markdown(f"**Working Directory:** `{action.get('cwd')}`")

                        with col2:
                            st.markdown("**Risk Assessment:**")
                            risk_score = action.get('risk_score', 0.5)
                            trust_level = action.get('trust_level', 'REVIEW')

                            # Risk score visualization
                            if risk_score < 0.3:
                                risk_color = "green"
                                risk_label = "Low Risk"
                            elif risk_score < 0.7:
                                risk_color = "orange"
                                risk_label = "Medium Risk"
                            else:
                                risk_color = "red"
                                risk_label = "High Risk"

                            st.markdown(f"**Trust Level:** {trust_level}")
                            st.progress(risk_score)
                            st.markdown(f"**Risk:** :{risk_color}[{risk_label} ({risk_score:.2f})]")

                            # Timestamp
                            timestamp = action.get('timestamp')
                            if timestamp:
                                dt = datetime.fromtimestamp(timestamp)
                                st.markdown(f"**Requested:** {dt.strftime('%Y-%m-%d %H:%M:%S')}")

                        # Notice about where to approve
                        st.info("💡 **To approve/deny this command**, open the tkinter GUI → System Control tab")

        except Exception as e:
            st.error(f"Error loading approval queue: {e}")

        # Auto-refresh logic
        if auto_refresh and pending:
            time.sleep(refresh_interval)
            st.rerun()
    else:
        st.error("CentralPost not available. Please ensure Felix system is properly initialized.")

# Tab 2: Live Terminal with Real-time Command Output
with tab2:
    st.subheader("⚡ Live Command Execution")

    st.markdown("""
    Monitor commands as they execute in real-time. This view shows:
    - Active commands with live output streaming
    - Recently completed commands (last 30 seconds)
    - Full stdout/stderr in terminal-style display

    **Note:** This is read-only monitoring. Commands are executed and approved via tkinter GUI.
    """)

    # Render the live terminal component
    render_live_terminal(
        db_path="felix_system_actions.db",
        auto_refresh=True,
        refresh_interval=2.0
    )

# Tab 3: Command History with Detailed Execution Viewer
with tab3:
    st.subheader("Command Execution History")

    st.markdown("""
    Browse historical command executions with full details.
    Select any execution to view comprehensive information including:
    - Execution metadata and timestamps
    - Full command and context
    - Complete stdout/stderr output
    - Environment info and approval details
    """)

    # Render the execution viewer component with selector
    render_execution_viewer(
        db_path="felix_system_actions.db",
        show_selector=True
    )

# Tab 4: System Statistics
with tab4:
    st.subheader("System Command Statistics")

    if command_history:
        try:
            # Get statistics from database
            conn = sqlite3.connect("felix_system_actions.db")

            # Trust level distribution
            trust_query = """
            SELECT trust_level, COUNT(*) as count
            FROM command_executions
            GROUP BY trust_level
            """
            trust_df = pd.read_sql_query(trust_query, conn)

            # Success rate by trust level
            success_query = """
            SELECT
                trust_level,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures,
                COUNT(*) as total
            FROM command_executions
            WHERE executed = 1
            GROUP BY trust_level
            """
            success_df = pd.read_sql_query(success_query, conn)

            # Top agents by command count
            agent_query = """
            SELECT agent_id, COUNT(*) as command_count
            FROM command_executions
            GROUP BY agent_id
            ORDER BY command_count DESC
            LIMIT 10
            """
            agent_df = pd.read_sql_query(agent_query, conn)

            # Command timeline
            timeline_query = """
            SELECT
                DATE(timestamp, 'unixepoch') as date,
                COUNT(*) as count,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
            FROM command_executions
            WHERE executed = 1
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
            """
            timeline_df = pd.read_sql_query(timeline_query, conn)

            conn.close()

            # Create visualizations
            col1, col2 = st.columns(2)

            with col1:
                # Trust level pie chart
                if not trust_df.empty:
                    fig_trust = px.pie(
                        trust_df,
                        values='count',
                        names='trust_level',
                        title="Commands by Trust Level",
                        color_discrete_map={
                            'SAFE': '#2ecc71',
                            'REVIEW': '#f39c12',
                            'BLOCKED': '#e74c3c'
                        }
                    )
                    st.plotly_chart(fig_trust, use_container_width=True)
                else:
                    st.info("No trust level data available")

                # Top agents bar chart
                if not agent_df.empty:
                    fig_agents = px.bar(
                        agent_df.head(10),
                        x='command_count',
                        y='agent_id',
                        orientation='h',
                        title="Top 10 Agents by Command Count",
                        labels={'command_count': 'Commands', 'agent_id': 'Agent'}
                    )
                    st.plotly_chart(fig_agents, use_container_width=True)
                else:
                    st.info("No agent data available")

            with col2:
                # Success rate by trust level
                if not success_df.empty:
                    success_df['success_rate'] = (success_df['successes'] / success_df['total'] * 100).round(1)
                    fig_success = px.bar(
                        success_df,
                        x='trust_level',
                        y='success_rate',
                        title="Success Rate by Trust Level",
                        labels={'success_rate': 'Success Rate (%)', 'trust_level': 'Trust Level'},
                        color='trust_level',
                        color_discrete_map={
                            'SAFE': '#2ecc71',
                            'REVIEW': '#f39c12',
                            'BLOCKED': '#e74c3c'
                        }
                    )
                    fig_success.update_layout(showlegend=False)
                    st.plotly_chart(fig_success, use_container_width=True)
                else:
                    st.info("No success rate data available")

                # Command timeline
                if not timeline_df.empty:
                    timeline_df['date'] = pd.to_datetime(timeline_df['date'])
                    timeline_df['success_rate'] = (timeline_df['successes'] / timeline_df['count'] * 100).round(1)

                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        x=timeline_df['date'],
                        y=timeline_df['count'],
                        mode='lines+markers',
                        name='Total Commands',
                        line=dict(color='#3498db')
                    ))
                    fig_timeline.add_trace(go.Scatter(
                        x=timeline_df['date'],
                        y=timeline_df['successes'],
                        mode='lines+markers',
                        name='Successful',
                        line=dict(color='#2ecc71')
                    ))
                    fig_timeline.update_layout(
                        title="Command Execution Timeline",
                        xaxis_title="Date",
                        yaxis_title="Commands",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                else:
                    st.info("No timeline data available")

            st.divider()
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_commands = trust_df['count'].sum() if not trust_df.empty else 0
                st.metric("Total Commands", total_commands)

            with col2:
                if not success_df.empty:
                    overall_success = success_df['successes'].sum() / success_df['total'].sum() * 100
                    st.metric("Overall Success Rate", f"{overall_success:.1f}%")
                else:
                    st.metric("Overall Success Rate", "N/A")

            with col3:
                active_agents = len(agent_df) if not agent_df.empty else 0
                st.metric("Active Agents", active_agents)

            with col4:
                if not trust_df.empty:
                    safe_commands = trust_df[trust_df['trust_level'] == 'SAFE']['count'].sum() if 'SAFE' in trust_df['trust_level'].values else 0
                    safe_pct = (safe_commands / total_commands * 100) if total_commands > 0 else 0
                    st.metric("Safe Commands", f"{safe_pct:.1f}%")
                else:
                    st.metric("Safe Commands", "N/A")

        except Exception as e:
            st.error(f"Error loading statistics: {e}")
    else:
        st.warning("Statistics database not available. Run system commands to generate statistics.")

# Footer
st.markdown("---")
st.caption("System Monitor Dashboard - Felix Framework | Read-only monitoring interface")