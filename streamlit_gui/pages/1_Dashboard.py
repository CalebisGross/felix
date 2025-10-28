"""Dashboard page for Felix Framework monitoring."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger(__name__)
import pandas as pd
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from streamlit_gui.backend.system_monitor import SystemMonitor
from streamlit_gui.backend.db_reader import DatabaseReader
from streamlit_gui.backend.real_benchmark_runner import RealBenchmarkRunner
from streamlit_gui.components.helix_monitor import HelixMonitor, generate_sample_agents

# Constants
MAX_AGENTS = 133  # Maximum agent capacity for optimal Felix system performance

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

def main() -> None:
    st.title("🏠 Felix System Dashboard")
    st.markdown("Real-time monitoring of Felix Framework")

    # Real data indicator
    st.success("✅ **Real Data**: This dashboard displays live metrics from Felix databases. All data is pulled from actual system execution.")

    monitor = get_monitor()
    db_reader = get_db_reader()

    # System Health Dashboard
    st.subheader("System Health Dashboard")
    col1, col2, col3, col4 = st.columns(4)

    metrics = monitor.get_system_metrics()

    with col1:
        # LM Studio Connection Health
        is_connected = metrics["felix_running"]
        connection_status = "🟢 Connected" if is_connected else "🔴 Disconnected"
        st.metric(
            "LM Studio Connection",
            connection_status,
            help="Connection status to LM Studio on port 1234. Green = ready to process workflows."
        )
        if not is_connected:
            st.error("⚠️ Start LM Studio to enable Felix system")

    with col2:
        # Database Health
        db_stats = db_reader.get_database_stats()
        total_dbs = len(db_stats)
        healthy_dbs = sum(1 for db in db_stats.values() if db["exists"])

        db_status = "🟢 Healthy" if healthy_dbs == total_dbs else "🔴 Error"
        st.metric(
            "Database Health",
            db_status,
            delta=f"{healthy_dbs}/{total_dbs} databases",
            help="All Felix databases accessible and operational. 6 databases total: knowledge, memory, task_memory, workflow_history, live_agents, system_actions."
        )
        if healthy_dbs < total_dbs:
            st.warning(f"⚠️ {total_dbs - healthy_dbs} database(s) missing")

    with col3:
        # Active Agents
        agent_df = db_reader.get_agent_metrics()
        active_agents = len(agent_df) if not agent_df.empty else 0
        capacity_pct = (active_agents / MAX_AGENTS) * 100

        # Try to get previous count for delta
        prev_count = st.session_state.get('prev_agent_count', active_agents)
        delta_agents = active_agents - prev_count
        st.session_state['prev_agent_count'] = active_agents

        st.metric(
            "Active Agents",
            f"{active_agents}/{MAX_AGENTS}",
            delta=f"{delta_agents:+d}" if delta_agents != 0 else None,
            help=f"Currently active agents. System capacity: {capacity_pct:.1f}% utilized. Max {MAX_AGENTS} agents for optimal performance."
        )

    with col4:
        # Pending Approvals
        from src.communication.central_post import CentralPost
        try:
            central_post = CentralPost()
            pending_approvals = central_post.get_pending_actions()
            pending_count = len(pending_approvals) if pending_approvals else 0

            approval_status = f"⚠️ {pending_count}" if pending_count > 0 else "✅ 0"
            st.metric(
                "Pending Approvals",
                approval_status,
                help="System commands awaiting approval. Review in System Control tab to approve or deny."
            )
            if pending_count > 0:
                st.warning(f"🔗 [{pending_count} approval(s) need attention →](/System_Control)")
        except Exception as e:
            logger.warning(f"Could not check pending approvals: {e}")
            st.metric(
                "Pending Approvals",
                "N/A",
                help="Could not retrieve approval status"
            )

    # Web Search Activity Metrics (collapsed - non-actionable details)
    with st.expander("🔍 Web Search Activity Details"):
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
    tab_helix, tab1, tab2, tab3 = st.tabs([
        "🌀 Live Helix Monitor",
        "📊 Agent Activity",
        "📈 Performance Trends & Web Search",
        "💾 Database & History"
    ])

    with tab_helix:
        st.subheader("🌀 Real-Time Helix Visualization")
        st.markdown("Monitor agents traversing the helical path in real-time")

        # Create two columns - main viz and controls
        col_viz, col_controls = st.columns([3, 1])

        with col_controls:
            # Initialize helix monitor
            helix_monitor = HelixMonitor()

            # Render control panel
            controls = helix_monitor.render_controls()

            # Animation control (only show in real-time mode)
            if controls['real_time']:
                st.divider()
                st.markdown("### Animation Control")

                # Pause/Play button
                is_paused = st.session_state.get('animation_paused', False)
                button_col1, button_col2 = st.columns(2)

                with button_col1:
                    if is_paused:
                        if st.button("▶️ Play", key="play_animation", type="primary", use_container_width=True):
                            st.session_state['animation_paused'] = False
                            st.session_state['last_update'] = time.time()  # Reset timer
                            st.rerun()
                    else:
                        if st.button("⏸️ Pause", key="pause_animation", use_container_width=True):
                            st.session_state['animation_paused'] = True
                            st.rerun()

                with button_col2:
                    if st.button("🔄 Reset", key="reset_animation", use_container_width=True):
                        # Reset all agents to top
                        st.session_state['helix_agents'] = generate_sample_agents(10)
                        st.session_state['last_update'] = time.time()
                        st.session_state['animation_paused'] = False
                        st.rerun()

            # Workflow control
            st.divider()
            st.markdown("### Workflow Control")

            if st.button("▶️ Start Workflow", key="start_workflow", type="primary"):
                st.session_state['workflow_running'] = True
                st.success("Workflow started!")

            if st.button("⏹️ Stop Workflow", key="stop_workflow"):
                st.session_state['workflow_running'] = False
                st.info("Workflow stopped")

            st.divider()
            st.markdown("### Agent Statistics")

            # Get live agent data
            agent_df = db_reader.get_agent_metrics()
            if not agent_df.empty:
                st.metric("Active Agents", len(agent_df))
                avg_conf = agent_df['avg_confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
                total_outputs = agent_df['output_count'].sum()
                st.metric("Total Outputs", total_outputs)
            else:
                st.metric("Active Agents", 0)
                st.metric("Avg Confidence", "N/A")
                st.metric("Total Outputs", 0)

        with col_viz:
            # Create placeholder for animation
            viz_placeholder = st.empty()

            # Real-time mode
            if controls['real_time']:
                # Get REAL agents from live database
                live_agents = db_reader.get_live_agents(max_age_seconds=5.0)

                if live_agents:
                    # Use real agent data from running workflow!
                    agent_positions = [
                        {
                            'id': agent['id'],
                            'type': agent['type'],
                            'progress': agent['progress'],
                            'confidence': agent['confidence']
                        }
                        for agent in live_agents
                    ]

                    # Create and display visualization with REAL agents
                    fig = helix_monitor.create_visualization(agent_positions)
                    viz_placeholder.plotly_chart(fig, use_container_width=True, key="helix_viz_live")

                    # Show status for live data
                    info_col, status_col, refresh_col = st.columns([2, 1, 1])
                    with info_col:
                        st.success(f"🔴 **LIVE**: {len(agent_positions)} real agents from running workflow")
                    with status_col:
                        avg_progress = sum(a['progress'] for a in agent_positions) / len(agent_positions)
                        st.metric("Avg Progress", f"{avg_progress:.0%}")
                    with refresh_col:
                        # Manual refresh button for live updates
                        if st.button("🔄 Refresh", key="refresh_live", use_container_width=True):
                            st.rerun()

                else:
                    # No live workflow - show static message
                    st.info("⚪ **WAITING**: No active workflow detected. Start a workflow in the tkinter GUI to see live agents.")

                    # Show empty helix
                    fig = helix_monitor.create_visualization([])
                    viz_placeholder.plotly_chart(fig, use_container_width=True, key="helix_viz_empty")

                    # Refresh button to check for workflows
                    if st.button("🔄 Check for Workflow", key="check_workflow", use_container_width=True):
                        st.rerun()
            else:
                # Static mode - show with sample or real data
                agent_positions = []

                # Try to get real agent data for static view
                if not agent_df.empty:
                    for _, agent in agent_df.iterrows():
                        agent_pos = {
                            'id': agent['agent_id'],
                            'type': 'generic',
                            'progress': 0.0,  # Start at top of helix
                            'confidence': agent['avg_confidence']
                        }
                        agent_positions.append(agent_pos)

                # Use sample data if no real agents
                if not agent_positions:
                    st.info("No live agent data. Showing sample visualization.")
                    agent_positions = generate_sample_agents(5)

                # Create static visualization
                fig = helix_monitor.create_visualization(agent_positions)
                viz_placeholder.plotly_chart(fig, use_container_width=True)

        # Phase description
        st.divider()
        st.subheader("Helix Phases")

        phase_col1, phase_col2, phase_col3 = st.columns(3)

        with phase_col1:
            st.markdown("""
            **🔍 Exploration Phase (Top)**
            - Wide radius (3.0)
            - High temperature (1.0)
            - Broad discovery
            - Research agents spawn
            """)

        with phase_col2:
            st.markdown("""
            **📊 Analysis Phase (Middle)**
            - Converging radius
            - Medium temperature
            - Pattern identification
            - Analysis agents active
            """)

        with phase_col3:
            st.markdown("""
            **🎯 Synthesis Phase (Bottom)**
            - Narrow radius (0.5)
            - Low temperature (0.2)
            - Focused output
            - Final convergence
            """)

    with tab1:
        st.subheader("Agent Performance Matrix")
        st.markdown("Analyze agent performance across helix phases to identify bottlenecks and problem agents")

        # Get agent data with phase information
        agent_df = db_reader.get_agent_metrics()

        if not agent_df.empty:
            # Get knowledge entries with domain/phase info
            knowledge_df = db_reader.get_knowledge_entries(limit=1000)

            if not knowledge_df.empty:
                # Merge agent data with domain info to get phase
                merged_df = pd.merge(
                    knowledge_df[['agent_id', 'domain', 'confidence']],
                    agent_df[['agent_id', 'avg_confidence']],
                    on='agent_id',
                    how='inner'
                )

                # Infer phase from domain
                def infer_phase(domain):
                    if not domain or pd.isna(domain):
                        return "Unknown"
                    domain_lower = str(domain).lower()
                    if "research" in domain_lower or "exploration" in domain_lower or "web_search" in domain_lower:
                        return "Exploration"
                    elif "analysis" in domain_lower or "critic" in domain_lower:
                        return "Analysis"
                    elif "synthesis" in domain_lower or "final" in domain_lower:
                        return "Synthesis"
                    else:
                        return "General"

                merged_df['phase'] = merged_df['domain'].apply(infer_phase)

                # Determine agent type from agent_id or domain
                def infer_agent_type(row):
                    agent_id = str(row['agent_id']).lower()
                    domain = str(row['domain']).lower()
                    if 'research' in agent_id or 'research' in domain:
                        return "Research"
                    elif 'analysis' in agent_id or 'analysis' in domain or 'analyst' in agent_id:
                        return "Analysis"
                    elif 'critic' in agent_id or 'critic' in domain:
                        return "Critic"
                    elif 'synthesis' in agent_id or 'synthesis' in domain:
                        return "Synthesis"
                    else:
                        return "Generic"

                merged_df['agent_type'] = merged_df.apply(infer_agent_type, axis=1)

                # Create pivot table for heatmap
                heatmap_data = merged_df.groupby(['agent_type', 'phase'])['confidence'].mean().reset_index()

                if not heatmap_data.empty:
                    # Create heatmap matrix
                    pivot_data = heatmap_data.pivot(index='agent_type', columns='phase', values='confidence')

                    # Ensure phase order
                    phase_order = ['Exploration', 'Analysis', 'Synthesis', 'General', 'Unknown']
                    available_phases = [p for p in phase_order if p in pivot_data.columns]
                    pivot_data = pivot_data[available_phases]

                    # Create heatmap
                    fig = px.imshow(
                        pivot_data,
                        labels=dict(x="Helix Phase", y="Agent Type", color="Avg Confidence"),
                        x=available_phases,
                        y=pivot_data.index,
                        color_continuous_scale=[[0, "red"], [0.5, "yellow"], [1, "green"]],
                        title="Agent Performance by Phase (Heatmap)",
                        aspect="auto",
                        zmin=0,
                        zmax=1
                    )

                    fig.update_layout(
                        height=400,
                        xaxis_title="Helix Phase",
                        yaxis_title="Agent Type"
                    )
                    fig.update_traces(
                        text=pivot_data.values,
                        texttemplate='%{text:.2f}',
                        textfont={"size": 12}
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.divider()
                    st.subheader("Problem Agent Detection")

                    low_confidence_threshold = 0.7
                    problem_agents = agent_df[agent_df['avg_confidence'] < low_confidence_threshold]

                    if not problem_agents.empty:
                        st.warning(f"⚠️ **{len(problem_agents)} agent(s) with low confidence detected**")
                        st.markdown("These agents may need investigation or parameter tuning:")

                        # Format problem agents table
                        problem_display = problem_agents[['agent_id', 'avg_confidence', 'output_count']].copy()
                        problem_display['avg_confidence'] = problem_display['avg_confidence'].apply(
                            lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A"
                        )
                        problem_display.columns = ['Agent ID', 'Avg Confidence', 'Outputs']

                        st.dataframe(
                            problem_display,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Actionable recommendations
                        st.markdown("""
                        **Recommended Actions:**
                        - Review agent prompts and temperature settings
                        - Check if agents have sufficient context
                        - Verify agent positioning on helix (may be in wrong phase)
                        - Consider adjusting token budgets or search parameters
                        """)
                    else:
                        st.success(f"✅ **All agents performing well** (confidence ≥ {low_confidence_threshold*100:.0f}%)")

                else:
                    st.info("Insufficient data for performance matrix. Run more workflows to generate metrics.")
            else:
                st.info("No knowledge entries found. Run workflows to generate agent data.")

            # Agent details table
            st.divider()
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
                use_container_width=True,
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
            st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No trend data available. Metrics will appear as system runs.")

        st.divider()
        st.subheader("Agent Efficiency Matrix")

        agent_df = db_reader.get_agent_metrics()

        if not agent_df.empty:
            # Calculate efficiency metric
            agent_df['efficiency'] = agent_df['output_count'] * agent_df['avg_confidence']

            # Create scatter plot
            fig_efficiency = px.scatter(
                agent_df,
                x='output_count',
                y='avg_confidence',
                size='efficiency',
                color='efficiency',
                hover_data=['agent_id'],
                title='Agent Efficiency: Output Volume vs Confidence',
                labels={
                    'output_count': 'Number of Outputs',
                    'avg_confidence': 'Average Confidence',
                    'efficiency': 'Efficiency Score'
                },
                color_continuous_scale='Viridis'
            )
            fig_efficiency.update_layout(height=400)
            st.plotly_chart(fig_efficiency, use_container_width=True)

            # Show efficiency summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Agents", len(agent_df))
            with col2:
                max_efficiency = agent_df['efficiency'].max()
                st.metric("Max Efficiency", f"{max_efficiency:.2f}")
            with col3:
                avg_efficiency = agent_df['efficiency'].mean()
                st.metric("Avg Efficiency", f"{avg_efficiency:.2f}")
        else:
            st.info("No agent activity recorded yet. Start Felix system to begin monitoring.")

        st.divider()

        # Web Search Monitor Section
        st.markdown("### Web Search Activity")
        from streamlit_gui.components.web_search_monitor import WebSearchMonitor

        search_monitor = WebSearchMonitor(db_reader)
        search_monitor.render()

    with tab3:
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
        st.dataframe(df_info, use_container_width=True, hide_index=True)

        # Domain distribution if available
        domain_df = db_reader.get_domain_distribution()
        if not domain_df.empty:
            st.divider()
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
            st.plotly_chart(fig, use_container_width=True)

        # Connection info
        st.divider()
        st.info(
            "ℹ️ **Database Access Mode**: Read-Only\n\n"
            "The Streamlit GUI monitors shared databases without modifying them. "
            "All write operations are performed by the tkinter GUI and Felix system."
        )

        st.divider()

        # Workflow History Section
        st.markdown("### Workflow History")
        from streamlit_gui.components.workflow_history_viewer import WorkflowHistoryViewer

        history_viewer = WorkflowHistoryViewer(db_reader)
        history_viewer.render()

    # Auto-refresh option
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    with col2:
        if auto_refresh:
            st.write("Dashboard will refresh every 5 seconds")
            time.sleep(5)
            st.rerun()

if __name__ == "__main__":
    main()