"""Streamlit GUI for Felix Framework monitoring and visualization."""

import streamlit as st
import sys
import os
import warnings
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Temporarily suppress Streamlit deprecation warnings (tracked in TECH_DEBT.md)
# Issue: Streamlit 1.50.0 shows deprecation for use_container_width but width='stretch' not fully implemented
# Target removal: After Streamlit > 1.50.0 releases with proper width parameter support
# Note: Terminal warnings cannot be suppressed without affecting performance, but UI is clean
warnings.filterwarnings('ignore', category=DeprecationWarning, module='streamlit')
warnings.filterwarnings('ignore', message='.*use_container_width.*')
warnings.filterwarnings('ignore', message='.*Please replace.*use_container_width.*width.*')

# Configure Streamlit page
st.set_page_config(
    page_title="Felix Framework Monitor",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/CalebisGross/felix/issues',
        'Report a bug': "https://github.com/CalebisGross/felix/issues",
        'About': """
        # Felix Framework Monitor

        A read-only monitoring and visualization interface for the Felix AI Framework.

        Version: 3.0.0
        """
    }
)

# Custom CSS for dark professional theme - WCAG compliant
st.markdown("""
    <style>
        /* Dark professional background */
        .stApp {
            background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        }

        /* Main content area - dark blue-gray */
        .main {
            background: #2d3748;
            border: 1px solid #4a5568;
            border-radius: 8px;
            padding: 20px;
            margin: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        /* Sidebar - dark theme */
        .stSidebar {
            background: #1a202c !important;
            border-right: 1px solid #4a5568;
        }

        .stSidebar [data-testid="stMarkdownContainer"] {
            color: #e2e8f0 !important;
        }

        .stSidebar .stRadio label {
            color: #e2e8f0 !important;
        }

        /* Headers - light on dark (WCAG AA compliant) */
        h1, h2, h3 {
            color: #f7fafc !important;
        }

        /* Body text - light gray on dark */
        p, div, span, label {
            color: #e2e8f0 !important;
        }

        .stMarkdown {
            color: #e2e8f0 !important;
        }

        /* Alert boxes - dark theme with borders */
        div[data-testid="stAlert"] {
            background-color: #2d3748 !important;
            border: 1px solid #4a5568 !important;
            color: #e2e8f0 !important;
        }

        div[data-testid="stAlert"] p {
            color: #e2e8f0 !important;
        }

        div[data-testid="stAlert"] strong {
            color: #f7fafc !important;
        }

        /* Button styling - blue accent on dark */
        .stButton > button {
            background: #3182ce !important;
            color: #ffffff !important;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            transition: background 0.2s;
        }

        .stButton > button:hover {
            background: #2c5282 !important;
        }

        /* Metric containers - dark with subtle borders */
        [data-testid="metric-container"] {
            background: #1a202c;
            border: 1px solid #4a5568;
            border-radius: 6px;
            padding: 12px;
        }

        [data-testid="metric-container"] [data-testid="stMetricLabel"] {
            color: #a0aec0 !important;
            font-weight: 500;
        }

        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #63b3ed !important;
            font-weight: 600;
        }

        /* Expander styling - dark theme */
        .streamlit-expanderHeader {
            background: #1a202c !important;
            color: #f7fafc !important;
            border: 1px solid #4a5568;
            font-weight: 500;
        }

        .streamlit-expanderHeader:hover {
            background: #2d3748 !important;
        }

        /* Table styling - dark theme */
        .stDataFrame {
            color: #e2e8f0 !important;
        }

        .stDataFrame thead tr th {
            background: #1a202c !important;
            color: #f7fafc !important;
            font-weight: 600;
        }

        /* Links - blue accent on dark */
        a {
            color: #63b3ed !important;
        }

        a:hover {
            color: #4299e1 !important;
            text-decoration: underline;
        }

        /* Input fields - dark theme */
        .stTextInput > div > div > input {
            background-color: #2d3748 !important;
            color: #e2e8f0 !important;
            border: 1px solid #4a5568 !important;
        }

        .stSelectbox > div > div > div {
            background-color: #2d3748 !important;
            color: #e2e8f0 !important;
        }
    </style>
""", unsafe_allow_html=True)

def main() -> None:
    """Main application page."""

    # Header
    st.title("🌀 Felix Framework Monitor")
    st.markdown("Real-time monitoring dashboard for Felix AI Framework | **Read-Only Interface**")

    st.divider()

    # Live System Status
    st.subheader("📊 System Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Felix system status
        try:
            from backend.system_monitor import SystemMonitor
            monitor = SystemMonitor()
            is_running = monitor.check_felix_running()

            if is_running:
                st.metric("Felix System", "🟢 Running", delta="Active")
            else:
                st.metric("Felix System", "🔴 Stopped", delta="Inactive", delta_color="inverse")
        except:
            st.metric("Felix System", "⚠️ Unknown", delta="Check Connection")

    with col2:
        # Database connection status
        db_files = ["felix_knowledge.db", "felix_memory.db", "felix_task_memory.db",
                    "felix_workflow_history.db", "felix_agent_performance.db", "felix_system_actions.db"]
        connected = sum(1 for db in db_files if Path(db).exists())
        total = len(db_files)

        if connected == total:
            st.metric("Databases", f"{connected}/{total}", delta="All Connected")
        else:
            st.metric("Databases", f"{connected}/{total}", delta=f"{total-connected} Missing", delta_color="inverse")

    with col3:
        # Active agents (from database if available)
        try:
            import sqlite3
            if Path("felix_knowledge.db").exists():
                conn = sqlite3.connect("felix_knowledge.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(DISTINCT agent_id) FROM knowledge")
                agent_count = cursor.fetchone()[0]
                conn.close()
                st.metric("Agents", agent_count, delta="Recorded")
            else:
                st.metric("Agents", "N/A", delta="No Data")
        except:
            st.metric("Agents", "N/A", delta="DB Error")

    with col4:
        # Pending approvals (from system actions DB)
        try:
            if Path("felix_system_actions.db").exists():
                conn = sqlite3.connect("felix_system_actions.db")
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM command_executions WHERE executed = 0 AND approved_at IS NULL")
                pending = cursor.fetchone()[0]
                conn.close()

                if pending > 0:
                    st.metric("Pending Approvals", pending, delta="Needs Attention", delta_color="inverse")
                else:
                    st.metric("Pending Approvals", 0, delta="All Clear")
            else:
                st.metric("Pending Approvals", "N/A", delta="No Data")
        except:
            st.metric("Pending Approvals", "N/A", delta="DB Error")

    st.divider()

    # Quick navigation hint
    st.info("👈 **Use the sidebar to navigate** | View Dashboard, Testing, Benchmarking, and more")

    # Quick Start Guide (collapsed by default)
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        ### Navigation

        - **Dashboard**: System overview and real-time agent monitoring
        - **Configuration**: View and export system settings
        - **Testing**: Analyze workflow results
        - **Benchmarking**: Validate hypothesis performance
        - **System Monitor**: View command execution and approvals
        - **Advanced Analytics**: Strategic insights and performance analytics

        ### Control vs Monitor

        **This interface is read-only**. Use the tkinter GUI for system control (start/stop, spawn agents, approve commands).
        """)

    # Database Details (collapsed)
    with st.expander("📁 Database Details"):
        db_info = {
            "felix_knowledge.db": "Agent knowledge and outputs",
            "felix_memory.db": "Task memory and patterns",
            "felix_task_memory.db": "Workflow memory",
            "felix_workflow_history.db": "Workflow execution history",
            "felix_agent_performance.db": "Agent performance metrics",
            "felix_system_actions.db": "System command execution logs"
        }

        for db_file, description in db_info.items():
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                if Path(db_file).exists():
                    st.markdown(f"🟢 **{db_file}**")
                else:
                    st.markdown(f"🔴 **{db_file}**")

            with col2:
                if Path(db_file).exists():
                    size_mb = Path(db_file).stat().st_size / (1024 * 1024)
                    st.markdown(f"`{size_mb:.2f} MB`")
                else:
                    st.markdown("`Not Found`")

            with col3:
                st.markdown(f"*{description}*")

    # Footer
    st.divider()
    st.caption("Felix Framework Monitor v3.0.0 | Read-Only Monitoring Interface")

if __name__ == "__main__":
    main()