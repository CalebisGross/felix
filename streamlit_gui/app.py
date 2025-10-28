"""Streamlit GUI for Felix Framework monitoring and visualization."""

import streamlit as st
import sys
import os
import warnings
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Temporarily suppress Streamlit deprecation warnings
# TODO: Remove when Streamlit fixes width parameter issue (tracked in TECH_DEBT.md)
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
        'Get Help': 'https://github.com/your-org/felix/issues',
        'Report a bug': "https://github.com/your-org/felix/issues",
        'About': """
        # Felix Framework Monitor

        A read-only monitoring and visualization interface for the Felix AI Framework.

        Version: 3.0.0
        """
    }
)

# Custom CSS for professional theme - SIMPLIFIED FOR VISIBILITY
st.markdown("""
    <style>
        /* Light professional background - WCAG compliant */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        }

        /* Main content area - crisp white */
        .main {
            background: #ffffff;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            padding: 20px;
            margin: 20px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        /* Sidebar - light gray with dark text (WCAG AA compliant) */
        .stSidebar {
            background: #f6f8fa !important;
            border-right: 1px solid #d0d7de;
        }

        .stSidebar [data-testid="stMarkdownContainer"] {
            color: #24292f !important;
        }

        .stSidebar .stRadio label {
            color: #24292f !important;
        }

        /* Headers - WCAG AAA compliant (7:1 contrast ratio) */
        h1, h2, h3 {
            color: #0d1117 !important;
        }

        /* Body text - WCAG AAA compliant */
        p, div, span, label {
            color: #24292f !important;
        }

        .stMarkdown {
            color: #24292f !important;
        }

        /* Alert boxes - high contrast */
        div[data-testid="stAlert"] {
            background-color: #ffffff !important;
            border: 1px solid #d0d7de !important;
            color: #24292f !important;
        }

        div[data-testid="stAlert"] p {
            color: #24292f !important;
        }

        div[data-testid="stAlert"] strong {
            color: #0d1117 !important;
        }

        /* Button styling - WCAG AA compliant */
        .stButton > button {
            background: #0969da !important;
            color: #ffffff !important;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            transition: background 0.2s;
        }

        .stButton > button:hover {
            background: #0550ae !important;
        }

        /* Metric containers - high contrast */
        [data-testid="metric-container"] {
            background: #f6f8fa;
            border: 1px solid #d0d7de;
            border-radius: 6px;
            padding: 12px;
        }

        [data-testid="metric-container"] [data-testid="stMetricLabel"] {
            color: #57606a !important;
            font-weight: 500;
        }

        [data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #0969da !important;
            font-weight: 600;
        }

        /* Expander styling - high contrast */
        .streamlit-expanderHeader {
            background: #f6f8fa !important;
            color: #0d1117 !important;
            border: 1px solid #d0d7de;
            font-weight: 500;
        }

        .streamlit-expanderHeader:hover {
            background: #eaeef2 !important;
        }

        /* Table styling - WCAG compliant */
        .stDataFrame {
            color: #24292f !important;
        }

        .stDataFrame thead tr th {
            background: #f6f8fa !important;
            color: #0d1117 !important;
            font-weight: 600;
        }

        /* Links - WCAG AA compliant */
        a {
            color: #0969da !important;
        }

        a:hover {
            color: #0550ae !important;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application page."""

    # Header
    st.title("🌀 Felix Framework Monitor")
    st.markdown("""
    ### Real-time Monitoring & Visualization Interface

    Welcome to the Felix Framework monitoring dashboard. This interface provides read-only access
    to system metrics, agent activity, and performance analytics.
    """)

    # System check
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        📊 **Monitor Mode**

        This GUI operates in read-only mode, monitoring shared databases without interfering
        with the control interface.
        """)

    with col2:
        st.success("""
        🔄 **Real-time Updates**

        View live metrics, agent activities, and system performance as they happen.
        """)

    with col3:
        st.warning("""
        🎛️ **Control Interface**

        Use the tkinter GUI to start/stop Felix, spawn agents, and modify settings.
        """)

    st.divider()

    # Quick Start Guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        ### Getting Started

        1. **Start Felix System**: Launch the tkinter GUI and start the Felix system
        2. **Navigate Pages**: Use the sidebar to access different monitoring views:
           - 🏠 **Dashboard**: System overview and real-time metrics
           - ⚙️ **Configuration**: View and export system configuration
           - 🧪 **Testing**: Analyze workflow results and test outcomes
           - 📊 **Benchmarking**: Performance benchmarks and hypothesis validation

        ### Two-GUI Architecture

        The Felix Framework uses a dual-GUI approach:

        | tkinter GUI (Control) | Streamlit GUI (Monitor) |
        |----------------------|------------------------|
        | Start/Stop System | Visualize Metrics |
        | Spawn Agents | Analyze Performance |
        | Modify Settings | Run Benchmarks |
        | Execute Workflows | Export Reports |

        Both interfaces share the same databases, allowing seamless monitoring while maintaining
        operational separation.
        """)

    # Database Status Check
    st.divider()
    st.subheader("📁 Database Status")

    # Check for database files
    db_files = {
        "felix_knowledge.db": "Knowledge Store",
        "felix_memory.db": "Task Memory",
        "felix_task_memory.db": "Workflow Memory"
    }

    db_cols = st.columns(len(db_files))

    for idx, (db_file, db_name) in enumerate(db_files.items()):
        with db_cols[idx]:
            if Path(db_file).exists():
                size_mb = Path(db_file).stat().st_size / (1024 * 1024)
                st.metric(
                    label=db_name,
                    value="Connected",
                    delta=f"{size_mb:.2f} MB",
                    delta_color="normal"
                )
            else:
                st.metric(
                    label=db_name,
                    value="Not Found",
                    delta="No database",
                    delta_color="off"
                )

    # Navigation hint
    st.divider()
    st.info("""
    👈 **Use the sidebar to navigate to different monitoring views**

    Each page provides specialized tools for monitoring and analyzing different aspects
    of the Felix Framework.
    """)

    # Footer
    st.divider()
    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption("""
        Felix Framework Monitor v3.0.0 | Read-Only Monitoring Interface

        This interface complements the tkinter control GUI by providing advanced visualization
        and analytics capabilities.
        """)

    with col2:
        # Check system status
        try:
            from backend.system_monitor import SystemMonitor
            monitor = SystemMonitor()
            is_running = monitor.check_felix_running()

            if is_running:
                st.success("🟢 Felix System Running")
            else:
                st.error("🔴 Felix System Stopped")
        except:
            st.warning("⚠️ Status Unknown")

if __name__ == "__main__":
    main()