"""
Streamlit GUI for Felix Framework

This is the main entry point for the Streamlit monitoring and visualization interface.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

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

# Custom CSS for better styling
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .main {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 20px;
            margin: 20px;
        }
        .stSidebar {
            background-color: rgba(255, 255, 255, 0.9);
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 10px 0;
        }
        h1 {
            color: #2c3e50;
        }
        .stAlert {
            background-color: #f8f9fa;
            border-left: 4px solid #4CAF50;
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
            from streamlit_gui.backend.system_monitor import SystemMonitor
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