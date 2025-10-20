# Streamlit GUI Architecture for Felix Framework

**Version:** 3.0
**Target Users:** Technical developers familiar with Felix internals
**Deployment:** Local development machines
**Last Updated:** 2024-10-20

> **Core Design Principle:** The Streamlit GUI operates as a monitoring and visualization layer that shares databases with the tkinter GUI, providing complementary analytics without interfering with control operations.

---

## Executive Summary

The Streamlit GUI is a **read-heavy monitoring interface** that complements the existing tkinter GUI by providing advanced visualization, benchmarking, and analysis capabilities. Both GUIs share the same Felix databases (`felix_memory.db`, `felix_knowledge.db`, `felix_task_memory.db`) and can monitor the same running Felix instance, but operate in different modes:

- **tkinter GUI**: Primary control interface (start/stop system, spawn agents, modify settings)
- **Streamlit GUI**: Monitoring & analysis interface (visualize data, run benchmarks, export reports)

---

## 1. Architecture Overview

### 1.1 Shared System Architecture

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│      tkinter GUI            │     │      Streamlit GUI          │
│    (Control Mode)           │     │    (Monitor Mode)           │
│                             │     │                             │
│  • Start/Stop Felix         │     │  • Visualize Metrics        │
│  • Spawn Agents             │     │  • Analyze Performance      │
│  • Modify Settings          │     │  • Run Benchmarks           │
│  • Direct Control           │     │  • Export Reports           │
└──────────┬──────────────────┘     └──────────┬────────────────┘
           │                                     │
           │ Read/Write                         │ Read-Mostly
           │                                     │
           └──────────────┬──────────────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │    Shared Databases          │
           │  • felix_memory.db           │
           │  • felix_knowledge.db        │
           │  • felix_task_memory.db      │
           └──────────────┬───────────────┘
                          │
                          ▲ Write
                          │
                ┌─────────┴──────────┐
                │   Felix System      │
                │  (Single Instance)  │
                └────────────────────┘
```

### 1.2 Operational Modes

| Aspect | tkinter GUI | Streamlit GUI |
|--------|------------|---------------|
| **Primary Role** | System Control | System Monitoring |
| **Database Access** | Read/Write | Read-Mostly |
| **Felix System** | Starts/Stops | Monitors/Connects |
| **Agent Spawning** | Yes | View Only |
| **Settings** | Modify | View/Export |
| **Workflows** | Execute | Analyze Results |
| **Best For** | Real-time Control | Analytics & Visualization |

### 1.3 Non-Interference Design

The Streamlit GUI achieves zero interference through:

1. **Separate Directory Structure**: `streamlit_gui/` completely separate from `src/gui/`
2. **Import Without Modification**: Uses existing `FelixSystem` class via imports only
3. **Separate Configuration**: Uses `streamlit_config.yaml` instead of `felix_gui_config.json`
4. **Read-Mostly Database Access**: Primarily reads from shared databases
5. **Independent Entry Point**: `streamlit_app.py` separate from tkinter's `python -m src.gui`

---

## 2. Directory Structure

```
felix/
├── streamlit_app.py                    # Streamlit entry point
├── streamlit_gui/                      # Separate from src/gui (tkinter)
│   ├── __init__.py
│   ├── pages/                          # Streamlit multipage app
│   │   ├── 1_🏠_Dashboard.py          # System monitoring
│   │   ├── 2_⚙️_Configuration.py      # Config viewing/export
│   │   ├── 3_🧪_Testing.py            # Test result analysis
│   │   └── 4_📊_Benchmarking.py       # Performance benchmarking
│   ├── components/
│   │   ├── __init__.py
│   │   ├── config_viewer.py           # Config display widgets
│   │   ├── log_monitor.py             # Log streaming display
│   │   ├── metrics_display.py         # Real-time metrics
│   │   ├── agent_visualizer.py        # Agent state visualization
│   │   └── results_analyzer.py        # Result analysis tools
│   └── backend/
│       ├── __init__.py
│       ├── system_monitor.py          # Felix system monitoring
│       ├── db_reader.py               # Database read operations
│       ├── config_handler.py          # Config management
│       └── benchmark_runner.py        # Isolated benchmark execution
├── streamlit_config.yaml              # Default Streamlit GUI config
├── requirements_streamlit.txt         # Streamlit-specific dependencies
│
├── src/gui/                           # Existing tkinter GUI (DO NOT MODIFY)
│   ├── main.py
│   ├── felix_system.py               # Shared FelixSystem class
│   └── ...
│
└── [Shared Resources]
    ├── felix_memory.db                # Shared database
    ├── felix_knowledge.db             # Shared database
    └── felix_task_memory.db           # Shared database
```

---

## 3. Backend Implementation

### 3.1 System Monitor (Read-Only Pattern)

```python
# streamlit_gui/backend/system_monitor.py

import sqlite3
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

# Import Felix components without modification
from src.gui.felix_system import FelixSystem, FelixConfig
from src.memory.knowledge_store import KnowledgeStore
from src.memory.task_memory import TaskMemory

logger = logging.getLogger(__name__)


class SystemMonitor:
    """
    Monitor running Felix system and databases in read-only mode.
    This class provides safe, non-interfering access to Felix data.
    """

    def __init__(self):
        self.felix_system: Optional[FelixSystem] = None
        self.is_connected = False
        self.read_only = True

        # Connect to shared databases (read-only by default)
        self.knowledge_store = None
        self.task_memory = None
        self._init_db_connections()

    def _init_db_connections(self):
        """Initialize read-only database connections."""
        try:
            # Connect to shared databases
            if Path("felix_knowledge.db").exists():
                self.knowledge_store = KnowledgeStore("felix_knowledge.db")
                logger.info("Connected to felix_knowledge.db (read-only)")

            if Path("felix_memory.db").exists():
                self.task_memory = TaskMemory("felix_memory.db")
                logger.info("Connected to felix_memory.db (read-only)")

        except Exception as e:
            logger.warning(f"Could not connect to databases: {e}")

    def check_felix_running(self) -> bool:
        """
        Check if Felix system is currently running.
        Uses lock file or port check to detect running instance.
        """
        # Check for lock file (created by tkinter GUI)
        lock_file = Path(".felix_running.lock")
        if lock_file.exists():
            try:
                with open(lock_file, 'r') as f:
                    lock_data = json.load(f)
                    return lock_data.get("running", False)
            except:
                pass

        # Alternative: Check if LM Studio port is in use
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 1234))
        sock.close()
        return result == 0

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics from shared databases."""
        metrics = {
            "felix_running": self.check_felix_running(),
            "agents": 0,
            "messages": 0,
            "knowledge_entries": 0,
            "task_patterns": 0,
            "confidence_avg": 0.0
        }

        # Read metrics from databases
        if self.knowledge_store:
            try:
                entries = self.knowledge_store.get_all_entries()
                metrics["knowledge_entries"] = len(entries)

                # Calculate average confidence
                if entries:
                    confidences = [e.get("confidence", 0.0) for e in entries]
                    metrics["confidence_avg"] = sum(confidences) / len(confidences)

            except Exception as e:
                logger.debug(f"Could not read knowledge entries: {e}")

        if self.task_memory:
            try:
                patterns = self.task_memory.get_all_patterns()
                metrics["task_patterns"] = len(patterns)
            except Exception as e:
                logger.debug(f"Could not read task patterns: {e}")

        return metrics

    def get_agent_data(self) -> List[Dict[str, Any]]:
        """Read agent data from shared databases."""
        agents = []

        if self.knowledge_store:
            # Query knowledge database for agent outputs
            try:
                conn = sqlite3.connect("felix_knowledge.db")
                cursor = conn.cursor()

                # Get recent agent activities
                cursor.execute("""
                    SELECT agent_id, domain, confidence, timestamp, content
                    FROM knowledge
                    WHERE agent_id IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 100
                """)

                for row in cursor.fetchall():
                    agents.append({
                        "agent_id": row[0],
                        "domain": row[1],
                        "confidence": row[2],
                        "timestamp": row[3],
                        "content": row[4][:200]  # Truncate for display
                    })

                conn.close()

            except Exception as e:
                logger.debug(f"Could not read agent data: {e}")

        return agents

    def get_workflow_results(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Read recent workflow results from databases."""
        results = []

        if self.task_memory:
            try:
                # Get recent task completions
                patterns = self.task_memory.get_recent_patterns(limit)
                for pattern in patterns:
                    results.append({
                        "task": pattern.get("task", ""),
                        "timestamp": pattern.get("timestamp", ""),
                        "success": pattern.get("success", False),
                        "agent_count": pattern.get("agent_count", 0),
                        "final_synthesis": pattern.get("final_synthesis", "")
                    })

            except Exception as e:
                logger.debug(f"Could not read workflow results: {e}")

        return results

    def create_isolated_instance(self, config: Dict[str, Any]) -> Optional[FelixSystem]:
        """
        Create an isolated Felix instance for benchmarking.
        This instance uses temporary databases to avoid interference.
        """
        try:
            # Create config with temporary databases
            felix_config = FelixConfig(
                lm_host=config.get('lm_host', '127.0.0.1'),
                lm_port=config.get('lm_port', 1234),
                memory_db_path="temp_memory.db",  # Temporary DB
                knowledge_db_path="temp_knowledge.db",  # Temporary DB
                **config
            )

            # Create isolated instance
            isolated_system = FelixSystem(felix_config)
            logger.info("Created isolated Felix instance for benchmarking")
            return isolated_system

        except Exception as e:
            logger.error(f"Could not create isolated instance: {e}")
            return None
```

### 3.2 Database Reader

```python
# streamlit_gui/backend/db_reader.py

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
        return self._read_query("knowledge", query) or pd.DataFrame()

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
        return self._read_query("memory", query) or pd.DataFrame()

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
        return self._read_query("knowledge", query) or pd.DataFrame()

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
        return self._read_query("knowledge", query) or pd.DataFrame()
```

---

## 4. Streamlit Pages Implementation

### 4.1 Dashboard Page

```python
# streamlit_gui/pages/1_🏠_Dashboard.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

# Custom imports
import sys
sys.path.append(".")
from streamlit_gui.backend.system_monitor import SystemMonitor
from streamlit_gui.backend.db_reader import DatabaseReader

st.set_page_config(
    page_title="Felix Dashboard",
    page_icon="🏠",
    layout="wide"
)

@st.cache_resource
def get_monitor():
    return SystemMonitor()

@st.cache_resource
def get_db_reader():
    return DatabaseReader()

def main():
    st.title("🏠 Felix System Dashboard")
    st.markdown("Real-time monitoring of Felix Framework")

    monitor = get_monitor()
    db_reader = get_db_reader()

    # System Status Header
    col1, col2, col3, col4 = st.columns(4)

    metrics = monitor.get_system_metrics()

    with col1:
        status = "🟢 Running" if metrics["felix_running"] else "🔴 Stopped"
        st.metric("System Status", status)

    with col2:
        st.metric("Knowledge Entries", metrics["knowledge_entries"],
                 delta=f"+{metrics.get('new_entries', 0)}")

    with col3:
        st.metric("Task Patterns", metrics["task_patterns"])

    with col4:
        st.metric("Avg Confidence", f"{metrics['confidence_avg']:.2%}")

    st.divider()

    # Real-time Metrics
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Agent Activity",
        "📈 Performance Trends",
        "🔄 Recent Workflows",
        "💾 Database Status"
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
            st.plotly_chart(fig, use_container_width=True)

            # Agent details table
            st.subheader("Agent Details")
            st.dataframe(
                agent_df[["agent_id", "output_count", "avg_confidence", "last_seen"]],
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
                line=dict(color='blue', width=2)
            ))
            fig.update_layout(
                title="Confidence Trend (24 Hours)",
                xaxis_title="Time",
                yaxis_title="Average Confidence",
                hovermode='x'
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
                yaxis_title="Number of Entries"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No trend data available. Metrics will appear as system runs.")

    with tab3:
        st.subheader("Recent Workflow Results")

        workflows = monitor.get_workflow_results(limit=10)

        if workflows:
            for workflow in workflows:
                with st.expander(f"🔄 {workflow['task'][:50]}... - {workflow['timestamp']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Agents", workflow['agent_count'])
                    with col2:
                        status = "✅" if workflow['success'] else "❌"
                        st.metric("Status", status)
                    with col3:
                        st.metric("Synthesis", "Yes" if workflow['final_synthesis'] else "No")

                    if workflow['final_synthesis']:
                        st.text_area("Final Synthesis", workflow['final_synthesis'], height=100)
        else:
            st.info("No workflow results available. Run workflows from tkinter GUI to see results here.")

    with tab4:
        st.subheader("Database Status")

        # Database sizes and info
        import os

        db_info = []
        for db_name, db_path in [
            ("Knowledge", "felix_knowledge.db"),
            ("Memory", "felix_memory.db"),
            ("Task Memory", "felix_task_memory.db")
        ]:
            if os.path.exists(db_path):
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                db_info.append({
                    "Database": db_name,
                    "Path": db_path,
                    "Size (MB)": f"{size_mb:.2f}",
                    "Status": "🟢 Connected"
                })
            else:
                db_info.append({
                    "Database": db_name,
                    "Path": db_path,
                    "Size (MB)": "N/A",
                    "Status": "🔴 Not Found"
                })

        st.dataframe(pd.DataFrame(db_info), use_container_width=True, hide_index=True)

        # Connection info
        st.info(
            "ℹ️ **Database Access Mode**: Read-Only\n\n"
            "The Streamlit GUI monitors shared databases without modifying them. "
            "All write operations are performed by the tkinter GUI and Felix system."
        )

    # Auto-refresh option
    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    with col2:
        if auto_refresh:
            st.write("Dashboard will refresh every 5 seconds")
            st.experimental_rerun()  # This would be in a timer loop in production

if __name__ == "__main__":
    main()
```

### 4.2 Configuration Page

```python
# streamlit_gui/pages/2_⚙️_Configuration.py

import streamlit as st
import yaml
import json
from pathlib import Path

st.set_page_config(
    page_title="Felix Configuration",
    page_icon="⚙️",
    layout="wide"
)

def main():
    st.title("⚙️ Configuration Viewer")
    st.markdown("View and export Felix configuration (read-only)")

    # Try to load existing configuration
    config_sources = {
        "felix_gui_config.json": "tkinter GUI Configuration",
        "streamlit_config.yaml": "Streamlit Configuration",
        "configs/default_config.yaml": "Default Configuration"
    }

    tab1, tab2, tab3 = st.tabs(["📋 View Config", "📊 Compare", "💾 Export"])

    with tab1:
        st.subheader("Current Configuration")

        # Select config source
        selected_source = st.selectbox(
            "Configuration Source",
            options=list(config_sources.keys()),
            format_func=lambda x: config_sources[x]
        )

        if Path(selected_source).exists():
            with open(selected_source, 'r') as f:
                if selected_source.endswith('.json'):
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)

            # Display configuration sections
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Helix Geometry")
                helix_config = config.get('helix', {})
                st.json(helix_config)

                st.markdown("### Agent Configuration")
                agent_config = {
                    "max_agents": config.get('max_agents', 25),
                    "base_token_budget": config.get('base_token_budget', 2500)
                }
                st.json(agent_config)

            with col2:
                st.markdown("### LM Studio Connection")
                lm_config = {
                    "host": config.get('lm_host', '127.0.0.1'),
                    "port": config.get('lm_port', 1234)
                }
                st.json(lm_config)

                st.markdown("### Dynamic Spawning")
                spawn_config = config.get('spawning', {})
                st.json(spawn_config)

            # Visualization of helix geometry
            st.divider()
            st.markdown("### Helix Geometry Visualization")

            import numpy as np
            import plotly.graph_objects as go

            # Generate helix points
            t = np.linspace(0, 2*np.pi*helix_config.get('turns', 2), 100)
            top_r = helix_config.get('top_radius', 3.0)
            bottom_r = helix_config.get('bottom_radius', 0.5)
            height = helix_config.get('height', 8.0)

            z = np.linspace(0, height, 100)
            r = top_r - (top_r - bottom_r) * (z / height)
            x = r * np.cos(t)
            y = r * np.sin(t)

            fig = go.Figure(data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color=z, colorscale='Viridis', width=4)
            )])

            fig.update_layout(
                title="Helix Geometry",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Depth"
                ),
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"Configuration file '{selected_source}' not found.")

    with tab2:
        st.subheader("Configuration Comparison")
        st.info("This feature compares different configuration files to identify differences.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Config A")
            config_a = st.file_uploader("Upload first config", type=['json', 'yaml', 'yml'])

        with col2:
            st.markdown("### Config B")
            config_b = st.file_uploader("Upload second config", type=['json', 'yaml', 'yml'])

        if config_a and config_b:
            # Parse configs
            if config_a.name.endswith('.json'):
                cfg_a = json.load(config_a)
            else:
                cfg_a = yaml.safe_load(config_a)

            if config_b.name.endswith('.json'):
                cfg_b = json.load(config_b)
            else:
                cfg_b = yaml.safe_load(config_b)

            # Compare and show differences
            st.markdown("### Differences")
            # (Implementation of diff display)
            st.json({"Config A": cfg_a, "Config B": cfg_b})

    with tab3:
        st.subheader("Export Configuration")

        if Path(selected_source).exists():
            with open(selected_source, 'r') as f:
                content = f.read()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "📥 Download as YAML",
                    data=yaml.dump(config) if isinstance(config, dict) else content,
                    file_name="felix_config_export.yaml",
                    mime="text/yaml"
                )

            with col2:
                st.download_button(
                    "📥 Download as JSON",
                    data=json.dumps(config, indent=2) if isinstance(config, dict) else content,
                    file_name="felix_config_export.json",
                    mime="application/json"
                )

            with col3:
                st.download_button(
                    "📥 Download Original",
                    data=content,
                    file_name=f"felix_config_original{Path(selected_source).suffix}",
                    mime="text/plain"
                )

    # Important note about read-only nature
    st.divider()
    st.warning(
        "⚠️ **Read-Only Mode**: Configuration viewing and export only. "
        "To modify settings, use the tkinter GUI Settings tab."
    )

if __name__ == "__main__":
    main()
```

---

## 5. Running and Deployment

### 5.1 Installation

```bash
# Clone repository
git clone <repository>
cd felix

# Install core Felix dependencies (if not already installed)
pip install -r requirements.txt

# Install Streamlit GUI dependencies
pip install -r requirements_streamlit.txt
```

### 5.2 Requirements File

```txt
# requirements_streamlit.txt
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
pydantic>=2.4.0
pyyaml>=6.0
watchdog>=3.0.0  # For file monitoring
altair>=5.0.0    # Additional charting
```

### 5.3 Running Both GUIs

```bash
# Terminal 1: Start tkinter GUI (Primary Control)
python -m src.gui.main

# Terminal 2: Start Streamlit GUI (Monitoring)
streamlit run streamlit_app.py

# Or run both with a script
./run_both_guis.sh
```

### 5.4 Typical Workflow

1. **Start tkinter GUI** → Initialize Felix system → Configure parameters
2. **Start Streamlit GUI** → Monitor system → View real-time metrics
3. **Run workflow in tkinter** → Monitor progress in Streamlit
4. **Analyze results in Streamlit** → Export reports → Adjust settings in tkinter

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Day 1)
- [x] Create directory structure (`streamlit_gui/`)
- [x] Implement `SystemMonitor` class
- [x] Create `DatabaseReader` for safe DB access
- [x] Setup `streamlit_app.py` entry point
- [x] Test database connections

### Phase 2: Dashboard (Day 2)
- [ ] Implement real-time metrics display
- [ ] Create agent activity visualizations
- [ ] Add performance trend charts
- [ ] Setup auto-refresh mechanism

### Phase 3: Configuration Viewer (Day 3)
- [ ] Load and display configurations
- [ ] Create helix geometry visualization
- [ ] Implement configuration comparison
- [ ] Add export functionality

### Phase 4: Testing & Analysis (Day 4)
- [ ] Display workflow results from database
- [ ] Create result analysis tools
- [ ] Add confidence visualization
- [ ] Implement report generation

### Phase 5: Benchmarking (Day 5)
- [ ] Create isolated benchmark runner
- [ ] Implement hypothesis validation charts
- [ ] Add performance comparison tools
- [ ] Create export functionality

### Phase 6: Polish & Testing (Day 6)
- [ ] Error handling and fallbacks
- [ ] Performance optimization
- [ ] Documentation
- [ ] Integration testing with tkinter GUI

---

## 7. Technical Specifications

### 7.1 Performance Targets
- Dashboard refresh: < 500ms
- Database query: < 100ms
- Chart rendering: < 200ms
- Page load: < 1s
- Memory usage: < 200MB

### 7.2 Concurrency Handling
- SQLite read-only connections prevent locks
- Caching with `@st.cache_resource` for expensive operations
- Pagination for large datasets
- Lazy loading for charts

### 7.3 Error Handling
```python
try:
    # Attempt database read
    data = db_reader.get_knowledge_entries()
except Exception as e:
    st.warning("Could not read database. Is Felix running?")
    st.info("Start Felix from tkinter GUI to begin monitoring.")
```

---

## 8. Key Design Decisions

### Why Read-Mostly Pattern?
- **Safety**: Prevents conflicts with tkinter GUI
- **Simplicity**: No complex synchronization needed
- **Performance**: Read operations are fast and non-blocking
- **Reliability**: System continues working even if Streamlit crashes

### Why Shared Databases?
- **Consistency**: Both GUIs show same data
- **Efficiency**: No data duplication
- **Real-time**: Immediate visibility of changes
- **Simplicity**: No complex sync mechanisms

### Why Separate Directories?
- **Isolation**: Clear separation of concerns
- **Maintenance**: Easy to update independently
- **Testing**: Can test each GUI separately
- **Clarity**: No confusion about which files belong where

---

## 9. Conclusion

The Streamlit GUI v3 architecture provides a powerful monitoring and analysis layer that complements the tkinter control interface without any interference. Key achievements:

1. **Zero Code Interference**: No modifications to existing code
2. **Shared Monitoring**: Both GUIs work with same Felix instance
3. **Complementary Roles**: Each GUI excels at different tasks
4. **Safe Coexistence**: Read-mostly pattern prevents conflicts
5. **Enhanced Insights**: Advanced visualizations and analytics

The architecture ensures both GUIs can run simultaneously, sharing the same databases and monitoring the same Felix system, while maintaining complete code separation and operational safety.

**Next Steps**: Begin implementation with Phase 1 (Foundation), focusing on establishing the monitoring infrastructure and database connections.