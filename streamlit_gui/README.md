# Streamlit GUI for Felix Framework

A read-only monitoring and visualization interface for the Felix multi-agent AI framework. Provides real-time analytics, performance monitoring, and hypothesis validation while complementing the tkinter control GUI.

**Status**: ✅ Production Ready

---

## Features

### 🏠 Dashboard
Real-time system monitoring with interactive visualizations:
- Knowledge entries and agent activity tracking
- Performance trends and confidence metrics
- Agent performance overview charts
- Workflow execution history

### ⚙️ Configuration
Settings viewer and export capabilities:
- Helix geometry parameter visualization
- Interactive 3D helix model
- Export configurations to YAML/JSON
- Configuration comparison tools

### 🧪 Testing
Comprehensive workflow analysis:
- Workflow execution timeline
- Success/failure pattern analysis
- Performance metrics over time
- Multiple report formats (Summary, Detailed, Performance, Confidence)

### 📊 Benchmarking
Hypothesis validation with dual-mode operation:
- **Demo Mode**: Statistical simulation for quick demonstration
- **Real Mode**: Tests actual Felix components (HelixGeometry, CentralPost, ContextCompressor)
- **H1**: Helical Progression (20% improvement target)
- **H2**: Hub-Spoke Efficiency (15% gain target)
- **H3**: Memory Compression (25% improvement target)

---

## Installation

### Prerequisites
- Python 3.8+
- Felix Framework installed
- Virtual environment activated

### Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

**Dependencies:**
- `streamlit>=1.28.0`
- `plotly>=5.17.0`
- `pandas>=2.0.0`
- `pyyaml>=6.0`

---

## Quick Start

### Run Streamlit GUI Alone

```bash
# From project root
streamlit run streamlit_app.py

# Or using Python launcher
python run_streamlit_gui.py
```

The GUI will open in your browser at `http://localhost:8501`

### Run Both GUIs Together

**Windows:**
```bash
run_both_guis.bat
```

**Linux/Mac:**
```bash
chmod +x run_both_guis.sh
./run_both_guis.sh
```

### Typical Workflow

1. **Start tkinter GUI** → Initialize Felix system → Configure parameters
2. **Start Streamlit GUI** → Monitor system → View real-time metrics
3. **Run workflows in tkinter** → Monitor progress in Streamlit
4. **Analyze results in Streamlit** → Export reports → Adjust settings

---

## Architecture Overview

### Dual-GUI Design

Felix uses complementary interfaces for different purposes:

| Aspect | tkinter GUI | Streamlit GUI |
|--------|-------------|---------------|
| **Role** | System Control | System Monitoring |
| **Database** | Read/Write | Read-Only |
| **Felix System** | Start/Stop | Monitor Only |
| **Best For** | Real-time Control | Analytics & Visualization |

### Shared Resources

Both GUIs access the same databases:
- `felix_knowledge.db` - Agent knowledge entries
- `felix_memory.db` - Task memory patterns
- `felix_task_memory.db` - Workflow execution history

### Non-Interference Design

The Streamlit GUI ensures zero interference through:
- **Separate Directory**: `streamlit_gui/` isolated from `src/gui/`
- **Read-Only Access**: SQLite read-only connections
- **Import-Only Pattern**: Uses Felix classes without modification
- **Independent Process**: Runs in separate Python process

---

## Directory Structure

```
streamlit_gui/
├── streamlit_app.py           # Entry point
├── pages/                      # Streamlit multipage app
│   ├── 1_Dashboard.py
│   ├── 2_Configuration.py
│   ├── 3_Testing.py
│   └── 4_Benchmarking.py
├── backend/                    # Backend modules
│   ├── system_monitor.py      # Felix system monitoring
│   ├── db_reader.py           # Database read operations
│   ├── config_handler.py      # Configuration management
│   └── benchmark_runner.py    # Benchmark execution
└── components/                 # Reusable UI components
    ├── metrics_display.py
    ├── agent_visualizer.py
    └── results_analyzer.py
```

---

## Usage Tips

### When to Use Each GUI

**Use tkinter GUI for:**
- Starting/stopping Felix system
- Spawning agents
- Modifying configurations
- Executing workflows

**Use Streamlit GUI for:**
- Monitoring real-time metrics
- Analyzing historical data
- Visualizing agent interactions
- Running benchmarks
- Generating reports

### Benchmark Modes

- **Demo Mode**: Fast (~10 seconds), uses statistical simulation, no Felix components required
- **Real Mode**: Slower, tests actual Felix components, provides accurate validation

Use Demo mode for quick demonstrations, Real mode for actual hypothesis validation.

### Real Data vs Simulated Data

- **Dashboard/Testing pages**: Always use real data from databases
- **Benchmarking page**: Clearly labeled with data source (REAL or SIMULATED)
- If no data available: Pages show helpful messages explaining how to generate data

---

## Troubleshooting

### Port Already in Use

```bash
# Use alternative port
streamlit run streamlit_app.py --server.port 8502
```

### Database Not Found

- Ensure Felix system has been run at least once to create databases
- Check that you're running from the project root directory
- Verify database files exist: `felix_knowledge.db`, `felix_memory.db`, `felix_task_memory.db`

### No Data Displayed

1. Start Felix system from tkinter GUI
2. Run some workflows to generate data
3. Refresh Streamlit GUI (press R in browser)
4. Check "Database Status" tab on Dashboard for connection info

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements_streamlit.txt --force-reinstall
```

---

## Documentation

- **[Architecture](streamlit_gui_architecture.md)** - Detailed architecture with diagrams
- **[Integration Summary](INTEGRATION_SUMMARY.md)** - Full feature and integration details
- **[Main README](../README.md)** - Felix Framework overview

---

## Performance

- Dashboard refresh: < 500ms
- Database queries: < 150ms
- Page load: < 1 second
- Memory usage: ~100-200 MB

---

## Contributing

When contributing to the Streamlit GUI:

1. **Maintain Read-Only Pattern**: Never modify shared databases
2. **Use Caching**: Leverage Streamlit's `@st.cache_resource` and `@st.cache_data`
3. **Handle Errors Gracefully**: Provide fallbacks for missing data
4. **Follow Structure**: Place pages in `pages/`, backend logic in `backend/`, components in `components/`
5. **Test Isolation**: Ensure GUI works without Felix running

---

## License

Part of the Felix Framework project. See main [LICENSE](../LICENSE) file.
