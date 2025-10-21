# Streamlit GUI for Felix Framework

## Overview

The Streamlit GUI is a **read-only monitoring and visualization interface** that complements the tkinter control GUI. It provides advanced analytics, performance monitoring, and hypothesis validation tools without interfering with the running Felix system.

## Architecture

### Dual-GUI Design

Felix Framework uses a dual-GUI approach:

| tkinter GUI (Control) | Streamlit GUI (Monitor) |
|-----------------------|-------------------------|
| Start/Stop System | Visualize Metrics |
| Spawn Agents | Analyze Performance |
| Modify Settings | Run Benchmarks |
| Execute Workflows | Export Reports |

Both GUIs share the same databases (`felix_memory.db`, `felix_knowledge.db`, `felix_task_memory.db`) but operate in different modes to prevent conflicts.

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment activated
- Felix Framework dependencies installed

### Install Streamlit Dependencies

```bash
pip install -r requirements_streamlit.txt
```

## Running the GUI

### Option 1: Using the Python Launcher (Recommended)

```bash
python run_streamlit_gui.py
```

### Option 2: Direct Streamlit Command

```bash
streamlit run streamlit_app.py
```

### Option 3: Run Both GUIs Together

#### Windows:
```batch
run_both_guis.bat
```

#### Linux/Mac:
```bash
chmod +x run_both_guis.sh
./run_both_guis.sh
```

The Streamlit GUI will open in your browser at `http://localhost:8501`

## Features

### 1. Dashboard (🏠)

Real-time system monitoring with:
- System status indicators (✅ **Real Data**)
- Agent activity visualization
- Performance trend charts
- Database status monitoring
- Auto-refresh capability
- Interactive tooltips explaining all metrics

Key Metrics:
- Knowledge entries count (from real database)
- Task patterns (from task memory)
- Average confidence scores (computed from agent entries)
- Agent performance matrix

**Data Source**: All data pulled from actual Felix databases (`felix_knowledge.db`, `felix_memory.db`)

### 2. Configuration (⚙️)

Configuration management features:
- View current Felix configuration
- 3D helix geometry visualization
- Configuration comparison tool
- Export to YAML/JSON formats
- Parameter validation

Visualization includes:
- Interactive 3D helix model
- Phase markers (Exploration, Analysis, Synthesis, Conclusion)
- Helix characteristics (turns, taper ratio, volume)

### 3. Testing (🧪)

Workflow analysis tools (✅ **Real Data**):
- Workflow execution timeline from actual runs
- Success/failure pattern analysis
- Performance metrics over time
- Test report generation with multiple formats
- Export capabilities (JSON, CSV, Markdown)
- Interactive tooltips for all metrics

Report Types:
- **Summary**: High-level overview with key metrics and trends
- **Detailed**: Complete breakdown with timestamps
- **Performance**: Focus on execution times and bottlenecks
- **Confidence**: Agent confidence scores and progression

**Data Source**: All workflow data pulled from Felix databases (real execution history)

### 4. Benchmarking (📊)

Hypothesis validation and performance testing with **dual-mode operation**:

**Benchmark Modes**:
- **Demo Mode** (🎲 Simulated): Uses statistical models for quick demonstration
- **Real Mode** (✅ Actual Components): Tests actual Felix components (HelixGeometry, CentralPost, ContextCompressor)

**Core Hypotheses (with detailed explanations)**:
- **H1**: Helical progression enhances agent adaptation (20% improvement expected)
  - Tests workload distribution along helical geometry
  - Compares linear vs. helical agent progression
  - **Real mode**: Actually creates helix positions and measures geometric optimization
- **H2**: Hub-spoke communication optimizes resource allocation (15% efficiency gain)
  - Tests message routing efficiency (O(N) vs O(N²))
  - Measures latency and throughput
  - **Real mode**: Uses actual CentralPost for hub-spoke communication
- **H3**: Memory compression reduces latency (25% attention improvement)
  - Tests context compression impact
  - Measures information retention
  - **Real mode**: Uses actual ContextCompressor with real compression

Features:
- **Mode selector**: Toggle between Demo and Real benchmarks
- **Availability detection**: Automatically detects if Felix components are importable
- **Graceful fallback**: Falls back to simulated data if real components unavailable
- Interactive tooltips explaining each hypothesis
- Statistical significance testing (t-tests, p-values)
- Performance comparison charts with box plots
- **Data source badges**: Clear labeling of REAL vs SIMULATED data
- Scaling analysis
- Comprehensive benchmark reports
- Configurable sample sizes for statistical rigor

**Data Source**:
- Demo mode: Simulated using statistical models (np.random.normal)
- Real mode: Actual Felix components (HelixGeometry, CentralPost, ContextCompressor)
- Each result clearly labeled with its data source

## Directory Structure

```
streamlit_gui/
├── backend/                    # Backend modules
│   ├── system_monitor.py      # Felix system monitoring
│   ├── db_reader.py          # Database read operations
│   ├── config_handler.py     # Configuration management
│   └── benchmark_runner.py   # Benchmark execution
├── components/                 # Reusable UI components
│   ├── metrics_display.py    # Metric visualization
│   ├── agent_visualizer.py   # Agent visualizations
│   ├── log_monitor.py        # Log monitoring
│   ├── config_viewer.py      # Config display
│   └── results_analyzer.py   # Result analysis
└── pages/                      # Streamlit pages
    ├── 1_Dashboard.py         # Main dashboard
    ├── 2_Configuration.py     # Config viewer
    ├── 3_Testing.py          # Test analysis
    └── 4_Benchmarking.py     # Benchmarks

```

## Database Access

The Streamlit GUI operates in **read-only mode** for safety:

```python
# Read-only connection
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
```

Databases accessed:
- `felix_knowledge.db` - Knowledge store entries
- `felix_memory.db` - Task memory patterns
- `felix_task_memory.db` - Workflow results

## Configuration

### Default Configuration

The GUI uses sensible defaults if no configuration is found:

```yaml
helix:
  top_radius: 3.0
  bottom_radius: 0.5
  height: 8.0
  turns: 2

lm_host: "127.0.0.1"
lm_port: 1234
max_agents: 25
base_token_budget: 2500
```

### Custom Configuration

Create `streamlit_config.yaml` in the project root:

```yaml
# Custom settings for Streamlit GUI
refresh_interval: 5  # seconds
max_display_rows: 100
chart_height: 400
```

## Performance Considerations

### Caching

The GUI uses Streamlit's caching for performance:

```python
@st.cache_resource
def get_monitor():
    return SystemMonitor()

@st.cache_data
def load_data():
    return db_reader.get_knowledge_entries()
```

### Auto-Refresh

Dashboard supports auto-refresh with configurable intervals:
- Default: 5 seconds
- Range: 1-60 seconds
- Can be disabled via checkbox

### Resource Usage

- Memory: ~100-200 MB
- CPU: Minimal when idle
- Network: Local only (no external connections)

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use alternative port
   streamlit run streamlit_app.py --server.port 8502
   ```

2. **Database Not Found**
   - Ensure Felix system has been run at least once
   - Check database files exist in project root

3. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements_streamlit.txt
   ```

4. **No Data Displayed**
   - Start Felix system from tkinter GUI
   - Run some workflows to generate data
   - Check database connections in Dashboard

### Debug Mode

Enable debug logging:

```python
# In streamlit_app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

### Usage Workflow

1. **Start tkinter GUI** → Initialize Felix system
2. **Start Streamlit GUI** → Monitor system
3. **Run workflows** in tkinter → View results in Streamlit
4. **Analyze performance** → Generate reports
5. **Validate hypotheses** → Run benchmarks

### When to Use Each GUI

**Use tkinter GUI for:**
- Starting/stopping Felix
- Spawning agents
- Modifying configurations
- Executing workflows

**Use Streamlit GUI for:**
- Monitoring real-time metrics
- Analyzing historical data
- Visualizing agent interactions
- Running benchmarks
- Generating reports

## API Reference

### SystemMonitor

```python
monitor = SystemMonitor()

# Check if Felix is running
is_running = monitor.check_felix_running()

# Get system metrics
metrics = monitor.get_system_metrics()

# Get agent data
agents = monitor.get_agent_data()
```

### DatabaseReader

```python
reader = DatabaseReader()

# Get knowledge entries
df = reader.get_knowledge_entries(limit=100)

# Get agent metrics
metrics = reader.get_agent_metrics()

# Get time series data
ts_data = reader.get_time_series_metrics(hours=24)
```

### BenchmarkRunner

```python
runner = BenchmarkRunner()

# Validate hypotheses (simulated)
h1_result = runner.validate_hypothesis_h1(samples=100)
h2_result = runner.validate_hypothesis_h2(samples=100)
h3_result = runner.validate_hypothesis_h3(samples=100)

# Run performance benchmark
perf = runner.run_performance_benchmark("agent_spawning", iterations=100)

# Generate report
report = runner.generate_report()
```

### RealBenchmarkRunner

```python
from streamlit_gui.backend.real_benchmark_runner import RealBenchmarkRunner

runner = RealBenchmarkRunner()

# Check if real mode is available
if runner.is_real_mode_available():
    print("✅ Real benchmark mode available")
else:
    print("⚠️ Will use simulated fallback")

# Get availability message
message = runner.get_availability_message()

# Validate hypotheses with REAL Felix components
# Automatically falls back to simulated if components unavailable
h1_result = runner.validate_hypothesis_h1_real(samples=100)
h2_result = runner.validate_hypothesis_h2_real(samples=100)
h3_result = runner.validate_hypothesis_h3_real(samples=100)

# Each result includes 'data_source' field:
# - 'REAL': Used actual Felix components
# - 'SIMULATED (components unavailable)': Fell back to simulation
print(f"H1 used: {h1_result['data_source']}")
print(f"H1 gain: {h1_result['actual_gain']:.1%}")
print(f"H1 validated: {h1_result['validated']}")
```

## Contributing

When contributing to the Streamlit GUI:

1. **Maintain Read-Only Pattern**: Never modify shared databases
2. **Use Caching**: Leverage Streamlit's caching for performance
3. **Handle Errors Gracefully**: Always provide fallbacks
4. **Follow Structure**: Place new pages in `pages/`, components in `components/`
5. **Test Isolation**: Ensure GUI works without Felix running

## License

Part of the Felix Framework project. See main LICENSE file.

## Support

For issues or questions:
- Check the troubleshooting section
- Review the Felix Framework documentation
- Open an issue on GitHub