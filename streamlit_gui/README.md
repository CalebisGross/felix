# Streamlit GUI for Felix Framework

Read-only monitoring and visualization interface for the Felix multi-agent AI framework. Provides real-time analytics, workflow history, and hypothesis validation benchmarking.

## Features

- **Dashboard**: Real-time monitoring with agent metrics, workflow history browser, and performance analytics
- **Configuration**: View and export helix parameters with interactive 3D visualization
- **Testing**: Analyze workflow execution history with detailed reports and search capabilities
- **Benchmarking**: Validate H1/H2/H3 hypotheses using real Felix component tests

## Installation

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements_streamlit.txt
```

## Quick Start

```bash
# Run Streamlit GUI
streamlit run streamlit_gui/app.py

# Or using launcher
python run_streamlit_gui.py

# Run both GUIs together (Windows)
run_both_guis.bat

# Run both GUIs together (Linux/Mac)
./run_both_guis.sh
```

Opens at `http://localhost:8501`

## Pages

### Dashboard
- **System Metrics**: Knowledge entries, agent count, workflow success rate
- **Workflow History**: Browse, search, and filter past executions with synthesis outputs
- **Performance Trends**: Agent activity over time, confidence distribution charts
- **Agent Performance**: Metrics by type and phase with efficiency analysis

### Configuration
- **Helix Geometry**: Interactive 3D visualization with parameter display
- **System Settings**: Token budgets, temperature gradients, feature toggles
- **Web Search Config**: Provider settings, blocked domains, confidence thresholds
- **Export Options**: YAML/JSON configuration export

### Testing
- **Workflow Browser**: Search and filter execution history
- **Detailed Views**: Task input, synthesis output, agent details, performance metrics
- **Truth Assessment**: Validation badges for workflows using verification
- **Report Generation**: Export summary, detailed, performance, and confidence reports

### Benchmarking
- **Hypothesis Testing**: Validate H1 (20%), H2 (15%), H3 (25%) improvement targets
- **Real Mode**: Tests actual Felix components (HelixGeometry, CentralPost, ContextCompressor)
- **Configuration**: Select hypotheses, set iterations, toggle real LLM usage
- **Results Display**: Metrics table, box plots, JSON export

## Key Components

### New Monitoring Features
- **Web Search Monitor**: Track Research agent queries with DuckDuckGo/SearxNG integration
- **Workflow History Viewer**: Advanced browser with search, filters, and detailed views
- **Truth Assessment Display**: Automatic validation detection and confidence-based badges

### Backend Services
- **DatabaseReader**: Read-only SQLite access to Felix databases
- **SystemMonitor**: Real-time Felix system state monitoring
- **BenchmarkRunner**: Subprocess execution of `tests/run_hypothesis_validation.py`
- **ConfigHandler**: YAML/JSON configuration management

### Databases (Read-Only)
- `felix_knowledge.db`: Agent knowledge entries and insights
- `felix_memory.db`: Task memory patterns
- `felix_task_memory.db`: GUI task storage
- `felix_workflow_history.db`: Complete workflow execution records

## Directory Structure

```
streamlit_gui/
├── app.py                    # Entry point (v3.0.0)
├── pages/                    # Streamlit multipage app
│   ├── 1_Dashboard.py
│   ├── 2_Configuration.py
│   ├── 3_Testing.py
│   └── 4_Benchmarking.py
├── backend/                  # Backend services
│   ├── system_monitor.py
│   ├── db_reader.py
│   ├── config_handler.py
│   ├── benchmark_runner.py
│   └── real_benchmark_runner.py
├── components/               # UI components
│   ├── metrics_display.py
│   ├── agent_visualizer.py
│   ├── results_analyzer.py
│   ├── web_search_monitor.py
│   ├── workflow_history_viewer.py
│   └── truth_assessment_display.py
├── tests/                    # Component tests
└── docs/                     # Detailed documentation
```

## Usage Guide

### Workflow Monitoring
1. Start tkinter GUI and initialize Felix system
2. Run workflows to generate data
3. Open Streamlit GUI to monitor in real-time
4. View synthesis outputs in Workflow History tab

### Running Benchmarks
1. Navigate to Benchmarking page
2. Select hypotheses to test (H1, H2, H3)
3. Configure iterations (10-100)
4. Toggle "Use Real LLM" if LM Studio is running
5. Click "Run Benchmarks" and wait for results

### Truth Assessment Workflows
Workflows containing keywords like "validate", "verify", "truth", "assess" automatically display validation badges:
- **✓ Validated** (green): Confidence ≥ 85%
- **⚠ Needs Review** (yellow): Confidence 70-84%
- **✗ Failed** (red): Confidence < 70%

## Troubleshooting

### No Data Displayed
- Run workflows from tkinter GUI first to generate data
- Check Dashboard → Database Status for connection info
- Refresh page (R in browser)

### Port Already in Use
```bash
streamlit run streamlit_gui/app.py --server.port 8502
```

### Import Errors
```bash
pip install -r requirements_streamlit.txt --force-reinstall
```

### Benchmark Timeouts
- Default timeout: 5 minutes per hypothesis
- Reduce iterations or disable "Use Real LLM" for faster tests

## Documentation

- [Architecture Guide](docs/streamlit_gui_architecture.md) - System design and data flow
- [Integration Summary](docs/INTEGRATION_SUMMARY.md) - Implementation phases and features
- [Development Notes](docs/DEVELOPMENT_NOTES.md) - Technical details and decisions

## Performance

- Page load: < 1 second
- Database queries: < 150ms
- Dashboard refresh: < 500ms
- Memory usage: ~100-200 MB

## License

Part of the Felix Framework project. See main LICENSE file.