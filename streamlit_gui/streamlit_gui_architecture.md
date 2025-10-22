# Streamlit GUI Architecture

**Version**: 4.1
**Last Updated**: 2025-10-22
**Status**: Production

---

## Overview

The Streamlit GUI is a read-only monitoring and visualization interface for the Felix Framework, complementing the tkinter control GUI. It provides real-time system monitoring, performance analytics, and hypothesis validation without interfering with the running Felix system.

---

## System Architecture

### Dual-GUI Design

Felix employs a dual-GUI architecture where each interface serves distinct purposes:

```mermaid
graph TB
    subgraph "Control Layer"
        TK[tkinter GUI<br/>Control Interface]
    end

    subgraph "Monitoring Layer"
        ST[Streamlit GUI<br/>Read-Only Monitor]
    end

    subgraph "Shared Resources"
        DB1[(felix_knowledge.db)]
        DB2[(felix_memory.db)]
        DB3[(felix_task_memory.db)]
    end

    subgraph "Core System"
        FS[Felix System<br/>CentralPost + Agents]
    end

    TK -->|Read/Write| DB1
    TK -->|Read/Write| DB2
    TK -->|Read/Write| DB3
    TK -->|Start/Stop/Control| FS

    ST -->|Read-Only| DB1
    ST -->|Read-Only| DB2
    ST -->|Read-Only| DB3
    ST -.->|Monitor Only| FS

    FS -->|Write| DB1
    FS -->|Write| DB2
    FS -->|Write| DB3

    style TK fill:#e1f5ff
    style ST fill:#ffe1f5
    style FS fill:#f0f0f0
```

**Operational Modes:**

| Aspect | tkinter GUI | Streamlit GUI |
|--------|-------------|---------------|
| Primary Role | System Control | System Monitoring |
| Database Access | Read/Write | Read-Only |
| Felix System | Start/Stop | Monitor Only |
| Agent Spawning | Yes | View Only |
| Configuration | Modify | View/Export |
| Workflows | Execute | Analyze Results |

### Non-Interference Design

The Streamlit GUI achieves zero interference through:

1. **Separate Directory Structure**: `streamlit_gui/` isolated from `src/gui/`
2. **Read-Only Database Access**: Uses SQLite read-only connections
3. **Import Without Modification**: Uses existing Felix classes via imports only
4. **Separate Configuration**: Uses `streamlit_config.yaml` (tkinter uses `felix_gui_config.json`)
5. **Independent Entry Point**: `streamlit_app.py` separate from tkinter's entry point

---

## Component Architecture

### Directory Structure & Dependencies

```mermaid
graph TB
    subgraph "Entry Point"
        APP[streamlit_app.py]
    end

    subgraph "Pages Layer"
        P1[1_Dashboard.py]
        P2[2_Configuration.py]
        P3[3_Testing.py]
        P4[4_Benchmarking.py]
    end

    subgraph "Backend Layer"
        SM[system_monitor.py<br/>SystemMonitor]
        DBR[db_reader.py<br/>DatabaseReader]
        CH[config_handler.py<br/>ConfigHandler]
        BR[benchmark_runner.py<br/>BenchmarkRunner]
        RBR[real_benchmark_runner.py<br/>RealBenchmarkRunner]
    end

    subgraph "Components Layer"
        MD[metrics_display.py]
        AV[agent_visualizer.py]
        LM[log_monitor.py]
        CV[config_viewer.py]
        RA[results_analyzer.py]
    end

    subgraph "Data Layer"
        DB1[(felix_knowledge.db)]
        DB2[(felix_memory.db)]
        DB3[(felix_task_memory.db)]
    end

    APP --> P1
    APP --> P2
    APP --> P3
    APP --> P4

    P1 --> SM
    P1 --> DBR
    P1 --> MD
    P1 --> AV

    P2 --> CH
    P2 --> CV

    P3 --> SM
    P3 --> DBR
    P3 --> RA

    P4 --> BR
    P4 --> RBR

    SM --> DB1
    SM --> DB2
    SM --> DB3
    DBR --> DB1
    DBR --> DB2
    DBR --> DB3

    style APP fill:#ff9999
    style P1 fill:#99ccff
    style P2 fill:#99ccff
    style P3 fill:#99ccff
    style P4 fill:#99ccff
    style SM fill:#99ff99
    style DBR fill:#99ff99
```

---

## Data Flow Architecture

### Dashboard Data Flow

```mermaid
sequenceDiagram
    participant User
    participant Dashboard
    participant SystemMonitor
    participant DatabaseReader
    participant KnowledgeDB as felix_knowledge.db
    participant TaskDB as felix_task_memory.db

    User->>Dashboard: Load Dashboard
    Dashboard->>SystemMonitor: get_system_metrics()
    SystemMonitor->>KnowledgeDB: Query knowledge_entries
    KnowledgeDB-->>SystemMonitor: Raw entries
    SystemMonitor->>SystemMonitor: Calculate avg confidence<br/>(CASE confidence_level)
    SystemMonitor-->>Dashboard: Metrics dict

    Dashboard->>DatabaseReader: get_agent_metrics()
    DatabaseReader->>KnowledgeDB: Query with GROUP BY source_agent
    KnowledgeDB-->>DatabaseReader: Aggregated data
    DatabaseReader-->>Dashboard: Pandas DataFrame

    Dashboard->>SystemMonitor: get_workflow_results()
    SystemMonitor->>TaskDB: Query task_executions
    TaskDB-->>SystemMonitor: Workflow records
    SystemMonitor->>SystemMonitor: Parse success_metrics_json
    SystemMonitor-->>Dashboard: Workflow list

    Dashboard->>User: Render metrics & charts
```

### Configuration Data Flow

```mermaid
sequenceDiagram
    participant User
    participant ConfigPage as Configuration Page
    participant ConfigHandler
    participant FileSystem
    participant Plotly

    User->>ConfigPage: Select config source
    ConfigPage->>ConfigHandler: load_config_file(path)
    ConfigHandler->>FileSystem: Read YAML/JSON file
    FileSystem-->>ConfigHandler: Raw config content
    ConfigHandler->>ConfigHandler: Parse YAML/JSON
    ConfigHandler-->>ConfigPage: Config dict

    ConfigPage->>ConfigPage: Extract helix params
    ConfigPage->>Plotly: create_helix_3d_visualization()
    Plotly->>Plotly: Generate 3D helix mesh
    Plotly-->>ConfigPage: Plotly figure

    ConfigPage->>User: Display config + 3D viz

    User->>ConfigPage: Export config
    ConfigPage->>ConfigHandler: Convert to format
    ConfigHandler-->>ConfigPage: YAML/JSON string
    ConfigPage->>User: Download file
```

### Workflow Results Data Flow

```mermaid
sequenceDiagram
    participant User
    participant TestingPage as Testing Page
    participant SystemMonitor
    participant TaskMemoryDB as felix_task_memory.db

    User->>TestingPage: View Workflow Results
    TestingPage->>SystemMonitor: get_workflow_results(limit=50)

    alt task_executions table exists
        SystemMonitor->>TaskMemoryDB: Query task_executions
        TaskMemoryDB-->>SystemMonitor: Rows with JSON fields
        SystemMonitor->>SystemMonitor: Parse success_metrics_json
        Note right of SystemMonitor: Extract:<br/>- avg_confidence<br/>- agent_count
        SystemMonitor->>SystemMonitor: Parse agents_used_json
        SystemMonitor-->>TestingPage: Workflow list with metrics
    else Fallback to task_patterns
        SystemMonitor->>TaskMemoryDB: Query task_patterns
        TaskMemoryDB-->>SystemMonitor: Pattern records
        SystemMonitor-->>TestingPage: Basic workflow info
    end

    TestingPage->>User: Render workflow table
```

### Benchmarking Data Flow

```mermaid
sequenceDiagram
    participant User
    participant BenchmarkPage as Benchmarking Page
    participant RealBenchmarkRunner
    participant FelixComponents as Felix Components<br/>(HelixGeometry, CentralPost,<br/>ContextCompressor)
    participant Simulator as Statistical Simulator

    User->>BenchmarkPage: Select mode (Demo/Real)
    BenchmarkPage->>RealBenchmarkRunner: is_real_mode_available()

    alt Real mode available
        RealBenchmarkRunner->>FelixComponents: Check imports
        FelixComponents-->>RealBenchmarkRunner: Components available
        RealBenchmarkRunner-->>BenchmarkPage: True
    else Real mode unavailable
        RealBenchmarkRunner-->>BenchmarkPage: False
    end

    User->>BenchmarkPage: Run H1 benchmark
    BenchmarkPage->>RealBenchmarkRunner: validate_hypothesis_h1_real(samples)

    alt Real mode ON
        RealBenchmarkRunner->>FelixComponents: Create HelixGeometry
        RealBenchmarkRunner->>FelixComponents: Test helical vs linear
        FelixComponents-->>RealBenchmarkRunner: Real performance data
        RealBenchmarkRunner-->>BenchmarkPage: {data_source: "REAL", ...}
    else Fallback to Demo mode
        RealBenchmarkRunner->>Simulator: Generate statistical data
        Simulator-->>RealBenchmarkRunner: Simulated data
        RealBenchmarkRunner-->>BenchmarkPage: {data_source: "SIMULATED", ...}
    end

    BenchmarkPage->>User: Display results with source badge
```

---

## Core Features

### Confidence Calculation

The system maps TEXT confidence levels to numeric values for meaningful chart visualization:

| Database Value | Numeric Value | Percentage |
|----------------|---------------|------------|
| `"low"` | 0.3 | 30% |
| `"medium"` | 0.6 | 60% |
| `"high"` | 0.9 | 90% |
| `NULL` / Other | 0.5 | 50% (default) |

**SQL Implementation:**
```sql
CASE confidence_level
    WHEN 'low' THEN 0.3
    WHEN 'medium' THEN 0.6
    WHEN 'high' THEN 0.9
    ELSE 0.5
END
```

This replaced the previous incorrect use of `success_rate` column (always 1.0), enabling realistic confidence visualization ranging from 30-90%.

### Agent Awareness

The system tracks agents by helical phase for monitoring convergence:

```mermaid
graph TB
    subgraph "Phase Classification"
        P1[Exploration<br/>depth 0.0-0.3<br/>Wide radius exploration]
        P2[Analysis<br/>depth 0.3-0.7<br/>Converging analysis]
        P3[Synthesis<br/>depth 0.7-1.0<br/>Focused synthesis]
    end

    subgraph "Convergence Detection"
        CHECK{depth ≥ 0.7<br/>AND<br/>confidence ≥ 0.8?}
        READY[Synthesis Ready]
        TRIGGER[CentralPost<br/>Triggers Synthesis]
    end

    P1 --> P2
    P2 --> P3
    P3 --> CHECK
    CHECK -->|Yes| READY
    READY --> TRIGGER
    CHECK -->|No| P1

    style P1 fill:#99ff99
    style P2 fill:#99ccff
    style P3 fill:#ffcc99
    style TRIGGER fill:#ff9999
```

**Capabilities:**
- Monitor agents by helical phase (exploration/analysis/synthesis)
- Track convergence status (threshold: confidence ≥ 0.8, depth ≥ 0.7)
- Infer agent position from domain and activity patterns

### Incremental Token Streaming

Real-time token-by-token display for live agent output monitoring:

**Configuration:**
- `enable_streaming`: Boolean flag in FelixConfig
- `streaming_batch_interval`: Time between UI updates (default: 0.1s)
- Supports multiple concurrent streams from different agents

**Status Monitoring:**
```python
system_monitor.get_streaming_status()
# Returns: {
#     'enabled': bool,
#     'active_streams': int,
#     'partial_thoughts': [...]
# }
```

---

## Hypothesis Validation Architecture

### Combined Hypothesis Overview

```mermaid
graph TB
    subgraph "H1: Helical Progression 20% gain"
        H1A[Linear agents<br/>Same behavior all depths]
        H1B[Helical agents<br/>Adapt by position]
        H1C[Workload Distribution<br/>Std Deviation]
        H1A --> H1C
        H1B --> H1C
    end

    subgraph "H2: Hub-Spoke Efficiency 15% gain"
        H2A[Mesh O N²<br/>All-to-all messages]
        H2B[Hub-Spoke O N<br/>CentralPost routing]
        H2C[Message Latency<br/>Throughput]
        H2A --> H2C
        H2B --> H2C
    end

    subgraph "H3: Memory Compression 25% gain"
        H3A[Raw Context<br/>10,000 tokens]
        H3B[Compressed Context<br/>3,000 tokens 0.3 ratio]
        H3C[Attention Focus<br/>Information Retention]
        H3A --> H3C
        H3B --> H3C
    end

    subgraph "Real Mode Testing"
        REAL1[HelixGeometry]
        REAL2[CentralPost]
        REAL3[ContextCompressor]
    end

    H1C -.->|Tests with| REAL1
    H2C -.->|Tests with| REAL2
    H3C -.->|Tests with| REAL3

    style H1B fill:#99ff99
    style H2B fill:#99ccff
    style H3B fill:#ffcc99
    style REAL1 fill:#e1f5ff
    style REAL2 fill:#e1f5ff
    style REAL3 fill:#e1f5ff
```

**Benchmark Modes:**
- **Demo Mode**: Simulated data using statistical models for quick demonstration
- **Real Mode**: Tests actual Felix components (HelixGeometry, CentralPost, ContextCompressor)
- Each result clearly labeled with data source (REAL vs SIMULATED)

---

## Safety & Resilience

### Path Resolution

Databases are located using absolute path resolution from any working directory:

```python
# Resolve project root from backend file location
backend_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_gui_dir = os.path.dirname(backend_dir)
project_root = os.path.dirname(streamlit_gui_dir)

# Construct absolute paths
self.knowledge_db_path = os.path.join(project_root, "felix_knowledge.db")
self.memory_db_path = os.path.join(project_root, "felix_memory.db")
self.task_memory_db_path = os.path.join(project_root, "felix_task_memory.db")
```

**Benefits**: Works from any directory, consistent behavior, no relative path assumptions.

### Error Handling Strategy

```mermaid
flowchart TD
    Operation[Database Operation] --> Try[Try primary approach]
    Try --> Success1{Success?}

    Success1 -->|Yes| Return[Return data]
    Success1 -->|No| Fallback1{Fallback<br/>available?}

    Fallback1 -->|Yes| TryFallback[Try fallback]
    Fallback1 -->|No| EmptyData[Return empty structure]

    TryFallback --> Success2{Success?}
    Success2 -->|Yes| Return
    Success2 -->|No| EmptyData

    EmptyData --> UserMsg[Show user-friendly message]
    Return --> Display[Display data]
    UserMsg --> Display

    style Success1 fill:#99ff99
    style EmptyData fill:#ffcc99
    style UserMsg fill:#99ccff
```

**Fallback Levels:**
1. **Database Connection**: Read-only mode → Regular connection → Empty DataFrame
2. **Query Execution**: Primary table → Fallback table → Simulated data
3. **Data Parsing**: JSON parse → Safe extraction → Default values
4. **UI Rendering**: Real data → Placeholder → User message

### Security & Integration

**Read-Only Enforcement:**
- SQLite read-only URI mode: `file:path?mode=ro` prevents writes at database level
- No write methods exposed in backend classes (no `insert()`, `update()`, `delete()`)
- Separate directory structure prevents accidental modification of control GUI
- Import-only pattern uses existing Felix classes without modification

**Felix Core Integration:**

```mermaid
graph TB
    subgraph "Streamlit GUI"
        SM[SystemMonitor]
        RBR[RealBenchmarkRunner]
    end

    subgraph "Felix Core"
        CP[CentralPost]
        HG[HelixGeometry]
        CC[ContextCompressor]
        AG[SpecializedAgents]
    end

    SM -.->|Import & Monitor| CP
    RBR -->|Import & Test| HG
    RBR -->|Import & Test| CC
    RBR -->|Import & Test| AG

    style SM fill:#ffe1f5
    style RBR fill:#ffe1f5
```

**Integration Principles:**
1. Zero changes to `src/` directory
2. Monitoring only (no control operations)
3. Real component testing when available
4. Graceful fallback to simulated data

---

## Deployment Architecture

### Multi-Process Operation

```mermaid
graph TB
    subgraph "Terminal 1"
        T1[tkinter GUI Process<br/>python -m src.gui]
    end

    subgraph "Terminal 2"
        T2[Streamlit Process<br/>streamlit run streamlit_app.py]
    end

    subgraph "Browser"
        B[Streamlit UI<br/>localhost:8501]
    end

    subgraph "Shared State"
        DB[(Databases)]
        LS[LM Studio<br/>localhost:1234]
    end

    T1 -->|Read/Write| DB
    T1 -->|LLM Requests| LS

    T2 -->|Read-Only| DB
    T2 -->|Serves| B

    B -->|User Interaction| T2

    style T1 fill:#e1f5ff
    style T2 fill:#ffe1f5
    style DB fill:#f0f0f0
```

**Process Characteristics:**

- **Independent Processes**: Can start/stop independently without affecting each other
- **No Process Communication**: Communicate only via shared databases (no sockets/pipes)
- **Crash Isolation**: Streamlit crash doesn't affect tkinter GUI or Felix system
- **Resource Isolation**: Each process has own memory space and event loop

**Typical Workflow:**
1. Start tkinter GUI → Initialize Felix system → Configure parameters
2. Start Streamlit GUI → Monitor system → View real-time metrics
3. Run workflows in tkinter → Monitor progress in Streamlit
4. Analyze results in Streamlit → Export reports → Adjust settings in tkinter

---

## Future Architecture Considerations

### Planned Enhancements

```mermaid
mindmap
    root((Streamlit GUI<br/>Future))
        Real-Time Streaming
            WebSocket connection
            Live token display
            Progress bars
        Enhanced Awareness
            Direct AgentRegistry access
            Live collaboration graph
            Phase transition animations
        Advanced Analytics
            ML-powered insights
            Anomaly detection
            Predictive modeling
        Export Capabilities
            Custom report templates
            Automated scheduling
            Multi-format support
```

---

## Summary

The Streamlit GUI architecture provides a robust, safe, and performant monitoring layer for the Felix Framework:

**Key Achievements:**

- ✅ Zero interference with Felix core system
- ✅ Read-only safety enforced at multiple levels
- ✅ Absolute path resolution works from any directory
- ✅ Comprehensive error handling with graceful degradation
- ✅ Real-time data visualization with sub-second latency
- ✅ Dual-mode benchmarking (simulated and real components)
- ✅ Agent awareness integration with phase tracking
- ✅ Streaming support for real-time token display

**Design Principles:**

1. **Separation of Concerns**: Monitoring vs Control - each GUI has clear responsibilities
2. **Safety First**: Read-only by default with multiple enforcement layers
3. **Resilience**: Multiple fallback strategies ensure GUI always works
4. **Performance**: Caching and optimization for sub-second response times
5. **User Experience**: Informative errors and real-time feedback

---

**Document Version**: 4.1
**Architecture Status**: Production-Ready
**Last Validated**: 2025-10-22
