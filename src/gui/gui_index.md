# Felix GUI Module - Internal Index

This index documents only the GUI module itself (`src/gui/`) - its files, dependencies, and internal structure. For Felix framework integration details, see the main project documentation.

## GUI Module Files

### Core Application Files

#### `__init__.py`
**Purpose**: Module exports and version information

**Exports**:
- `MainApp` - Main application class
- `FelixSystem`, `FelixConfig`, `AgentManager` - System management
- `DashboardFrame`, `WorkflowsFrame`, `MemoryFrame`, `AgentsFrame` - Tab frames
- `ThreadManager`, `DBHelper`, `logger` - Utilities
- `setup_gui_logging`, `TkinterTextHandler`, `QueueHandler`, `add_text_widget_to_logger` - Logging

**Internal Imports**:
```python
from .main import MainApp
from .felix_system import FelixSystem, FelixConfig, AgentManager
from .dashboard import DashboardFrame
from .workflows import WorkflowsFrame
from .memory import MemoryFrame
from .agents import AgentsFrame
from .utils import ThreadManager, DBHelper, logger
from .logging_handler import setup_gui_logging, TkinterTextHandler, QueueHandler, add_text_widget_to_logger
```

#### `__main__.py`
**Purpose**: Entry point for `python -m src.gui`

**Internal Imports**:
```python
from .main import MainApp
```

**Usage**: `python -m src.gui`

#### `main.py`
**Purpose**: Main application window with tabbed interface

**Class**: `MainApp(tk.Tk)`

**External Libraries**:
- `tkinter` (tk, ttk, messagebox)
- `json` - Config file handling
- `os` - File operations

**Internal Imports**:
```python
from .utils import ThreadManager, DBHelper, logger
from .dashboard import DashboardFrame
from .workflows import WorkflowsFrame
from .memory import MemoryFrame
from .agents import AgentsFrame
from .settings import SettingsFrame
from .felix_system import FelixSystem, FelixConfig
```

**Key Attributes**:
- `felix_system: FelixSystem` - System manager instance
- `system_running: bool` - System state flag
- `config_file: str` - "felix_gui_config.json"
- `lm_host: str` - LM Studio host
- `lm_port: int` - LM Studio port
- `notebook: ttk.Notebook` - Tab container
- `thread_manager: ThreadManager` - Thread manager
- `db_helper: DBHelper` - Database helper
- `status_var: tk.StringVar` - Status bar text

**Methods**:
- `start_system()` - Initialize and start Felix system
- `stop_system()` - Shutdown Felix system
- `_load_config()` - Load from felix_gui_config.json
- `_validate_system_health()` - Check system readiness
- `_enable_all_features()` - Enable features in all tabs
- `_disable_all_features()` - Disable features in all tabs

### Tab Frame Files

#### `dashboard.py`
**Purpose**: System control and log monitoring

**Class**: `DashboardFrame(ttk.Frame)`

**External Libraries**:
- `tkinter` (tk, ttk, scrolledtext, messagebox)
- `logging`

**Internal Imports**:
```python
from .utils import log_queue, logger
```

**Felix Framework Imports** (optional):
```python
from src.communication import central_post  # Fallback, not actively used
from src.llm import lm_studio_client        # Fallback, not actively used
```

**Key Attributes**:
- `main_app: MainApp` - Reference to main application
- `system_running: bool` - Local system state
- `log_text: ScrolledText` - Log display widget
- `start_button: ttk.Button` - Start button
- `stop_button: ttk.Button` - Stop button

**Methods**:
- `start_system()` - Delegate to main_app.start_system()
- `stop_system()` - Delegate to main_app.stop_system()
- `poll_log_queue()` - Poll log_queue every 100ms
- `_poll_system_ready()` - Poll for system startup
- `_poll_system_stopped()` - Poll for system shutdown
- `_update_local_state()` - Sync with main_app state

#### `workflows.py`
**Purpose**: Task execution and results display

**Class**: `WorkflowsFrame(ttk.Frame)`

**External Libraries**:
- `tkinter` (tk, ttk, messagebox, filedialog)
- `logging`
- `textwrap` - Text formatting
- `datetime` - Timestamps

**Internal Imports**:
```python
from .utils import ThreadManager, log_queue, logger
from .logging_handler import setup_gui_logging
```

**Key Attributes**:
- `main_app: MainApp` - Reference to main application
- `thread_manager: ThreadManager` - Thread manager
- `task_entry: tk.Text` - Task input widget
- `output_text: tk.Text` - Output display widget
- `progress: ttk.Progressbar` - Progress bar
- `run_button: ttk.Button` - Run button
- `save_button: ttk.Button` - Save results button
- `last_workflow_result: dict` - Last workflow output
- `workflow_logger: Logger` - Module-specific logger

**Methods**:
- `run_workflow()` - Execute workflow
- `_run_pipeline_thread()` - Worker thread for workflow
- `save_results()` - Export results to file
- `_enable_features()` - Enable when system running
- `_disable_features()` - Disable when system stopped
- `_setup_logging()` - Configure logging to output_text

#### `memory.py`
**Purpose**: Memory and knowledge database browsing

**Classes**:
- `MemoryFrame(ttk.Frame)` - Container with notebook
- `MemorySubFrame(ttk.Frame)` - Individual memory/knowledge browser

**External Libraries**:
- `tkinter` (tk, ttk, messagebox)
- `sqlite3` - Direct database access
- `json` - Data formatting

**Internal Imports**:
```python
from .utils import ThreadManager, DBHelper, logger
```

**Felix Framework Imports** (optional):
```python
from src.memory import knowledge_store, task_memory
```

**Key Attributes** (MemorySubFrame):
- `db_name: str` - Database filename
- `table_name: str` - Table name ('tasks' or 'knowledge')
- `entries: list` - Cached entries [(id, content, display)]
- `listbox: tk.Listbox` - Entry list display
- `view_text: tk.Text` - Details display
- `edit_entry: ttk.Entry` - Edit field
- `query_entry: ttk.Entry` - Search field

**Methods**:
- `load_entries()` - Load from database
- `refresh_entries()` - Reload entries
- `search()` - Search entries
- `delete_entry()` - Delete selected entry
- `on_select()` - Display entry details

#### `agents.py`
**Purpose**: Agent spawning, monitoring, and interaction

**Class**: `AgentsFrame(ttk.Frame)`

**External Libraries**:
- `tkinter` (tk, ttk, messagebox)
- `time`
- `textwrap` - Text formatting

**Internal Imports**:
```python
from .utils import ThreadManager, logger
```

**Felix Framework Imports** (optional):
```python
from ..agents import dynamic_spawning
from ..agents.agent import AgentState
from ..communication import mesh
from ..agents.specialized_agents import ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent
from ..communication.central_post import CentralPost, Message, MessageType
```

**Key Attributes**:
- `main_app: MainApp` - Reference to main application
- `thread_manager: ThreadManager` - Thread manager
- `agents: list` - Active agents list
- `agent_counter: int` - Agent ID counter
- `polling_active: bool` - Polling flag
- `type_combo: ttk.Combobox` - Agent type selector
- `domain_entry: ttk.Entry` - Domain input
- `tree: ttk.Treeview` - Agent list display
- `monitor_text: tk.Text` - Agent details display
- `message_entry: ttk.Entry` - Message input
- `spawn_button: ttk.Button` - Spawn button
- `send_button: ttk.Button` - Send message button

**Methods**:
- `spawn_agent()` - Spawn new agent
- `_spawn_thread()` - Worker thread for spawning
- `send_message()` - Send task to agent
- `_send_thread()` - Worker thread for messaging
- `_update_treeview()` - Refresh agent list
- `_poll_updates()` - Poll every 1.5s for updates
- `_update_agents_from_main()` - Sync with felix_system
- `on_select()` - Display agent details
- `_enable_features()` - Enable when system running
- `_disable_features()` - Disable when system stopped

#### `settings.py`
**Purpose**: Configuration management

**Class**: `SettingsFrame(ttk.Frame)`

**External Libraries**:
- `tkinter` (tk, ttk, messagebox, filedialog)
- `json` - Config file handling
- `os` - File operations
- `logging`
- `typing` (Dict, Any)

**Internal Imports**:
```python
from .utils import logger
```

**Key Attributes**:
- `main_app: MainApp` - Reference to main application
- `thread_manager: ThreadManager` - Thread manager
- `config_file: str` - "felix_gui_config.json"
- `setting_widgets: dict` - Widget references
- `canvas: tk.Canvas` - Scrollable container
- `scrollable_frame: ttk.Frame` - Settings form
- `save_button: ttk.Button` - Save button
- `reset_button: ttk.Button` - Reset button
- `load_button: ttk.Button` - Load button
- `status_label: ttk.Label` - Status message

**Configuration Fields**:
- LM Studio: lm_host, lm_port
- Helix: helix_top_radius, helix_bottom_radius, helix_height, helix_turns
- Agents: max_agents, base_token_budget
- Spawning: confidence_threshold, volatility_threshold, time_window_minutes, token_budget_limit
- Memory: memory_db_path, knowledge_db_path, compression_target_length, compression_ratio, compression_strategy
- Features: enable_metrics, enable_memory, enable_dynamic_spawning, enable_compression, enable_spoke_topology, verbose_llm_logging

**Methods**:
- `save_settings()` - Save to config file
- `load_settings()` - Load from config file
- `load_from_file()` - Load from user-selected file
- `reset_to_defaults()` - Reset all settings
- `validate_settings()` - Validate configuration
- `get_settings_dict()` - Export as dictionary
- `set_settings_dict()` - Import from dictionary
- `_enable_features()` - Disable editing (system running)
- `_disable_features()` - Enable editing (system stopped)

### System Integration Files

#### `felix_system.py`
**Purpose**: Unified system manager for Felix integration

**Classes**:
- `FelixConfig` (dataclass) - System configuration
- `AgentManager` - Agent lifecycle management
- `FelixSystem` - System coordinator

**External Libraries**:
- `logging`
- `asyncio` - Async operations
- `typing` (Optional, Dict, Any, List)
- `dataclasses` (@dataclass)

**Felix Framework Imports**:
```python
from src.core.helix_geometry import HelixGeometry
from src.communication.central_post import CentralPost, AgentFactory, Message, MessageType
from src.communication.spoke import SpokeManager
from src.llm.lm_studio_client import LMStudioClient
from src.llm.token_budget import TokenBudgetManager
from src.memory.knowledge_store import KnowledgeStore
from src.memory.task_memory import TaskMemory
from src.memory.context_compression import ContextCompressor, CompressionConfig, CompressionStrategy, CompressionLevel
from src.agents import ResearchAgent, AnalysisAgent, SynthesisAgent, CriticAgent, PromptOptimizer
from src.agents.agent import AgentState
```

**FelixConfig Fields**:
- lm_host, lm_port
- helix_top_radius, helix_bottom_radius, helix_height, helix_turns
- max_agents, base_token_budget
- memory_db_path, knowledge_db_path
- compression_target_length, compression_ratio, compression_strategy
- Feature flags (enable_*)

**AgentManager**:
- `agents: dict` - agent_id → agent instance
- `agent_outputs: dict` - agent_id → output data
- `register_agent()`, `deregister_agent()`, `get_agent()`, `get_all_agents()`
- `store_agent_output()`, `get_agent_output()`

**FelixSystem**:
- `start()` - Initialize all components
- `stop()` - Shutdown and cleanup
- `spawn_agent()` - Create new agent
- `send_task_to_agent()` - Send task to agent
- `run_workflow()` - Execute workflow
- `get_system_status()` - Get metrics
- `advance_time()` - Update simulation time

### Utility Files

#### `utils.py`
**Purpose**: Threading, database, and logging utilities

**Classes**:
- `ThreadManager` - Thread lifecycle management
- `DBHelper` - Database helper
- `QueueHandler` - Queue-based logging handler

**External Libraries**:
- `threading`
- `queue`
- `sqlite3`
- `logging`
- `os`

**Module-Level Objects**:
- `log_queue: queue.Queue` - Shared log queue
- `logger: Logger` - GUI logger instance ('felix_gui')
- `formatter: Formatter` - Log message formatter

**ThreadManager**:
- `root: tk.Tk` - Root window reference
- `threads: list` - Active threads
- `queue: queue.Queue` - Message queue
- `start_thread(target, args)` - Start daemon thread
- `poll_queue()` - Poll every 100ms
- `join_threads()` - Wait for threads

**DBHelper**:
- `memory_db: str` - 'felix_memory.db'
- `knowledge_db: str` - 'felix_knowledge.db'
- `lock: threading.Lock` - Thread safety
- `ks: KnowledgeStore` - Knowledge store instance
- `tm: TaskMemory` - Task memory instance
- `connect(db_name)` - Create connection
- `query(db_name, sql, params)` - Execute query
- `execute(db_name, sql, params)` - Execute statement
- `get_table_names(db_name)` - List tables

**Logging Setup**:
- File handler → 'felix_gui.log'
- Queue handler → log_queue
- Logger name: 'felix_gui'
- Level: INFO

#### `logging_handler.py`
**Purpose**: Custom logging handlers for Tkinter

**Classes**:
- `TkinterTextHandler(logging.Handler)` - Write to Text widget
- `QueueHandler(logging.Handler)` - Write to queue

**External Libraries**:
- `logging`
- `tkinter` (tk)
- `queue` (Queue)
- `typing` (Optional)

**Functions**:
- `setup_gui_logging(text_widget, log_queue, level, module_name)` - Configure logging
- `add_text_widget_to_logger(logger, text_widget, level)` - Add handler
- `remove_handler(logger, handler)` - Remove handler

**TkinterTextHandler**:
- Thread-safe text widget updates
- Uses `after(0, callback)` for main thread scheduling
- Handles disabled widget states
- Format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

**QueueHandler**:
- Non-blocking queue writes
- Format: '%(asctime)s - %(levelname)s - %(message)s'

**setup_gui_logging**:
- Creates module-specific logger (not root)
- Disables propagation to avoid conflicts
- Configures Felix module loggers
- Returns configured logger

## GUI-Specific Data Files

### Configuration
- **felix_gui_config.json** - Application settings
  - Location: Project root
  - Format: JSON
  - Loaded by: MainApp
  - Managed by: SettingsFrame

### Logging
- **felix_gui.log** - Application log file
  - Location: Project root
  - Handler: FileHandler in utils.py
  - Format: '%(asctime)s - %(levelname)s - %(message)s'

## Internal Dependencies

### Dependency Graph
```
main.py
├── utils.py (ThreadManager, DBHelper, logger)
├── dashboard.py
│   └── utils.py (log_queue, logger)
├── workflows.py
│   ├── utils.py (ThreadManager, logger)
│   └── logging_handler.py (setup_gui_logging)
├── memory.py
│   └── utils.py (ThreadManager, DBHelper, logger)
├── agents.py
│   └── utils.py (ThreadManager, logger)
├── settings.py
│   └── utils.py (logger)
└── felix_system.py
    └── (Felix framework imports only)

__init__.py (exports all)
└── main.py, felix_system.py, dashboard.py, workflows.py, memory.py, agents.py, utils.py, logging_handler.py

__main__.py
└── main.py
```

### Cross-Frame Communication
All frames communicate through `main_app` reference:
- `main_app.felix_system` - Shared system instance
- `main_app.system_running` - Shared state flag
- `main_app.start_system()` / `stop_system()` - System control
- `main_app.memory_frame` - Direct frame access for refresh

## External Library Usage by File

### Tkinter (GUI Framework)
- **All frame files**: tk, ttk
- **dashboard.py**: scrolledtext, messagebox
- **workflows.py**: messagebox, filedialog
- **memory.py**: messagebox
- **agents.py**: messagebox
- **settings.py**: messagebox, filedialog
- **main.py**: messagebox
- **logging_handler.py**: tk (Text widget)

### Threading & Concurrency
- **utils.py**: threading, queue
- **main.py**: (uses ThreadManager)
- **All frame files**: (uses ThreadManager)

### Database
- **utils.py**: sqlite3
- **memory.py**: sqlite3

### Other Standard Library
- **main.py, settings.py**: json, os
- **workflows.py, agents.py**: textwrap
- **workflows.py**: datetime
- **felix_system.py**: asyncio, dataclasses
- **All files**: logging, typing

## GUI Architecture Patterns

### Main Thread Safety
All GUI updates use `after(0, callback)` to execute on main thread:
```python
self.after(0, lambda: self._write_output(message))
```

### Worker Thread Pattern
Long-running operations use ThreadManager:
```python
self.thread_manager.start_thread(self._worker_thread, args=(param,))
```

### State Synchronization
Frames poll or use callbacks to sync with main_app:
```python
if self.main_app and self.main_app.system_running:
    # Do something
```

### Feature Enablement
Frames implement `_enable_features()` and `_disable_features()`:
```python
def _enable_features(self):
    """Enable controls when system running."""
    self.run_button.config(state=tk.NORMAL)

def _disable_features(self):
    """Disable controls when system stopped."""
    self.run_button.config(state=tk.DISABLED)
```

### Logging Architecture
- Module-specific loggers avoid conflicts
- Text widgets receive logs via TkinterTextHandler
- Queue-based logging for async delivery
- Polling (100ms) for log_queue updates

## Widget Types Used

### Input Widgets
- `ttk.Entry` - Single-line text input
- `tk.Text` - Multi-line text input/display
- `ttk.Combobox` - Dropdown selection
- `ttk.Checkbutton` - Boolean toggle
- `ttk.Button` - Action buttons

### Display Widgets
- `tk.Text` - Output display, logs, monitoring
- `scrolledtext.ScrolledText` - Auto-scrolling text
- `ttk.Treeview` - Tabular data (agents)
- `tk.Listbox` - List display (memory entries)
- `ttk.Label` - Static text
- `ttk.Progressbar` - Progress indication

### Layout Widgets
- `ttk.Frame` - Container
- `ttk.Notebook` - Tabbed interface
- `tk.Canvas` - Scrollable content
- `ttk.Scrollbar` - Scrolling
- `ttk.Separator` - Visual separation

### Dialogs
- `messagebox.showinfo/showwarning/showerror/askyesno` - User notifications
- `filedialog.askopenfilename/asksaveasfilename` - File selection

## GUI Event Handling

### Button Callbacks
- Direct method references: `command=self.method_name`
- Lambda wrappers: `command=lambda: self.method(arg)`

### Widget Events
- `<<TreeviewSelect>>` - Treeview selection
- `<<ListboxSelect>>` - Listbox selection
- `<Configure>` - Widget resize

### Polling Loops
- `poll_log_queue()` - 100ms intervals
- `_poll_updates()` - 1500ms intervals (agents)
- `_poll_system_ready()` - 250ms intervals
- `poll_queue()` - 100ms intervals (ThreadManager)

## Configuration Management

### Load Sequence
1. MainApp loads felix_gui_config.json on startup
2. SettingsFrame displays current values
3. User modifies settings
4. Save button writes to felix_gui_config.json
5. Restart system to apply changes

### Validation
SettingsFrame validates before save:
- Numeric ranges (0.0-1.0, 1-133, etc.)
- Port range (1-65535)
- Helix constraints (top > bottom)
- Type conversions (str → int/float/bool)

### Default Values
```python
defaults = {
    "lm_host": "127.0.0.1",
    "lm_port": "1234",
    "helix_top_radius": "3.0",
    "helix_bottom_radius": "0.5",
    "helix_height": "8.0",
    "helix_turns": "2.0",
    "max_agents": "25",
    "base_token_budget": "2500",
    # ... (see settings.py)
}
```

## GUI Module Summary

**Total Files**: 11 Python files
- 1 init file (`__init__.py`)
- 1 entry point (`__main__.py`)
- 1 main app (`main.py`)
- 5 tab frames (`dashboard.py`, `workflows.py`, `memory.py`, `agents.py`, `settings.py`)
- 1 system manager (`felix_system.py`)
- 2 utilities (`utils.py`, `logging_handler.py`)

**Lines of Code** (approximate):
- main.py: ~220
- dashboard.py: ~140
- workflows.py: ~360
- memory.py: ~320
- agents.py: ~500
- settings.py: ~450
- felix_system.py: ~620
- utils.py: ~100
- logging_handler.py: ~190
- **Total**: ~2,900 lines

**External Dependencies**:
- tkinter (standard library)
- threading, queue, sqlite3, json, os, logging, asyncio, datetime, textwrap
- No third-party packages required for GUI itself

**Felix Framework Integration**:
- felix_system.py is the single integration point
- Other GUI files only import from felix_system.py or optionally check for Felix modules
- Clean separation between GUI and framework
